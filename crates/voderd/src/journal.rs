//! Append-only, hash-chained, fsynced journal; rejects torn/tampered tails.
//! This authenticates integrity only against accidental edits, not a malicious
//! actor who can rewrite the entire ledger and recompute its hashes.
use crate::{DaemonError, Result};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Seek, SeekFrom, Write},
    path::Path,
};
use voder_protocol::json::{self, Value};
use voder_runtime::artifact_store::sha256;

pub const MAX_EVENT_BYTES: usize = 128 * 1024;
pub const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub seq: u64,
    pub previous: String,
    pub hash: String,
    pub body: Value,
}
pub struct Journal {
    file: File,
    entries: Vec<Entry>,
}
fn crypto(bytes: &[u8]) -> Result<String> {
    sha256(bytes).map_err(|e| DaemonError::Integrity(e.to_string()))
}
fn calc(seq: u64, prev: &str, body: &Value) -> Result<String> {
    crypto(format!("voder-ledger-v1\n{seq}\n{prev}\n{}", body.encode()).as_bytes())
}
fn field<'a>(v: &'a Value, k: &str) -> Result<&'a Value> {
    v.get(k)
        .ok_or(DaemonError::Invalid("journal missing field"))
}
fn parse(line: &[u8]) -> Result<Entry> {
    let v = json::parse(line).map_err(|e| DaemonError::Integrity(e.to_string()))?;
    let Value::Object(ref o) = v else {
        return Err(DaemonError::Invalid("journal must be object"));
    };
    if o.len() != 4 {
        return Err(DaemonError::Invalid("journal field mismatch"));
    }
    let seq = field(&v, "seq")?
        .as_u64()
        .ok_or(DaemonError::Invalid("bad journal sequence"))?;
    let previous = field(&v, "prev")?
        .as_str()
        .ok_or(DaemonError::Invalid("bad predecessor"))?
        .to_owned();
    let hash = field(&v, "hash")?
        .as_str()
        .ok_or(DaemonError::Invalid("bad digest"))?
        .to_owned();
    let body = field(&v, "body")?.clone();
    if !matches!(body, Value::Object(_)) {
        return Err(DaemonError::Invalid("bad journal payload"));
    }
    if hash.len() != 64 || previous.len() != 64 {
        return Err(DaemonError::Invalid("invalid digest length"));
    }
    Ok(Entry {
        seq,
        previous,
        hash,
        body,
    })
}
impl Journal {
    pub fn open(path: &Path) -> Result<Self> {
        if let Ok(meta) = fs::symlink_metadata(path) {
            if !meta.file_type().is_file() || meta.nlink_guard() {
                return Err(DaemonError::Invalid("journal is not plain regular file"));
            }
        }
        let created = !path.exists();
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
            opts.custom_flags(0o400000);
        }
        let mut file = opts.open(path)?;
        if created {
            file.sync_all()?;
            File::open(
                path.parent()
                    .ok_or(DaemonError::Invalid("missing parent"))?,
            )?
            .sync_all()?;
        }
        let mut input = BufReader::new(file.try_clone()?);
        let mut entries: Vec<Entry> = Vec::new();
        loop {
            let mut line = Vec::new();
            let read = read_bounded_line(&mut input, &mut line)?;
            if read == 0 {
                break;
            }
            if read > MAX_EVENT_BYTES || line.last() != Some(&b'\n') {
                return Err(DaemonError::Integrity(
                    "oversized or torn journal record".into(),
                ));
            }
            line.pop();
            let e = parse(&line)?;
            let previous = entries.last().map_or(ZERO_HASH, |v| v.hash.as_str());
            let next = entries
                .last()
                .map_or(Some(1), |v| v.seq.checked_add(1))
                .ok_or(DaemonError::Invalid("sequence overflow"))?;
            if e.seq != next || e.previous != previous || calc(e.seq, previous, &e.body)? != e.hash
            {
                return Err(DaemonError::Integrity(format!(
                    "hash chain failure at event {}",
                    e.seq
                )));
            }
            entries.push(e);
        }
        file.seek(SeekFrom::End(0))?;
        Ok(Self { file, entries })
    }
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }
    pub fn append(&mut self, body: Value) -> Result<Entry> {
        if !matches!(body, Value::Object(_)) {
            return Err(DaemonError::Invalid("body must be object"));
        }
        let seq = self
            .entries
            .last()
            .map_or(Some(1), |v| v.seq.checked_add(1))
            .ok_or(DaemonError::Invalid("sequence overflow"))?;
        let previous = self
            .entries
            .last()
            .map_or(ZERO_HASH, |v| v.hash.as_str())
            .to_owned();
        let hash = calc(seq, &previous, &body)?;
        let row = Value::object([
            ("seq".into(), Value::Number(seq.to_string())),
            ("prev".into(), Value::String(previous.clone())),
            ("hash".into(), Value::String(hash.clone())),
            ("body".into(), body.clone()),
        ]);
        let line = format!("{}\n", row.encode());
        if line.len() > MAX_EVENT_BYTES {
            return Err(DaemonError::Invalid("journal event exceeds cap"));
        }
        // One writer holds OS lock; on partial IO or fsync failure poison process by
        // returning error; caller MUST stop, never append a second event.
        self.file.write_all(line.as_bytes())?;
        self.file.sync_all()?;
        let entry = Entry {
            seq,
            previous,
            hash,
            body,
        };
        self.entries.push(entry.clone());
        Ok(entry)
    }
}

fn read_bounded_line(reader: &mut impl BufRead, output: &mut Vec<u8>) -> Result<usize> {
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            return Ok(output.len());
        }
        let end = buf
            .iter()
            .position(|b| *b == b'\n')
            .map(|p| p + 1)
            .unwrap_or(buf.len());
        if output.len().saturating_add(end) > MAX_EVENT_BYTES {
            return Err(DaemonError::Integrity("oversized journal record".into()));
        }
        let done = buf[end - 1] == b'\n';
        output.extend_from_slice(&buf[..end]);
        reader.consume(end);
        if done {
            return Ok(output.len());
        }
    }
}

// POSIX nlink check prevents hardlink aliases to the journal.
trait LinkGuard {
    fn nlink_guard(&self) -> bool;
}
impl LinkGuard for fs::Metadata {
    fn nlink_guard(&self) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            self.nlink() != 1
        }
        #[cfg(not(unix))]
        {
            false
        }
    }
}
