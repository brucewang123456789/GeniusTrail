//! Verifies a fsynced Daemon journal before allowing ONE trusted Rust dispatch.
//! Does not accept a model-supplied token, boolean or unsigned JSON receipt.
//! Only the Daemon's locked, private journal should be passed by trusted code.
use crate::{artifact_store::sha256, Result, RuntimeError};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use voder_core::action_commit::PreparedAction;
use voder_protocol::json::{self, Value};

const MAX_EVENT: usize = 128 * 1024;
const ZERO: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn fail(message: &'static str) -> RuntimeError {
    RuntimeError::Unauthorized(message)
}
fn field<'a>(v: &'a Value, name: &str) -> Result<&'a str> {
    v.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| fail("incomplete dispatch journal"))
}
fn read_line(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut buf = Vec::new();
    loop {
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            if buf.is_empty() {
                return Ok(None);
            }
            return Err(fail("torn dispatch journal record"));
        }
        let n = bytes
            .iter()
            .position(|x| *x == b'\n')
            .map_or(bytes.len(), |i| i + 1);
        if buf.len().saturating_add(n) > MAX_EVENT {
            return Err(fail("oversized dispatch journal record"));
        }
        let done = bytes[n - 1] == b'\n';
        buf.extend_from_slice(&bytes[..n]);
        reader.consume(n);
        if done {
            buf.pop();
            return Ok(Some(buf));
        }
    }
}
/// Returns only if the *last durable event* grants exactly this Core-prepared
/// action. Hash-chain validation detects accidental corruption, not malicious
/// replacement by a privileged process. The Daemon must hold its OS writer lock.
pub(crate) fn verify(path: &Path, session: &str, a: &PreparedAction) -> Result<()> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(fail("dispatch journal must be plain file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.nlink() != 1 || meta.mode() & 0o077 != 0 {
            return Err(fail("unsafe dispatch journal permissions"));
        }
    }
    let mut reader = BufReader::new(fs::File::open(path)?);
    let mut previous = ZERO.to_owned();
    let mut seq = 0u64;
    let mut prepared = false;
    let mut dispatches = 0u32;
    let mut last_is_target_dispatch = false;
    while let Some(bytes) = read_line(&mut reader)? {
        let v = json::parse(&bytes).map_err(|_| fail("unparseable dispatch journal"))?;
        let current_seq = v
            .get("seq")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("invalid dispatch sequence"))?;
        let body = v.get("body").ok_or_else(|| fail("journal event missing"))?;
        if !matches!(body, Value::Object(_))
            || current_seq
                != seq
                    .checked_add(1)
                    .ok_or_else(|| fail("dispatch sequence overflow"))?
            || field(&v, "prev")? != previous
        {
            return Err(fail("dispatch journal chain mismatch"));
        }
        let digest = sha256(
            format!(
                "voder-ledger-v1\n{current_seq}\n{previous}\n{}",
                body.encode()
            )
            .as_bytes(),
        )?;
        if field(&v, "hash")? != digest {
            return Err(fail("dispatch journal digest mismatch"));
        }
        previous = digest;
        seq = current_seq;
        last_is_target_dispatch = false;
        let kind = field(body, "type")?;
        let related = body.get("run_id").and_then(Value::as_str)
            == Some(a.intent().run_id.as_str())
            && body.get("action_id").and_then(Value::as_str) == Some(a.intent().action_id.as_str());
        if !related {
            continue;
        }
        if (kind == "vgr.action_prepared" || kind == "vgr.action_dispatched")
            && (field(body, "session")? != session
                || field(body, "reservation_id")? != a.intent().reservation_id.as_str()
                || field(body, "contract_digest")? != a.intent().contract.digest.as_str()
                || field(body, "workspace_digest")? != a.intent().workspace.digest.as_str())
        {
            return Err(fail("dispatch does not match prepared Core identities"));
        }
        match kind {
            "vgr.action_prepared" => {
                if prepared || dispatches != 0 {
                    return Err(fail("replayed preparation"));
                }
                if body
                    .get("response")
                    .and_then(|r| r.get("execution_authorized"))
                    != Some(&Value::Bool(false))
                {
                    return Err(fail("forged preparation authority"));
                }
                prepared = true;
            }
            "vgr.action_dispatched" => {
                if !prepared || dispatches != 0 {
                    return Err(fail("replayed dispatch"));
                }
                if body
                    .get("response")
                    .and_then(|r| r.get("execution_authorized"))
                    != Some(&Value::Bool(true))
                    || body.get("response").and_then(|r| r.get("delivery_allowed"))
                        != Some(&Value::Bool(false))
                {
                    return Err(fail("invalid dispatch permission"));
                }
                dispatches += 1;
                last_is_target_dispatch = true;
            }
            _ => {}
        }
    }
    if !last_is_target_dispatch || !prepared || dispatches != 1 {
        return Err(fail("no current durable dispatch for Core-prepared action"));
    }
    Ok(())
}
