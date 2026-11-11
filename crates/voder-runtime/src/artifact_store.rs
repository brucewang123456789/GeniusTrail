//! Content-addressed evidence; never puts large diagnostic bytes into model context by default.
use crate::{Result, RuntimeError};
use std::{
    fs::{self, File},
    io::{Read, Write, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// SHA-256 is delegated to the pinned system crypto tool, never a noncryptographic hasher.
pub fn sha256(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("/usr/bin/sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| RuntimeError::Integrity("hash stdin unavailable".into()))?
        .write_all(bytes)?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(RuntimeError::Integrity("sha256sum failed".into()));
    }
    let s = String::from_utf8(out.stdout)
        .map_err(|_| RuntimeError::Integrity("invalid hash encoding".into()))?;
    let digest = s.split_whitespace().next().unwrap_or("");
    if digest.len() != 64 || !digest.bytes().all(|x| x.is_ascii_hexdigit()) {
        return Err(RuntimeError::Integrity("bad sha256 result".into()));
    }
    Ok(digest.to_ascii_lowercase())
}

/// Hash a file without buffering its complete contents in the runtime.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut options = fs::OpenOptions::new(); options.read(true);
    #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt; options.custom_flags(0o400000 | 0o4000); }
    let file=options.open(path)?; hash_open_file(&file)
}
fn hash_open_file(file: &File) -> Result<String> {
    let before=file.metadata()?;
    if !before.is_file() { return Err(RuntimeError::UnsafePath("hash requires regular file".into())); }
    let after_handle=file.try_clone()?;
    let out=Command::new("/usr/bin/sha256sum").stdin(Stdio::from(file.try_clone()?)).stdout(Stdio::piped()).stderr(Stdio::null()).output()?;
    let after=after_handle.metadata()?;
    if !out.status.success() || before.len()!=after.len() || before.modified()?!=after.modified()? {
        return Err(RuntimeError::Integrity("file changed while hashing".into()));
    }
    let text=String::from_utf8(out.stdout).map_err(|_|RuntimeError::Integrity("invalid hash output".into()))?;
    let digest=text.split_whitespace().next().unwrap_or("");
    if digest.len()!=64 || !digest.bytes().all(|b|b.is_ascii_hexdigit()) { return Err(RuntimeError::Integrity("invalid file digest".into())); }
    Ok(digest.to_ascii_lowercase())
}

#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root: PathBuf,
}
impl ArtifactStore {
    pub fn open(root: &Path) -> Result<Self> {
        if let Ok(meta) = fs::symlink_metadata(root) {
            if !meta.is_dir() || meta.file_type().is_symlink() {
                return Err(RuntimeError::UnsafePath(
                    "artifact store is not a real directory".into(),
                ));
            }
        }
        fs::create_dir_all(root)?;
        let meta = fs::symlink_metadata(root)?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err(RuntimeError::UnsafePath(
                "artifact store is not a real directory".into(),
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self {
            root: root.canonicalize()?,
        })
    }
    pub fn save(&self, bytes: &[u8]) -> Result<String> {
        let id = sha256(bytes)?;
        let dst = self.root.join(&id);
        if dst.exists() {
            if self.read(&id)? != bytes {
                return Err(RuntimeError::Integrity(
                    "digest collision or corrupt object".into(),
                ));
            }
            return Ok(id);
        }
        // Never stream bytes directly into a published content-addressed
        // filename. A crash during write would leave a corrupt permanent
        // object at its final digest and make recovery impossible.
        // Stage in the same directory, fsync, then publish without overwrite.
        let mut nonce = [0u8; 16];
        File::open("/dev/urandom")?.read_exact(&mut nonce)?;
        let temp_name: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
        let tmp = self.root.join(format!(".voder-artifact-{temp_name}.tmp"));
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
            opts.custom_flags(0o400000); // Linux O_NOFOLLOW
        }
        let mut f = opts.open(&tmp)?;
        let result = (|| -> Result<()> {
            f.write_all(bytes)?;
            f.sync_all()?;
            drop(f);
            match fs::hard_link(&tmp, &dst) {
                Ok(()) => {
                    File::open(&self.root)?.sync_all()?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if self.read(&id)? != bytes {
                        return Err(RuntimeError::Integrity(
                            "conflicting artifact at same digest".into(),
                        ));
                    }
                }
                Err(e) => return Err(e.into()),
            }
            Ok(())
        })();
        // The temporary file belongs solely to this invocation. Never remove
        // any other object, including another writer's staged file.
        match result {
            Ok(()) => {
                fs::remove_file(&tmp)?;
                File::open(&self.root)?.sync_all()?;
                Ok(id)
            }
            Err(error) => {
                // A failed publish has no authority to remove the digest
                // path: it may already belong to a concurrent successful writer.
                let _ = fs::remove_file(&tmp);
                Err(error)
            }
        }
    }
    pub fn read(&self, id: &str) -> Result<Vec<u8>> {
        if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(RuntimeError::UnsafePath("invalid artifact id".into()));
        }
        let p = self.root.join(id);
        if fs::symlink_metadata(&p)?.file_type().is_symlink() {
            return Err(RuntimeError::UnsafePath("artifact symlink".into()));
        }
        let data = fs::read(p)?;
        if sha256(&data)? != id {
            return Err(RuntimeError::Integrity("artifact digest mismatch".into()));
        }
        Ok(data)
    }
    pub fn page(&self, id: &str, offset: usize, limit: usize) -> Result<Vec<u8>> {
        if limit == 0 || limit > 64 * 1024 {
            return Err(RuntimeError::Invalid("invalid page length".into()));
        }
        if id.len()!=64 || !id.bytes().all(|b|b.is_ascii_hexdigit()) { return Err(RuntimeError::UnsafePath("invalid artifact id".into())); }
        let p=self.root.join(id);let mut opts=fs::OpenOptions::new();opts.read(true);
        #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt; opts.custom_flags(0o400000 | 0o4000); }
        let mut file=opts.open(&p)?;let before=file.metadata()?;
        if hash_open_file(&file)?!=id { return Err(RuntimeError::Integrity("artifact digest mismatch".into())); }
        file.seek(SeekFrom::Start(u64::try_from(offset).map_err(|_|RuntimeError::Invalid("invalid artifact offset".into()))?))?;
        let mut page=Vec::with_capacity(limit);(&mut file).take(limit as u64).read_to_end(&mut page)?;
        let after=file.metadata()?;
        if before.len()!=after.len() || before.modified()?!=after.modified()? { return Err(RuntimeError::Integrity("artifact changed while paging".into())); }
        Ok(page)
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use std::{
        sync::Arc,
        thread,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn directory() -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("voder-atomic-store-{}-{n}", std::process::id()));
        fs::create_dir(&p).unwrap();
        p
    }
    #[test]
    fn file_flex_artifacts_above_previous_64mib_ceiling_preserve_digest() {
        let root=directory();let store=ArtifactStore::open(&root).unwrap();let bytes=vec![b'x';64*1024*1024+1];
        let digest=store.save(&bytes).unwrap();assert_eq!(sha256_file(&root.join(&digest)).unwrap(),digest);
        assert_eq!(store.page(&digest,bytes.len()-5,64).unwrap(),b"xxxxx");assert_eq!(store.page(&digest,bytes.len()+10,64).unwrap(),b"");assert_eq!(store.read(&digest).unwrap(),bytes);fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_identical_object_publishing_never_exposes_partial_bytes() {
        let root = directory();
        let store = Arc::new(ArtifactStore::open(&root).unwrap());
        let content = vec![b'x'; 1024 * 1024];
        let mut workers = Vec::new();
        for _ in 0..8 {
            let s = store.clone();
            let input = content.clone();
            workers.push(thread::spawn(move || s.save(&input).unwrap()));
        }
        let ids: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert!(ids.iter().all(|id| id == &ids[0]));
        assert_eq!(store.read(&ids[0]).unwrap(), content);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn partial_or_corrupt_published_object_is_never_accepted() {
        let root = directory();
        let store = ArtifactStore::open(&root).unwrap();
        let id = sha256(b"expected").unwrap();
        fs::write(root.join(&id), b"partial").unwrap();
        assert!(store.save(b"expected").is_err());
        assert_eq!(fs::read(root.join(&id)).unwrap(), b"partial");
        fs::remove_dir_all(root).unwrap();
    }
}
