//! Expected-content patching on an owned checkout; never changes the original repository.
//! Caller must serialize all writes. Hostile concurrent path-races require OS sandbox isolation.
use crate::{artifact_store::sha256, paths, Result, RuntimeError};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};
#[derive(Debug, Clone)]
pub enum PatchOperation {
    Create {
        new_bytes: Vec<u8>,
    },
    Replace {
        old_sha256: String,
        new_bytes: Vec<u8>,
    },
    Delete {
        old_sha256: String,
    },
}
#[derive(Debug, Clone)]
pub struct FilePatch {
    pub relative: String,
    pub operation: PatchOperation,
}
#[derive(Debug, Clone)]
pub struct PatchResult {
    pub relative: String,
    pub old_digest: Option<String>,
    pub new_digest: Option<String>,
}

pub fn apply(root: &Path, patch: &FilePatch) -> Result<PatchResult> {
    let file = paths::checked_destination(root, &patch.relative)?;
    let old = match fs::symlink_metadata(&file) {
        Ok(m) if m.is_file() && !m.file_type().is_symlink() => Some(fs::read(&file)?),
        Ok(_) => {
            return Err(RuntimeError::UnsafePath(
                "special file/symlink patch target".into(),
            ))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    let old_digest = old.as_ref().map(|v| sha256(v)).transpose()?;
    let new = match &patch.operation {
        PatchOperation::Create { new_bytes } if old.is_none() => Some(new_bytes),
        PatchOperation::Replace {
            old_sha256,
            new_bytes,
        } if old_digest.as_ref() == Some(old_sha256) => Some(new_bytes),
        PatchOperation::Delete { old_sha256 } if old_digest.as_ref() == Some(old_sha256) => None,
        _ => {
            return Err(RuntimeError::Integrity(
                "preimage mismatch or unexpected file state".into(),
            ))
        }
    };
    if let Some(bytes) = new {
        let parent = file
            .parent()
            .ok_or_else(|| RuntimeError::UnsafePath("missing parent".into()))?;
        let temp = parent.join(format!(".voder-patch-{}", std::process::id()));
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let result = (|| {
            f.write_all(bytes)?;
            f.sync_all()?;
            Ok::<_, std::io::Error>(())
        })();
        if let Err(e) = result {
            let _ = fs::remove_file(&temp);
            return Err(e.into());
        }
        drop(f);
        if let Err(e) = fs::rename(&temp, &file) {
            let _ = fs::remove_file(&temp);
            return Err(e.into());
        }
    } else {
        fs::remove_file(&file)?;
    }
    Ok(PatchResult {
        relative: patch.relative.clone(),
        old_digest,
        new_digest: new.map(|b| sha256(b)).transpose()?,
    })
}
