//! Per-file reversible actions for a single-writer owned checkout.
//! This is NOT VEP crash-atomic rollback; recovery after power loss remains for voderd.
use crate::{
    artifact_store::ArtifactStore,
    patch::{self, FilePatch, PatchOperation},
    paths, Result, RuntimeError,
};
use std::{fs, path::Path};
#[derive(Debug, Clone)]
pub struct FileSnapshot {
    pub relative: String,
    pub before: Option<String>,
    pub after: Option<Option<String>>,
}
pub fn snapshot(root: &Path, relative: &str, store: &ArtifactStore) -> Result<FileSnapshot> {
    let p = paths::checked_destination(root, relative)?;
    let before = if p.is_file() {
        Some(store.save(&fs::read(p)?)?)
    } else {
        None
    };
    Ok(FileSnapshot {
        relative: relative.into(),
        before,
        after: None,
    })
}
pub fn mark_after(s: &mut FileSnapshot, root: &Path, store: &ArtifactStore) -> Result<()> {
    let p = paths::checked_destination(root, &s.relative)?;
    s.after = Some(if p.is_file() {
        Some(store.save(&fs::read(p)?)?)
    } else {
        None
    });
    Ok(())
}
pub fn restore(root: &Path, s: &FileSnapshot, store: &ArtifactStore) -> Result<()> {
    let after = s.after.as_ref().ok_or(RuntimeError::Unauthorized(
        "rollback snapshot is not finalized",
    ))?;
    let current = paths::checked_destination(root, &s.relative)?;
    let current = if current.is_file() {
        Some(store.save(&fs::read(current)?)?)
    } else {
        None
    };
    if &current != after {
        return Err(RuntimeError::Integrity(
            "rollback preimage changed; manual recovery required".into(),
        ));
    }
    let operation = match (&s.before, after) {
        (None, None) => return Ok(()),
        (Some(before), None) => PatchOperation::Create {
            new_bytes: store.read(before)?,
        },
        (Some(before), Some(after)) => PatchOperation::Replace {
            old_sha256: after.clone(),
            new_bytes: store.read(before)?,
        },
        (None, Some(after)) => PatchOperation::Delete {
            old_sha256: after.clone(),
        },
    };
    patch::apply(
        root,
        &FilePatch {
            relative: s.relative.clone(),
            operation,
        },
    )?;
    Ok(())
}
