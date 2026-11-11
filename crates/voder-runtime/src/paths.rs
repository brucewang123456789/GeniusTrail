//! Lexical and symlink-safe existing-path validation.
//! Final workspace confinement must additionally be enforced by OS namespace mounts.
use crate::{Result, RuntimeError};
use std::path::{Component, Path, PathBuf};

pub fn relative_path(input: &str) -> Result<PathBuf> {
    if input.is_empty()
        || input.len() > 4096
        || input.contains('\0')
        || input.contains('\\')
        || input.contains(':')
        || input.chars().any(char::is_control)
    {
        return Err(RuntimeError::UnsafePath(input.into()));
    }
    let p = Path::new(input);
    if p.is_absolute()
        || input.split('/').any(|x| {
            x == ".."
                || x == "."
                || x.is_empty()
                || x == ".git"
                || x == ".voder"
                || x.ends_with('.')
                || x.ends_with(' ')
        })
    {
        return Err(RuntimeError::UnsafePath(input.into()));
    }
    if p.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(RuntimeError::UnsafePath(input.into()));
    }
    Ok(p.to_path_buf())
}

pub fn checked_existing(root: &Path, relative: &str) -> Result<PathBuf> {
    let rel = relative_path(relative)?;
    let canonical_root = root.canonicalize()?;
    let mut current = canonical_root.clone();
    for part in rel.components() {
        current.push(part.as_os_str());
        if std::fs::symlink_metadata(&current)?
            .file_type()
            .is_symlink()
        {
            return Err(RuntimeError::UnsafePath(relative.into()));
        }
    }
    if !current.canonicalize()?.starts_with(&canonical_root) {
        return Err(RuntimeError::UnsafePath(relative.into()));
    }
    Ok(current)
}

pub fn checked_destination(root: &Path, relative: &str) -> Result<PathBuf> {
    let rel = relative_path(relative)?;
    let canonical_root = root.canonicalize()?;
    let mut current = canonical_root.clone();
    let components: Vec<_> = rel.components().collect();
    for (i, part) in components.iter().enumerate() {
        current.push(part.as_os_str());
        match std::fs::symlink_metadata(&current) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err(RuntimeError::UnsafePath(relative.into()))
            }
            Ok(m) if i + 1 != components.len() && !m.is_dir() => {
                return Err(RuntimeError::UnsafePath(relative.into()))
            }
            Ok(_) => (),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if i + 1 != components.len() {
                    return Err(RuntimeError::UnsafePath("missing parent directory".into()));
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(current)
}
