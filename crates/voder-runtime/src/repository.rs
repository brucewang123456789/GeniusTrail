//! Repository probing is read-only and rejects dirty inputs for detached checkouts.
use crate::{artifact_store::sha256, paths, Result, RuntimeError};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone)]
pub struct Repository {
    root: PathBuf,
    pub head: String,
}
impl Repository {
    pub fn inspect(path: &Path) -> Result<Self> {
        let root = path.canonicalize()?;
        let out = git(&root, &["rev-parse", "--show-toplevel"])?;
        let canonical = PathBuf::from(out.trim()).canonicalize()?;
        if canonical != root {
            return Err(RuntimeError::Invalid("must use repository toplevel".into()));
        }
        let head = git(&root, &["rev-parse", "--verify", "HEAD"])?;
        if !head.trim().bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(RuntimeError::Integrity("invalid HEAD".into()));
        }
        Ok(Self {
            root,
            head: head.trim().into(),
        })
    }
    pub fn fingerprint(&self) -> Result<String> {
        fingerprint(&self.root)
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn clean(&self) -> Result<bool> {
        Ok(git(
            &self.root,
            &["status", "--porcelain=v1", "--untracked-files=all"],
        )?
        .is_empty())
    }
    pub fn inventory(&self, expected: &[String]) -> Result<Vec<String>> {
        let mut missing = vec![];
        for name in expected {
            paths::relative_path(name)?;
            if paths::checked_existing(&self.root, name).is_err() {
                missing.push(name.clone());
            }
        }
        Ok(missing)
    }
}
pub fn snapshot_files(root: &Path) -> Result<BTreeMap<String, String>> {
    let root = root.canonicalize()?;
    let mut stack = vec![root.clone()];
    let mut entries = BTreeMap::new();
    while let Some(dir) = stack.pop() {
        for child in fs::read_dir(&dir)? {
            let child = child?;
            if dir == root && child.file_name() == ".git" {
                continue;
            }
            let p = child.path();
            let rel = p
                .strip_prefix(&root)
                .map_err(|_| RuntimeError::Integrity("directory escape".into()))?;
            let name = rel.to_string_lossy().to_string();
            let meta = fs::symlink_metadata(&p)?;
            if meta.is_dir() {
                stack.push(p);
            } else if meta.file_type().is_symlink() {
                entries.insert(name, format!("L:{}", fs::read_link(&p)?.to_string_lossy()));
            } else if meta.is_file() {
                entries.insert(name, format!("F:{}", crate::artifact_store::sha256_file(&p)?));
            } else {
                return Err(RuntimeError::UnsafePath("special file in worktree".into()));
            }
            if entries.len() > 100_000 {
                return Err(RuntimeError::Invalid(
                    "workspace snapshot exceeds 100k files".into(),
                ));
            }
        }
    }
    Ok(entries)
}
pub fn hash_snapshot(entries: &BTreeMap<String, String>) -> Result<String> {
    let data = entries
        .iter()
        .map(|(k, v)| format!("{k}\0{v}\n"))
        .collect::<String>();
    sha256(data.as_bytes())
}
pub fn fingerprint(root: &Path) -> Result<String> {
    hash_snapshot(&snapshot_files(root)?)
}

pub fn git(root: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("/usr/bin/git")
        .arg("-c")
        .arg("core.hooksPath=/dev/null")
        .arg("-C")
        .arg(root)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .output()?;
    if !out.status.success() {
        return Err(RuntimeError::Execution(format!(
            "git {:?}: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    String::from_utf8(out.stdout).map_err(|_| RuntimeError::Integrity("git output non-UTF8".into()))
}
