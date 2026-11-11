//! Only Voder-owned detached worktrees are created and removed; never delete the user's repo.
use crate::{
    paths,
    repository::{git, Repository},
    Result, RuntimeError,
};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[derive(Debug)]
pub struct OwnedWorktree {
    path: PathBuf,
    repository: PathBuf,
    owned: bool,
}
impl OwnedWorktree {
    pub fn create(repo: &Repository, workspace_root: &Path, run_name: &str) -> Result<Self> {
        let rel = paths::relative_path(run_name)?;
        if rel.components().count() != 1 {
            return Err(RuntimeError::UnsafePath(
                "run id must be one component".into(),
            ));
        }
        if !repo.clean()? {
            return Err(RuntimeError::Invalid(
                "dirty repository: preserve user changes".into(),
            ));
        }
        fs::create_dir_all(workspace_root)?;
        let base = workspace_root.canonicalize()?;
        if base.starts_with(repo.root()) || repo.root().starts_with(&base) {
            return Err(RuntimeError::UnsafePath(
                "workspace root overlaps repository".into(),
            ));
        }
        let path = base.join(&rel);
        if path.exists() || fs::symlink_metadata(&path).is_ok() {
            return Err(RuntimeError::UnsafePath(
                "existing workspace cannot be claimed".into(),
            ));
        }
        // Git enforces HEAD attachment and repository origin.
        let out = std::process::Command::new("/usr/bin/git")
            .args(["-c", "core.hooksPath=/dev/null"])
            .arg("-C")
            .arg(repo.root())
            .args(["worktree", "add", "--detach", "--"])
            .arg(&path)
            .arg(&repo.head)
            .output()?;
        if !out.status.success() {
            return Err(RuntimeError::Execution(
                String::from_utf8_lossy(&out.stderr).into(),
            ));
        }
        let canonical = path.canonicalize()?;
        if !canonical.starts_with(&base) {
            return Err(RuntimeError::UnsafePath(
                "worktree escaped allocated directory".into(),
            ));
        }
        Ok(Self {
            path: canonical,
            repository: repo.root().to_path_buf(),
            owned: true,
        })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn finish(mut self) -> Result<()> {
        if !self.owned || !self.path.is_dir() {
            return Err(RuntimeError::Unauthorized("not an owned worktree"));
        }
        // No rm -rf: Git owns its own detached worktree cleanup.
        let out = std::process::Command::new("/usr/bin/git")
            .args(["-c", "core.hooksPath=/dev/null"])
            .arg("-C")
            .arg(&self.repository)
            .args(["worktree", "remove", "--force", "--"])
            .arg(&self.path)
            .output()?;
        if !out.status.success() {
            return Err(RuntimeError::Execution(
                String::from_utf8_lossy(&out.stderr).into(),
            ));
        }
        self.owned = false;
        Ok(())
    }
    pub fn git_diff(&self) -> Result<String> {
        git(&self.path, &["diff", "--no-ext-diff", "--binary"])
    }
}
