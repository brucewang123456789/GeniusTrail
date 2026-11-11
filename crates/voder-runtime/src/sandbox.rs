//! Sandbox policy is explicit and fail-closed. No unsafe host execution fallback.
use crate::process::{ProcessLimits, ProcessResult};
use crate::{Result, RuntimeError};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    pub worktree: PathBuf,
    pub timeout: Duration,
    pub memory_bytes: u64,
    pub cpu_seconds: u64,
    pub max_output_bytes: u64,
    pub network: bool,
    /// Explicit administrator-pinned toolchain directories, mounted read-only.
    /// Never supply home directories or credential stores.
    pub(crate) toolchains: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolchainKind {
    RustStable,
    Node,
}
impl SandboxPolicy {
    pub fn new(worktree: PathBuf, limits: ProcessLimits) -> Self {
        Self {
            worktree,
            timeout: limits.timeout,
            memory_bytes: limits.memory_bytes,
            cpu_seconds: limits.cpu_seconds,
            max_output_bytes: limits.max_output_bytes,
            network: false,
            toolchains: Vec::new(),
        }
    }
    /// Explicit deployment-owned directories, never arbitrary user-selected mount roots.
    pub fn add_toolchain(&mut self, tools_root: &Path, kind: ToolchainKind) -> Result<()> {
        if tools_root.file_name().and_then(|v| v.to_str()) != Some(".voder-tools") {
            return Err(RuntimeError::UnsafePath(
                "toolchain mount must originate from .voder-tools".into(),
            ));
        }
        let canonical_root = tools_root.canonicalize()?;
        if std::fs::symlink_metadata(tools_root)?
            .file_type()
            .is_symlink()
        {
            return Err(RuntimeError::UnsafePath("toolchain root symlink".into()));
        }
        let rel = match kind {
            ToolchainKind::RustStable => "rust/rustup/toolchains/stable-x86_64-unknown-linux-gnu",
            ToolchainKind::Node => "node",
        };
        let source = canonical_root.join(rel);
        if !source.is_dir() || std::fs::symlink_metadata(&source)?.file_type().is_symlink() {
            return Err(RuntimeError::CapabilityUnavailable(
                "pinned toolchain unavailable",
            ));
        }
        let source = source.canonicalize()?;
        if !source.starts_with(&canonical_root) {
            return Err(RuntimeError::UnsafePath(
                "toolchain escaped pinned root".into(),
            ));
        }
        if self.toolchains.contains(&source) {
            return Err(RuntimeError::Invalid("duplicate toolchain".into()));
        }
        if self.toolchains.len() >= 2 {
            return Err(RuntimeError::Invalid("toolchain limit".into()));
        }
        self.toolchains.push(source);
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.network {
            return Err(RuntimeError::Unauthorized(
                "network access disabled by default",
            ));
        }
        if !self.worktree.is_dir() || self.worktree.symlink_metadata()?.file_type().is_symlink() {
            return Err(RuntimeError::UnsafePath(
                "sandbox worktree is invalid".into(),
            ));
        }
        if self.toolchains.len() > 3 {
            return Err(RuntimeError::Invalid("too many toolchain mounts".into()));
        }
        for root in &self.toolchains {
            let meta = std::fs::symlink_metadata(root)?;
            if !meta.is_dir() || meta.file_type().is_symlink() {
                return Err(RuntimeError::UnsafePath(
                    "toolchain root must be a real directory".into(),
                ));
            }
            let name = root.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.is_empty()
                || [".ssh", ".aws", ".config", ".cargo", "home", "root"].contains(&name)
            {
                return Err(RuntimeError::UnsafePath("unsafe toolchain mount".into()));
            }
        }
        let _ = self.limits()?;
        Ok(())
    }
    pub fn limits(&self) -> Result<ProcessLimits> {
        ProcessLimits::new(
            self.timeout,
            self.max_output_bytes,
            self.memory_bytes,
            self.cpu_seconds,
        )
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxAvailability {
    Available,
    Unavailable(String),
}
pub(crate) mod private {
    pub trait Sealed {}
}
pub trait SandboxBackend: private::Sealed {
    fn probe(&self) -> SandboxAvailability;
    fn execute(
        &self,
        policy: &SandboxPolicy,
        program: &str,
        args: &[String],
        capture_dir: &Path,
    ) -> Result<ProcessResult>;
}
