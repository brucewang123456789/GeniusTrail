//! Non-authoritative execution adapters. Never makes delivery or evidence decisions.
//! A caller must supply core PreparedAction plus durable pre-dispatch authorization.
pub mod artifact_store;
pub mod diagnostics;
mod dispatch_proof;
pub mod durable_write;
pub mod environment;
pub mod patch;
pub mod paths;
pub mod process;
pub mod receipt;
pub mod repository;
pub mod rollback;
pub mod sandbox;
pub mod sandbox_bwrap;
pub mod sandbox_plugin;
pub mod tool_executor;
pub mod tool_registry;
pub mod validator;
pub mod worktree;

use std::fmt;
#[derive(Debug)]
pub enum RuntimeError {
    Io(std::io::Error),
    UnsafePath(String),
    Unauthorized(&'static str),
    Invalid(String),
    CapabilityUnavailable(&'static str),
    Execution(String),
    Integrity(String),
}
impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RuntimeError {}
impl From<std::io::Error> for RuntimeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
pub type Result<T> = std::result::Result<T, RuntimeError>;
