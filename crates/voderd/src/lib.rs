//! Single writer over Voder's durable command/event log. Wire messages carry
//! intentions, never authority. No model or sandbox execution until proven gates exist.
pub mod journal;
pub mod native_closure;
pub mod native_finalization;
pub mod native_trusted_receipt;
pub mod native_vgr;
pub mod native_write_recovery;
pub mod protocol_stdio;
pub mod provider_bridge;
pub mod recovery;
pub mod run_loop;
pub mod session;

use std::{fmt, io};
#[derive(Debug)]
pub enum DaemonError {
    Io(io::Error),
    Invalid(&'static str),
    Integrity(String),
    Conflict(&'static str),
}
impl fmt::Display for DaemonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for DaemonError {}
impl From<io::Error> for DaemonError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
pub type Result<T> = std::result::Result<T, DaemonError>;
