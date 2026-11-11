//! Root-cause evidence packets, with observation and hypothesis explicitly separate.
use crate::{artifact_store::ArtifactStore, Result, RuntimeError};
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosisStatus {
    NeedsInvestigation,
    SupportedByReproduction,
    Unresolved,
}
#[derive(Debug, Clone)]
pub struct DiagnosticPacket {
    pub status: DiagnosisStatus,
    pub stderr_id: String,
    pub stdout_id: String,
    pub excerpt: String,
    pub hypotheses: Vec<String>,
    pub next_experiment: String,
}
pub fn prepare(
    store: &ArtifactStore,
    stdout_id: &str,
    stderr_id: &str,
    hypotheses: Vec<String>,
    next_experiment: String,
) -> Result<DiagnosticPacket> {
    if next_experiment.trim().is_empty() || hypotheses.len() > 16 {
        return Err(RuntimeError::Invalid(
            "diagnosis needs bounded, explicit experiment".into(),
        ));
    }
    let bytes = store.page(stderr_id, 0, 8192)?;
    // Truncation is explicit; do not misrepresent an excerpt as a full stderr transcript.
    let excerpt = String::from_utf8_lossy(&bytes).to_string();
    let _ = store.read(stdout_id)?;
    Ok(DiagnosticPacket {
        status: DiagnosisStatus::NeedsInvestigation,
        stderr_id: stderr_id.into(),
        stdout_id: stdout_id.into(),
        excerpt,
        hypotheses,
        next_experiment,
    })
}
