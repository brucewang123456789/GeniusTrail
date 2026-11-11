//! A persisted unfinished external side effect is never replayed on restart.
//! Pending model invocations are reconciled once on daemon startup as BLOCKED.
//! We never automatically replay a potentially billable external model call.
use crate::{journal::Journal, session::State, DaemonError, Result};

pub fn recover(journal: &Journal) -> Result<State> {
    let state = State::reconstruct(journal.entries())?;
    for session in state.sessions.values() {
        for run in session.runs.values() {
            if !matches!(
                run.status.as_str(),
                "BLOCKED"
                    | "CANCELLED"
                    | "PROPOSED"
                    | "MODEL_PENDING"
                    | "VGR_ROLLED_BACK"
                    | "VGR_RECOVERED_ROLLBACK"
                    | "VGR_COMMITTED"
            ) {
                return Err(DaemonError::Integrity(
                    "unreconciled run phase; refuse startup".into(),
                ));
            }
        }
    }
    Ok(state)
}
