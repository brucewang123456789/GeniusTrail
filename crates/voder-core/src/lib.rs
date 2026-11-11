//! Pure evidence-governed domain core; no I/O and no model-side authority.
pub mod budget;
pub mod contract;
pub mod types;

pub mod obligation;
pub mod retry;

pub mod evidence;

pub mod validation;

pub mod action_commit;

pub mod consequence;
pub mod decision;
pub mod transition;
pub mod verdict;
