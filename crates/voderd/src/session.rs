//! State is derived exclusively from verified journal events, never wire hints.
use crate::{journal::Entry, DaemonError, Result};
use std::collections::{BTreeMap, HashMap};
use voder_protocol::json::Value;
use voder_runtime::artifact_store::sha256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run {
    pub id: String,
    pub status: String,
    pub reason: String,
    pub instruction_digest: String,
    pub proposal_digest: Option<String>,
    pub native_action_id: Option<String>,
    pub native_reservation_id: Option<String>,
    pub native_contract_digest: Option<String>,
    pub native_workspace_digest: Option<String>,
    pub native_receipt_digest: Option<String>,
    pub native_observed_workspace_digest: Option<String>,
    pub native_runtime_completed: Option<bool>,
    /// Durable budget settlement fingerprint; only set after observed runtime receipt.
    pub native_budget_digest: Option<String>,
    pub native_evidence_ids: std::collections::BTreeSet<String>,
    /// Only a replay-verified independent meta-audit can set this.
    pub native_meta_audit_digest: Option<String>,
    /// A replay-verified final commit is the sole delivery authority.
    pub native_final_commit_digest: Option<String>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Session {
    pub id: String,
    pub project_root: String,
    pub runs: BTreeMap<String, Run>,
    pub last_sequence: u64,
}
#[derive(Clone, Debug)]
pub struct Dedup {
    pub fingerprint: String,
    pub response: Value,
}
#[derive(Default)]
pub struct State {
    pub sessions: BTreeMap<String, Session>,
    pub dedup: HashMap<(String, String), Dedup>,
}
fn string<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
    v.get(k)
        .and_then(Value::as_str)
        .ok_or(DaemonError::Invalid("invalid event field"))
}
impl State {
    pub fn reconstruct(entries: &[Entry]) -> Result<Self> {
        let mut state = Self::default();
        for e in entries {
            state.apply(e)?;
        }
        Ok(state)
    }
    pub fn apply(&mut self, e: &Entry) -> Result<()> {
        let b = &e.body;
        let kind = string(b, "type")?;
        let session = string(b, "session")?;
        let key = string(b, "key")?;
        let request_hash = string(b, "fingerprint")?;
        let response = b
            .get("response")
            .ok_or(DaemonError::Invalid("missing response"))?
            .clone();
        let dedup_key = (
            if kind == "session.created" {
                String::new()
            } else {
                session.to_owned()
            },
            key.to_owned(),
        );
        if self.dedup.contains_key(&dedup_key) {
            return Err(DaemonError::Integrity(
                "repeated persisted idempotency key".into(),
            ));
        }
        if kind == "session.created" {
            if self.sessions.contains_key(session) {
                return Err(DaemonError::Integrity("duplicate session id".into()));
            }
            self.sessions.insert(
                session.into(),
                Session {
                    id: session.into(),
                    project_root: string(b, "project_root")?.into(),
                    runs: BTreeMap::new(),
                    last_sequence: e.seq,
                },
            );
        } else {
            let record = self
                .sessions
                .get_mut(session)
                .ok_or(DaemonError::Integrity("unknown session in event".into()))?;
            match kind {
                "task.blocked"
                | "task.cancelled"
                | "task.model_requested"
                | "task.model_proposed"
                | "task.model_failed" => {
                    let id = string(b, "run_id")?;
                    let reason = string(b, "reason")?;
                    if kind == "task.blocked" || kind == "task.model_requested" {
                        if record.runs.contains_key(id) {
                            return Err(DaemonError::Integrity("duplicate run".into()));
                        }
                        record.runs.insert(
                            id.into(),
                            Run {
                                id: id.into(),
                                status: if kind == "task.model_requested" {
                                    "MODEL_PENDING"
                                } else {
                                    "BLOCKED"
                                }
                                .into(),
                                reason: reason.into(),
                                instruction_digest: string(b, "instruction_digest")?.into(),
                                proposal_digest: None,
                                native_action_id: None,
                                native_reservation_id: None,
                                native_contract_digest: None,
                                native_workspace_digest: None,
                                native_receipt_digest: None,
                                native_observed_workspace_digest: None,
                                native_runtime_completed: None,
                                native_budget_digest: None,
                                native_evidence_ids: std::collections::BTreeSet::new(),
                                native_meta_audit_digest: None,
                                native_final_commit_digest: None,
                            },
                        );
                    } else {
                        let r = record
                            .runs
                            .get_mut(id)
                            .ok_or(DaemonError::Integrity("cancel unknown run".into()))?;
                        if kind == "task.cancelled" {
                            if !matches!(
                                r.status.as_str(),
                                "MODEL_PENDING" | "PROPOSED" | "BLOCKED"
                            ) {
                                return Err(DaemonError::Integrity(
                                    "cannot cancel native or completed run".into(),
                                ));
                            }
                            r.status = "CANCELLED".into();
                        } else {
                            if r.status != "MODEL_PENDING" {
                                return Err(DaemonError::Integrity(
                                    "model completion without pending request".into(),
                                ));
                            }
                            if kind == "task.model_proposed" {
                                r.status = "PROPOSED".into();
                                r.proposal_digest = Some(string(b, "proposal_digest")?.into());
                            } else {
                                r.status = "BLOCKED".into();
                            }
                        }
                        r.reason = reason.into();
                    }
                }
                "vgr.action_prepared" => {
                    let id = string(b, "run_id")?;
                    let run = record.runs.get_mut(id).ok_or(DaemonError::Integrity(
                        "preparing native VGR action for unknown run".into(),
                    ))?;
                    if run.status != "PROPOSED"
                        || run.proposal_digest.is_none()
                        || string(b, "action_id")?.is_empty()
                        || string(b, "reservation_id")?.is_empty()
                    {
                        return Err(DaemonError::Integrity(
                            "native VGR prepare without stored proposal".into(),
                        ));
                    }
                    if b.get("response")
                        .and_then(|v| v.get("execution_authorized"))
                        != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(false))
                    {
                        return Err(DaemonError::Integrity(
                            "native VGR preparation must not convey execution authority".into(),
                        ));
                    }
                    if string(b, "proposal_digest")? != run.proposal_digest.as_deref().unwrap_or("")
                        || string(b, "contract_digest")?.is_empty()
                        || string(b, "workspace_digest")?.is_empty()
                    {
                        return Err(DaemonError::Integrity(
                            "VGR prepare identity not bound".into(),
                        ));
                    }
                    run.native_action_id = Some(string(b, "action_id")?.into());
                    run.native_reservation_id = Some(string(b, "reservation_id")?.into());
                    run.native_contract_digest = Some(string(b, "contract_digest")?.into());
                    run.native_workspace_digest = Some(string(b, "workspace_digest")?.into());
                    run.status = "VGR_PREPARED".into();
                    run.reason = "prepared intent; no executable permit".into();
                }
                "vgr.action_dispatched" => {
                    let id = string(b, "run_id")?;
                    let run = record.runs.get_mut(id).ok_or(DaemonError::Integrity(
                        "dispatch for unknown VGR run".into(),
                    ))?;
                    if run.status != "VGR_PREPARED"
                        || run.native_action_id.as_deref() != Some(string(b, "action_id")?)
                        || run.native_reservation_id.as_deref()
                            != Some(string(b, "reservation_id")?)
                        || run.native_contract_digest.as_deref()
                            != Some(string(b, "contract_digest")?)
                        || run.native_workspace_digest.as_deref()
                            != Some(string(b, "workspace_digest")?)
                        || b.get("response")
                            .and_then(|v| v.get("execution_authorized"))
                            != Some(&Value::Bool(true))
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(false))
                    {
                        return Err(DaemonError::Integrity(
                            "invalid VGR dispatch transition".into(),
                        ));
                    }
                    run.status = "VGR_DISPATCHED".into();
                    run.reason = "durable one-step dispatch; external outcome unknown".into();
                }
                "vgr.action_observed" => {
                    let id = string(b, "run_id")?;
                    let run = record.runs.get_mut(id).ok_or(DaemonError::Integrity(
                        "observation for unknown VGR run".into(),
                    ))?;
                    if run.status != "VGR_DISPATCHED"
                        || run.native_action_id.as_deref() != Some(string(b, "action_id")?)
                        || run.native_contract_digest.as_deref()
                            != Some(string(b, "contract_digest")?)
                        || run.native_workspace_digest.as_deref()
                            != Some(string(b, "workspace_digest")?)
                        || string(b, "receipt_digest")?.is_empty()
                        || b.get("response")
                            .and_then(|v| v.get("execution_authorized"))
                            != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("evidence_admitted"))
                            != Some(&Value::Bool(false))
                    {
                        return Err(DaemonError::Integrity(
                            "invalid VGR observation transition".into(),
                        ));
                    }
                    run.native_receipt_digest = Some(string(b, "receipt_digest")?.into());
                    run.native_observed_workspace_digest = b
                        .get("observed_workspace_digest")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    run.native_runtime_completed = match b.get("runtime_completed") {
                        Some(Value::Bool(value)) => Some(*value),
                        _ => None,
                    };
                    run.status = "VGR_OBSERVED".into();
                    run.reason =
                        "measured receipt stored; independent evidence not admitted".into();
                }
                "vgr.budget_settled" => {
                    let id = string(b, "run_id")?;
                    let run = record.runs.get_mut(id).ok_or(DaemonError::Integrity(
                        "budget settlement for unknown VGR run".into(),
                    ))?;
                    if run.status != "VGR_OBSERVED"
                        || run.native_action_id.as_deref() != Some(string(b, "action_id")?)
                        || run.native_reservation_id.as_deref()
                            != Some(string(b, "reservation_id")?)
                        || run.native_receipt_digest.as_deref()
                            != Some(string(b, "receipt_digest")?)
                        || run.native_contract_digest.as_deref()
                            != Some(string(b, "contract_digest")?)
                        || run.native_workspace_digest.as_deref()
                            != Some(string(b, "workspace_digest")?)
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(false))
                        || b.get("response")
                            .and_then(|v| v.get("execution_authorized"))
                            != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("evidence_admitted"))
                            != Some(&Value::Bool(false))
                    {
                        return Err(DaemonError::Integrity(
                            "invalid VGR budget settlement".into(),
                        ));
                    }
                    let amount = b
                        .get("actual")
                        .ok_or(DaemonError::Invalid("missing actual usage"))?;
                    let upper = b
                        .get("reserved_upper")
                        .ok_or(DaemonError::Invalid("missing reserved budget"))?;
                    let spent = b
                        .get("budget_spent")
                        .ok_or(DaemonError::Invalid("missing spent budget"))?;
                    let reserved = b
                        .get("budget_reserved")
                        .ok_or(DaemonError::Invalid("missing remaining reserved budget"))?;
                    let budget_parts = [amount, upper, spent, reserved];
                    let mut vectors = [[0u64; 8]; 4];
                    for (index, part) in budget_parts.iter().enumerate() {
                        for (field_index, field) in [
                            "steps",
                            "tokens",
                            "model_calls",
                            "tool_calls",
                            "wall_time_ms",
                            "risk_microunits",
                            "context_tokens",
                            "api_cost_microusd",
                        ]
                        .iter()
                        .enumerate()
                        {
                            vectors[index][field_index] =
                                part.get(field)
                                    .and_then(Value::as_u64)
                                    .ok_or(DaemonError::Invalid("invalid budget field"))?;
                        }
                    }
                    for (used, limit) in vectors[0].iter().zip(vectors[1]) {
                        if *used > limit {
                            return Err(DaemonError::Integrity(
                                "budget settlement exceeds reservation".into(),
                            ));
                        }
                    }
                    let digest = string(b, "budget_digest")?;
                    if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit()) {
                        return Err(DaemonError::Integrity(
                            "bad budget settlement digest".into(),
                        ));
                    }
                    let canonical = format!(
                        "vgr-budget-v1\0{}\0{}\0{}\0{}\0{:?}\0{:?}\0{:?}\0{:?}",
                        session,
                        id,
                        string(b, "action_id")?,
                        string(b, "receipt_digest")?,
                        vectors[1],
                        vectors[0],
                        vectors[2],
                        vectors[3],
                    );
                    if sha256(canonical.as_bytes())
                        .map_err(|e| DaemonError::Integrity(e.to_string()))?
                        != digest
                    {
                        return Err(DaemonError::Integrity(
                            "budget settlement digest mismatch".into(),
                        ));
                    }
                    run.native_budget_digest = Some(digest.to_owned());
                    run.status = "VGR_BUDGET_SETTLED".into();
                    run.reason =
                        "bounded measured usage recorded; Core evidence and final commit pending"
                            .into();
                }
                "vgr.evidence_admitted" => {
                    let id = string(b, "run_id")?;
                    let run = record
                        .runs
                        .get_mut(id)
                        .ok_or(DaemonError::Integrity("unknown evidence run".into()))?;
                    if !matches!(
                        run.status.as_str(),
                        "VGR_BUDGET_SETTLED" | "VGR_EVIDENCE_ADMITTED"
                    ) || run.native_action_id.as_deref() != Some(string(b, "action_id")?)
                        || run.native_receipt_digest.as_deref()
                            != Some(string(b, "parent_receipt_digest")?)
                        || run.native_contract_digest.as_deref()
                            != Some(string(b, "contract_digest")?)
                        || run.native_observed_workspace_digest.as_deref()
                            != Some(string(b, "evidence_workspace_digest")?)
                        || b.get("response")
                            .and_then(|v| v.get("execution_authorized"))
                            != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("evidence_admitted"))
                            != Some(&Value::Bool(true))
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(false))
                    {
                        return Err(DaemonError::Integrity(
                            "invalid native evidence transition".into(),
                        ));
                    }
                    let evidence_id = string(b, "evidence_id")?;
                    let content_digest = string(b, "content_digest")?;
                    if evidence_id.is_empty()
                        || content_digest.is_empty()
                        || !run.native_evidence_ids.insert(evidence_id.to_owned())
                    {
                        return Err(DaemonError::Integrity(
                            "duplicate or empty native evidence".into(),
                        ));
                    }
                    run.status = "VGR_EVIDENCE_ADMITTED".into();
                    run.reason =
                        "Core sealed evidence admitted; independent meta-audit pending".into();
                }
                "vgr.meta_audited" => {
                    let id = string(b, "run_id")?;
                    let run = record.runs.get_mut(id).ok_or(DaemonError::Integrity(
                        "meta audit for unknown VGR run".into(),
                    ))?;
                    if run.status != "VGR_EVIDENCE_ADMITTED"
                        || run.native_action_id.as_deref() != Some(string(b, "action_id")?)
                        || run.native_contract_digest.as_deref()
                            != Some(string(b, "contract_digest")?)
                        || run.native_observed_workspace_digest.as_deref()
                            != Some(string(b, "audited_workspace_digest")?)
                        || run.native_receipt_digest.as_deref()
                            != Some(string(b, "parent_receipt_digest")?)
                        || run.native_budget_digest.as_deref() != Some(string(b, "budget_digest")?)
                        || run.native_runtime_completed != Some(true)
                        || run.native_evidence_ids.is_empty()
                        || b.get("response")
                            .and_then(|v| v.get("execution_authorized"))
                            != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(false))
                        || b.get("response")
                            .and_then(|v| v.get("status"))
                            .and_then(Value::as_str)
                            != Some("VGR_META_AUDITED")
                        || b.get("response")
                            .and_then(|v| v.get("run_id"))
                            .and_then(Value::as_str)
                            != Some(id)
                    {
                        return Err(DaemonError::Integrity(
                            "invalid independent meta audit transition".into(),
                        ));
                    }
                    let digest = crate::native_closure::verify_meta_event(
                        session,
                        b,
                        &run.native_evidence_ids,
                    )?;
                    run.native_meta_audit_digest = Some(digest);
                    run.status = "VGR_META_AUDITED".into();
                    run.reason =
                        "trusted validator meta-audit recorded; final commit pending".into();
                }
                "vgr.final_committed" => {
                    let id = string(b, "run_id")?;
                    let run = record.runs.get_mut(id).ok_or(DaemonError::Integrity(
                        "final commit for unknown run".into(),
                    ))?;
                    if run.status != "VGR_META_AUDITED"
                        || run.native_action_id.as_deref() != Some(string(b, "action_id")?)
                        || run.native_contract_digest.as_deref()
                            != Some(string(b, "contract_digest")?)
                        || run.native_observed_workspace_digest.as_deref()
                            != Some(string(b, "committed_workspace_digest")?)
                        || run.native_meta_audit_digest.as_deref()
                            != Some(string(b, "audit_digest")?)
                        || b.get("response")
                            .and_then(|v| v.get("execution_authorized"))
                            != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(true))
                        || b.get("response")
                            .and_then(|v| v.get("status"))
                            .and_then(Value::as_str)
                            != Some("VGR_COMMITTED")
                        || b.get("response")
                            .and_then(|v| v.get("run_id"))
                            .and_then(Value::as_str)
                            != Some(id)
                    {
                        return Err(DaemonError::Integrity(
                            "invalid native final commit transition".into(),
                        ));
                    }
                    let canonical = format!(
                        "voder-vgr4-commit-v1\0{}\0{}\0{}\0{}\0{}\0{}",
                        session,
                        id,
                        string(b, "action_id")?,
                        string(b, "contract_digest")?,
                        string(b, "committed_workspace_digest")?,
                        string(b, "audit_digest")?
                    );
                    let digest = sha256(canonical.as_bytes())
                        .map_err(|e| DaemonError::Integrity(e.to_string()))?;
                    if digest != string(b, "commit_digest")? {
                        return Err(DaemonError::Integrity(
                            "final commit digest mismatch".into(),
                        ));
                    }
                    run.native_final_commit_digest = Some(digest);
                    run.status = "VGR_COMMITTED".into();
                    run.reason = "journaled final commit after independent meta-audit".into();
                }
                "vgr.write_reconciled" => {
                    let id = string(b, "run_id")?;
                    let run = record.runs.get_mut(id).ok_or(DaemonError::Integrity(
                        "reconciliation for unknown run".into(),
                    ))?;
                    if !matches!(
                        run.status.as_str(),
                        "VGR_DISPATCHED"
                            | "VGR_OBSERVED"
                            | "VGR_BUDGET_SETTLED"
                            | "VGR_EVIDENCE_ADMITTED"
                            | "VGR_META_AUDITED"
                    ) || run.native_action_id.as_deref() != Some(string(b, "action_id")?)
                        || run.native_contract_digest.as_deref()
                            != Some(string(b, "contract_digest")?)
                        || run.native_workspace_digest.as_deref()
                            != Some(string(b, "restored_workspace_digest")?)
                        || !matches!(
                            string(b, "recovery_outcome")?,
                            "ROLLED_BACK" | "ALREADY_RESTORED"
                        )
                        || b.get("response")
                            .and_then(|v| v.get("execution_authorized"))
                            != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(false))
                        || b.get("response")
                            .and_then(|v| v.get("status"))
                            .and_then(Value::as_str)
                            != Some("VGR_RECOVERED_ROLLBACK")
                    {
                        return Err(DaemonError::Integrity(
                            "invalid durable write reconciliation".into(),
                        ));
                    }
                    let canonical = format!(
                        "voder-interrupted-write-reconcile-v1\0{}\0{}\0{}\0{}\0{}",
                        session,
                        id,
                        string(b, "action_id")?,
                        string(b, "restored_workspace_digest")?,
                        string(b, "recovery_outcome")?,
                    );
                    if voder_runtime::artifact_store::sha256(canonical.as_bytes())
                        .map_err(|e| DaemonError::Integrity(e.to_string()))?
                        != string(b, "fingerprint")?
                    {
                        return Err(DaemonError::Integrity(
                            "reconciliation identity digest mismatch".into(),
                        ));
                    }
                    run.status = "VGR_RECOVERED_ROLLBACK".into();
                    run.reason = "interrupted write rolled back; no delivery authority".into();
                }
                "vgr.action_rolled_back" => {
                    let id = string(b, "run_id")?;
                    let run = record
                        .runs
                        .get_mut(id)
                        .ok_or(DaemonError::Integrity("unknown rollback run".into()))?;
                    if !matches!(
                        run.status.as_str(),
                        "VGR_BUDGET_SETTLED" | "VGR_EVIDENCE_ADMITTED"
                    ) || run.native_action_id.as_deref() != Some(string(b, "action_id")?)
                        || run.native_contract_digest.as_deref()
                            != Some(string(b, "contract_digest")?)
                        || run.native_workspace_digest.as_deref()
                            != Some(string(b, "restored_workspace_digest")?)
                        || b.get("response")
                            .and_then(|v| v.get("execution_authorized"))
                            != Some(&Value::Bool(false))
                        || b.get("response").and_then(|v| v.get("delivery_allowed"))
                            != Some(&Value::Bool(false))
                    {
                        return Err(DaemonError::Integrity(
                            "invalid native rollback transition".into(),
                        ));
                    }
                    run.status = "VGR_ROLLED_BACK".into();
                    run.reason = "original workspace digest independently remeasured".into();
                }
                _ => {
                    return Err(DaemonError::Integrity(
                        "unrecognized authoritative event".into(),
                    ))
                }
            }
            record.last_sequence = e.seq;
        }
        self.dedup.insert(
            dedup_key,
            Dedup {
                fingerprint: request_hash.into(),
                response,
            },
        );
        Ok(())
    }
    pub fn get(&self, id: &str) -> Result<&Session> {
        self.sessions
            .get(id)
            .ok_or(DaemonError::Invalid("unknown session"))
    }
}
