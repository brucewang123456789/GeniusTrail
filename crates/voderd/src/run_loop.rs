//! The sole orchestration boundary. At this stage the gate is deliberately
//! fail-closed until a model Provider and verified Runtime dispatch permit exist.
//! We never fabricate receipts, a Core PASS, or a successful repair.
use voder_protocol::json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gate {
    MissingContext,
    ProviderUnavailable,
    SandboxNotCertified,
    ProviderReady,
}
pub struct Plan {
    pub gate: Gate,
    pub reason: &'static str,
}
pub fn evaluate(context_refs: &[String], context_available: impl Fn(&str) -> bool) -> Plan {
    if context_refs.iter().any(|id| !context_available(id)) {
        return Plan {
            gate: Gate::MissingContext,
            reason: "MISSING_CONTEXT: supply referenced context artifact; no tool execution",
        };
    }
    // Both capabilities are required. Provider is not connected in this crate;
    // Runtime sandbox certification also remains independently gated.
    Plan {
        gate: Gate::ProviderUnavailable,
        reason: "PROVIDER_UNAVAILABLE: model adapter and authorized dispatch are not connected",
    }
}
/// Seventh-module boundary: Provider availability admits a model proposal ONLY.
/// It does not authorize tools, sandbox dispatch, VGR commit, or final delivery.
pub fn evaluate_provider(
    context_refs: &[String],
    context_available: impl Fn(&str) -> bool,
    provider_available: bool,
) -> Plan {
    let plan = evaluate(context_refs, context_available);
    if provider_available && plan.gate == Gate::ProviderUnavailable {
        Plan {
            gate: Gate::ProviderReady,
            reason: "MODEL_READY: proposal-only",
        }
    } else {
        plan
    }
}
pub fn blocked_payload(run_id: &str, reason: &str) -> Value {
    Value::object([
        ("run_id".into(), Value::String(run_id.into())),
        ("status".into(), Value::String("BLOCKED".into())),
        ("reason".into(), Value::String(reason.into())),
        ("delivery_allowed".into(), Value::Bool(false)),
    ])
}
