//! End-to-end durable journal replay gates, including adversarial events.
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use voder_protocol::json::Value;
use voder_runtime::artifact_store::sha256;
use voderd::{journal::Journal, recovery, session::State};

fn temporary() -> PathBuf {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("voder-vgr3-{}-{time}", std::process::id()));
    fs::create_dir(&root).unwrap();
    root
}
fn strv(s: &str) -> Value {
    Value::String(s.to_owned())
}
fn event(kind: &str, key: &str, fields: &[(&str, Value)], response: Value) -> Value {
    let mut parts = vec![
        ("type".to_owned(), strv(kind)),
        ("session".to_owned(), strv("s-1")),
        ("key".to_owned(), strv(key)),
        ("fingerprint".to_owned(), strv("test-fixture")),
        ("response".to_owned(), response),
    ];
    parts.extend(fields.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Value::object(parts)
}
fn field(key: &'static str, value: &'static str) -> (&'static str, Value) {
    (key, strv(value))
}
fn deny() -> Value {
    Value::object([
        ("execution_authorized".into(), Value::Bool(false)),
        ("delivery_allowed".into(), Value::Bool(false)),
        ("evidence_admitted".into(), Value::Bool(false)),
    ])
}
fn setup(j: &mut Journal) {
    j.append(event(
        "session.created",
        "create",
        &[field("project_root", "/tmp/code")],
        Value::object([]),
    ))
    .unwrap();
    j.append(event(
        "task.model_requested",
        "model",
        &[
            field("run_id", "r-1"),
            field("reason", "provider"),
            field("instruction_digest", "req"),
        ],
        Value::object([]),
    ))
    .unwrap();
    j.append(event(
        "task.model_proposed",
        "proposal",
        &[
            field("run_id", "r-1"),
            field("reason", "inert"),
            field("proposal_digest", "proposal"),
        ],
        Value::object([]),
    ))
    .unwrap();
    j.append(event(
        "vgr.action_prepared",
        "prepared",
        &[
            field("run_id", "r-1"),
            field("action_id", "a-1"),
            field("reservation_id", "reserve-1"),
            field("proposal_digest", "proposal"),
            field("contract_digest", "contract"),
            field("workspace_digest", "before"),
        ],
        deny(),
    ))
    .unwrap();
    j.append(event(
        "vgr.action_dispatched",
        "dispatch",
        &[
            field("run_id", "r-1"),
            field("action_id", "a-1"),
            field("reservation_id", "reserve-1"),
            field("contract_digest", "contract"),
            field("workspace_digest", "before"),
        ],
        Value::object([
            ("execution_authorized".into(), Value::Bool(true)),
            ("delivery_allowed".into(), Value::Bool(false)),
        ]),
    ))
    .unwrap();
    j.append(event(
        "vgr.action_observed",
        "observed",
        &[
            field("run_id", "r-1"),
            field("action_id", "a-1"),
            field("contract_digest", "contract"),
            field("workspace_digest", "before"),
            field("observed_workspace_digest", "after"),
            field("receipt_digest", "receipt"),
            ("runtime_completed", Value::Bool(true)),
        ],
        deny(),
    ))
    .unwrap();
}
fn amount(steps: u64) -> Value {
    Value::object([
        ("steps".into(), Value::Number(steps.to_string())),
        ("tokens".into(), Value::Number("0".into())),
        ("model_calls".into(), Value::Number("0".into())),
        ("tool_calls".into(), Value::Number("0".into())),
        ("wall_time_ms".into(), Value::Number("0".into())),
        ("risk_microunits".into(), Value::Number("0".into())),
        ("context_tokens".into(), Value::Number("0".into())),
        ("api_cost_microusd".into(), Value::Number("0".into())),
    ])
}
fn settled(used: u64, upper: u64, valid_digest: bool) -> Value {
    let identity = vec![
        field("run_id", "r-1"),
        field("action_id", "a-1"),
        field("reservation_id", "reserve-1"),
        field("contract_digest", "contract"),
        field("workspace_digest", "before"),
        field("receipt_digest", "receipt"),
    ];
    let mut fields = identity;
    fields.push(("actual", amount(used)));
    fields.push(("reserved_upper", amount(upper)));
    fields.push(("budget_spent", amount(used)));
    fields.push(("budget_reserved", amount(0)));
    let arr = |x: u64| [x, 0, 0, 0, 0, 0, 0, 0];
    let canonical = format!(
        "vgr-budget-v1\0s-1\0r-1\0a-1\0receipt\0{:?}\0{:?}\0{:?}\0{:?}",
        arr(upper),
        arr(used),
        arr(used),
        arr(0)
    );
    let digest = if valid_digest {
        sha256(canonical.as_bytes()).unwrap()
    } else {
        "0".repeat(64)
    };
    fields.push(("budget_digest", strv(&digest)));
    event("vgr.budget_settled", "settled", &fields, deny())
}
#[test]
fn settlement_replays_but_does_not_grant_delivery_or_automatic_restart() {
    let dir = temporary();
    let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
    setup(&mut journal);
    journal.append(settled(1, 2, true)).unwrap();
    let state = State::reconstruct(journal.entries()).unwrap();
    let run = &state.get("s-1").unwrap().runs["r-1"];
    assert_eq!(run.status, "VGR_BUDGET_SETTLED");
    assert!(run.native_budget_digest.is_some());
    assert!(recovery::recover(&journal).is_err());
    drop(journal);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn inflated_budget_or_tampered_digest_fails_replay() {
    for (actual, upper, valid) in [(3, 2, true), (1, 2, false)] {
        let dir = temporary();
        let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
        setup(&mut journal);
        journal.append(settled(actual, upper, valid)).unwrap();
        assert!(State::reconstruct(journal.entries()).is_err());
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }
}
#[test]
fn settlement_without_observation_and_duplicate_settlement_are_rejected() {
    let dir = temporary();
    let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
    journal
        .append(event(
            "session.created",
            "c",
            &[field("project_root", "/tmp/code")],
            Value::object([]),
        ))
        .unwrap();
    journal.append(settled(1, 2, true)).unwrap();
    assert!(State::reconstruct(journal.entries()).is_err());
    drop(journal);
    fs::remove_dir_all(dir).unwrap();
    let dir = temporary();
    let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
    setup(&mut journal);
    journal.append(settled(1, 2, true)).unwrap();
    let mut second = settled(1, 2, true);
    if let Value::Object(ref mut o) = second {
        o.insert("key".into(), strv("different-key"));
    }
    journal.append(second).unwrap();
    assert!(State::reconstruct(journal.entries()).is_err());
    drop(journal);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn fake_evidence_after_settlement_rejected_if_runtime_receipt_mismatches() {
    let dir = temporary();
    let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
    setup(&mut journal);
    journal.append(settled(1, 2, true)).unwrap();
    journal
        .append(event(
            "vgr.evidence_admitted",
            "evidence",
            &[
                field("run_id", "r-1"),
                field("action_id", "a-1"),
                field("parent_receipt_digest", "FAKE"),
                field("receipt_digest", "validator-receipt"),
                field("contract_digest", "contract"),
                field("evidence_workspace_digest", "after"),
                field("evidence_id", "e-1"),
                field("content_digest", "sha256:x"),
            ],
            Value::object([
                ("execution_authorized".into(), Value::Bool(false)),
                ("evidence_admitted".into(), Value::Bool(true)),
                ("delivery_allowed".into(), Value::Bool(false)),
            ]),
        ))
        .unwrap();
    assert!(State::reconstruct(journal.entries()).is_err());
    drop(journal);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn rollback_requires_observed_and_settled_identity() {
    let dir = temporary();
    let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
    setup(&mut journal);
    journal.append(settled(1, 2, true)).unwrap();
    journal
        .append(event(
            "vgr.action_rolled_back",
            "rollback",
            &[
                field("run_id", "r-1"),
                field("action_id", "a-1"),
                field("contract_digest", "contract"),
                field("restored_workspace_digest", "WRONG"),
            ],
            deny(),
        ))
        .unwrap();
    assert!(State::reconstruct(journal.entries()).is_err());
    drop(journal);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn verified_rollback_is_terminal_and_allows_safe_recovery() {
    let dir = temporary();
    let mut journal = Journal::open(&dir.join("events.jsonl")).unwrap();
    setup(&mut journal);
    journal.append(settled(1, 2, true)).unwrap();
    journal
        .append(event(
            "vgr.action_rolled_back",
            "rollback",
            &[
                field("run_id", "r-1"),
                field("action_id", "a-1"),
                field("contract_digest", "contract"),
                field("restored_workspace_digest", "before"),
            ],
            deny(),
        ))
        .unwrap();
    let state = recovery::recover(&journal).unwrap();
    assert_eq!(
        state.get("s-1").unwrap().runs["r-1"].status,
        "VGR_ROLLED_BACK"
    );
    drop(journal);
    fs::remove_dir_all(dir).unwrap();
}
