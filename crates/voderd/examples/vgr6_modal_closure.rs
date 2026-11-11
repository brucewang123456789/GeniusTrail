//! Phase 6: REAL cloud read-only acceptance for the complete trusted VGR path.
//!
//! This is an administrator-run acceptance executable, NOT a user/model tool.
//! It creates a fresh isolated test workspace and an inert proposal, then uses
//! real Core, daemon, Runtime and Modal operations. It does not certify
//! write-action semantics, arbitrary semantic correctness or production budgets.
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voder_core::{
    action_commit::{ActionIntent, PreparedAction},
    budget::{BudgetAmount, BudgetLedger},
    contract::{
        ContractDraft, FrozenContract, RepoScope, Requirement, TaskAnchor, ToolRule,
        ValidatorBinding,
    },
    evidence::EvidenceLedger,
    obligation::{ObligationRegistry, ObligationSpec},
    retry::ActionDescriptor,
    types::{
        ContractIdentity, Digest, EvidenceRelation, Severity, ToolEffect, Version,
        WorkspaceIdentity,
    },
};
use voder_protocol::json::Value;
use voder_runtime::{
    artifact_store::{sha256, ArtifactStore},
    process::ProcessLimits,
    receipt::{self, CaptureContext, WorkspaceSnapshot},
    sandbox::{SandboxAvailability, SandboxBackend, SandboxPolicy},
    sandbox_plugin::PluginSandbox,
    tool_executor::DurableDispatchPermit,
    validator::{ValidatorCommand, ValidatorRegistry},
};
use voderd::{
    journal::Journal,
    native_closure::{self, ClosureContext, MetaAuditResources},
    native_finalization,
    native_trusted_receipt::{self, TrustedReceiptRequest, TrustedReceiptResources},
    native_vgr::{NativePreparation, NativePrepareRequest},
    session::State,
};

const SESSION: &str = "vgr6-accept-session";
const RUN: &str = "vgr6-accept-run";
const ACTION: &str = "vgr6-accept-action";
const VALIDATOR_COMMAND_SHA256: &str =
    "749519b1ca549e08ae19a35c6c53ec4a372859a7728234539e104b9f331a05ae";
const CONTENT: &[u8] = b"VODER_VGR6_REAL_MODAL_READ_ONLY\n";

fn id<T: TryFrom<&'static str>>(s: &'static str) -> T
where
    <T as TryFrom<&'static str>>::Error: std::fmt::Debug,
{
    T::try_from(s).unwrap()
}
fn strv(s: impl Into<String>) -> Value {
    Value::String(s.into())
}
fn event(kind: &str, key: &str, fields: Vec<(&str, Value)>) -> Value {
    let mut values = vec![
        ("type".to_owned(), strv(kind)),
        ("session".to_owned(), strv(SESSION)),
        ("key".to_owned(), strv(key)),
        (
            "fingerprint".to_owned(),
            strv("vgr6-acceptance-test-fixture"),
        ),
        ("response".to_owned(), Value::object([])),
    ];
    values.extend(fields.into_iter().map(|(k, v)| (k.to_owned(), v)));
    Value::object(values)
}
fn contract() -> FrozenContract {
    let digest = Digest::parse(sha256(b"vgr6-pinned-read-only-contract-v1").unwrap()).unwrap();
    FrozenContract::compile(ContractDraft {
        anchor: TaskAnchor {
            task_id: id("vgr6-task"),
            identity: "vgr6-modal-acceptance".into(),
            goal: "Verify the original source file with an exit-zero sandbox validator".into(),
            version: Version::new(1).unwrap(),
            accepted_facts: vec![],
            environment_constraints: vec!["read-only evaluation".into()],
            evaluation_boundary: vec![
                "exit code and exact byte output, not arbitrary correctness".into()
            ],
        },
        identity: ContractIdentity {
            contract_id: id("vgr6-contract"),
            version: Version::new(1).unwrap(),
            digest,
            toolchain: id("modal-sandbox"),
            environment: id("vgr6-accept-env"),
        },
        requirements: vec![Requirement {
            id: id("vgr6-requirement"),
            description: "pinned validator exits zero".into(),
            hard: true,
            severity: Severity::Critical,
        }],
        validators: vec![ValidatorBinding {
            id: id("vgr6-validator"),
            version: id("v1"),
            requirement_ids: vec![id("vgr6-requirement")],
        }],
        trusted_sources: vec![id("vgr6-daemon")],
        tools: vec![ToolRule {
            id: id("vgr6-read"),
            effect: ToolEffect::Read,
            requires_approval: false,
        }],
        read_scopes: vec![RepoScope::parse("src").unwrap()],
        write_scopes: vec![],
        forbidden_effects: vec![],
        required_inputs: vec![],
        sampling_rules: vec![],
    })
    .unwrap()
}
fn obligations(contract: &FrozenContract) -> ObligationRegistry {
    ObligationRegistry::compile(
        contract,
        vec![ObligationSpec {
            id: id("vgr6-obligation"),
            claim_id: "pinned-validator-exit-zero".into(),
            requirement_id: Some(id("vgr6-requirement")),
            threatened_judgement: "delivery".into(),
            expected_observation_type: "validator-process-exit".into(),
            admission_rule_id: "voder-validator-exit-zero-v1".into(),
            scope: RepoScope::parse("src").unwrap(),
            priority: 10,
            stop_condition: "pinned validator exited zero".into(),
            source: id("vgr6-daemon"),
            critical: true,
            closure_relation: EvidenceRelation::Supports,
            dependencies: vec![],
        }],
    )
    .unwrap()
}
fn init_proposal(journal: &mut Journal, root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    journal.append(event(
        "session.created",
        "vgr6-session-created",
        vec![("project_root", strv(root.to_string_lossy().into_owned()))],
    ))?;
    journal.append(event(
        "task.model_requested",
        "vgr6-model-requested",
        vec![
            ("run_id", strv(RUN)),
            ("reason", strv("harness inert fixture")),
            ("instruction_digest", strv(sha256(b"vgr6-read-test")?)),
        ],
    ))?;
    journal.append(event(
        "task.model_proposed",
        "vgr6-model-proposed",
        vec![
            ("run_id", strv(RUN)),
            ("reason", strv("non-authoritative test proposal")),
            ("proposal_digest", strv(sha256(b"vgr6-proposal-fixture")?)),
        ],
    ))?;
    Ok(())
}
fn stage_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let base = env::var_os("VODER_VGR6_EVIDENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir);
    fs::create_dir_all(&base)?;
    let path = base.join(format!(
        "voder-vgr6-real-modal-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&path)?;
    Ok(path)
}
fn check_status(state: &State, expected: &str) -> Result<(), Box<dyn std::error::Error>> {
    let actual = state
        .get(SESSION)?
        .runs
        .get(RUN)
        .ok_or("missing run")?
        .status
        .as_str();
    if actual != expected {
        return Err(format!("status mismatch: {actual} != {expected}").into());
    }
    println!("PHASE6_STATUS={expected}");
    Ok(())
}
fn run(base: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let root = base.join("workspace");
    fs::create_dir_all(root.join("src"))?;
    fs::write(root.join("src/input.txt"), CONTENT)?;
    let root = root.canonicalize()?;
    let journal_path = base.join("events.jsonl");
    let store = ArtifactStore::open(&base.join("artifacts"))?;
    let mut journal = Journal::open(&journal_path)?;
    init_proposal(&mut journal, &root)?;
    let mut state = State::reconstruct(journal.entries())?;
    let mut preparation = NativePreparation::recover(&journal)?;
    let contract = contract();
    let mut obligations = obligations(&contract);
    let mut evidence = EvidenceLedger::new();
    let hard_evidence = vec![(id("vgr6-requirement"), id("vgr6-evidence"))];
    let before = WorkspaceSnapshot::measure(&root)?;
    let workspace = before.workspace(id("vgr6-revision"))?;
    let amount = BudgetAmount {
        steps: 1,
        tool_calls: 1,
        wall_time_ms: 120_000,
        ..BudgetAmount::default()
    };
    let mut budget = BudgetLedger::new(amount);
    budget.reserve(id("vgr6-reservation"), amount)?;
    let action = preparation.prepare(
        &mut journal,
        &mut state,
        NativePrepareRequest {
            session_id: SESSION,
            contract: &contract,
            workspace: &workspace,
            verified_inputs: &[],
            budget: &budget,
            intent: ActionIntent {
                run_id: id(RUN),
                action_id: id(ACTION),
                descriptor: ActionDescriptor {
                    kind: "read-only-probe".into(),
                    target_obligation_id: Some(id("vgr6-obligation")),
                    hypothesis_id: None,
                    tool_id: id("vgr6-read"),
                    parameter_key: id("vgr6-fixed-args"),
                    scope: RepoScope::parse("src").unwrap(),
                },
                contract: contract.identity().clone(),
                workspace: workspace.clone(),
                reservation_id: id("vgr6-reservation"),
                expected_changes: vec![],
                required_validators: vec![],
                rollback_ref: None,
            },
            approval: None,
        },
    )?;
    check_status(&state, "VGR_PREPARED")?;
    if native_closure::final_commit(
        &mut journal,
        &mut state,
        &closure_context(
            &root,
            &action,
            &contract,
            &workspace,
            &obligations,
            &evidence,
            &hard_evidence,
        ),
    )
    .is_ok()
    {
        return Err("early final commit erroneously accepted".into());
    }
    preparation.dispatch(&mut journal, &mut state, SESSION, &action)?;
    let _dispatch_guard =
        DurableDispatchPermit::from_durable_journal(&journal_path, SESSION, &action)?;
    check_status(&state, "VGR_DISPATCHED")?;

    let node = PathBuf::from(env::var("VODER_NODE_BINARY")?);
    let cli = PathBuf::from(env::var("VODER_MODAL_CLI")?);
    let backend = PluginSandbox::new(node, cli, "modal")?;
    if !matches!(backend.probe(), SandboxAvailability::Available) {
        return Err("Modal sandbox unavailable".into());
    }
    let policy = SandboxPolicy::new(
        root.clone(),
        ProcessLimits::new(Duration::from_secs(120), 32 * 1024, 512 * 1024 * 1024, 120)?,
    );
    let process = backend.execute(
        &policy,
        "/usr/bin/cat",
        &["src/input.txt".into()],
        &base.join("original-capture"),
    )?;
    if process.exit_code() != Some(0) || process.timed_out() || process.output_limited() {
        return Err("real Modal action did not exit cleanly".into());
    }
    let receipt = receipt::capture(CaptureContext {
        run: id(RUN),
        action: id(ACTION),
        contract: contract.identity().clone(),
        prepared_before: workspace.clone(),
        snapshot: &before,
        process: &process,
        store: &store,
    })?;
    if store.read(&receipt.stdout_artifact)? != CONTENT
        || !store.read(&receipt.stderr_artifact)?.is_empty()
    {
        return Err("real Modal output differs from exact pinned fixture".into());
    }
    println!("PHASE6_REAL_MODAL_ACTION=PASS");
    preparation.observe(&mut journal, &mut state, SESSION, &action, &receipt, &store)?;
    check_status(&state, "VGR_OBSERVED")?;
    native_finalization::settle_observed_action(
        &mut journal,
        &mut state,
        SESSION,
        &action,
        &mut budget,
        BudgetAmount {
            steps: 1,
            tool_calls: 1,
            wall_time_ms: u64::try_from(receipt.elapsed_millis)?,
            ..BudgetAmount::default()
        },
    )?;
    check_status(&state, "VGR_BUDGET_SETTLED")?;
    let after = receipt.report.after.clone();
    let mut validators = ValidatorRegistry::default();
    let command = ValidatorCommand {
        id: id("vgr6-validator"),
        version: id("v1"),
        program: "/usr/bin/cat".into(),
        args: vec!["src/input.txt".into()],
    };
    let computed = sha256(
        format!(
            "{}\0{}\0{:?}",
            command.program, command.version, command.args
        )
        .as_bytes(),
    )?;
    if computed != VALIDATOR_COMMAND_SHA256 {
        return Err("admin-pinned validator command changed".into());
    }
    validators.register(command)?;
    native_trusted_receipt::execute_verify_and_admit(
        &mut journal,
        &mut state,
        TrustedReceiptRequest {
            session_id: SESSION,
            action: &action,
            contract: &contract,
            observed_workspace: &after,
            obligation_id: id("vgr6-obligation"),
            observation_id: id("vgr6-observation"),
            evidence_id: id("vgr6-evidence"),
            obligations: &mut obligations,
            ledger: &mut evidence,
            expected_validator_command_sha256: VALIDATOR_COMMAND_SHA256,
        },
        &TrustedReceiptResources {
            validators: &validators,
            backend: &backend,
            policy: &policy,
            store: &store,
            capture_dir: &base.join("evidence-capture"),
        },
    )?;
    check_status(&state, "VGR_EVIDENCE_ADMITTED")?;
    println!("PHASE6_CORE_SEALED_EVIDENCE=PASS");
    let closure = closure_context(
        &root,
        &action,
        &contract,
        &after,
        &obligations,
        &evidence,
        &hard_evidence,
    );
    if native_closure::final_commit(&mut journal, &mut state, &closure).is_ok() {
        return Err("final commit succeeded without independent meta-audit".into());
    }
    let audit_root = base.join("meta-audit");
    fs::create_dir(&audit_root)?;
    let audit = native_closure::meta_audit(
        &mut journal,
        &mut state,
        &closure,
        &MetaAuditResources {
            registry: &validators,
            backend: &backend,
            policy: &policy,
            store: &store,
            capture_root: &audit_root,
        },
    )?;
    if audit.len() != 64 {
        return Err("invalid meta-audit identity".into());
    }
    check_status(&state, "VGR_META_AUDITED")?;
    println!("PHASE6_INDEPENDENT_META_AUDIT=PASS");
    // Simulate a process restart in the audited-but-not-committed interval.
    // The recovery state must not grant delivery until final_commit is fsynced.
    let before_commit = Journal::open(&journal_path)?;
    let restored_audit = State::reconstruct(before_commit.entries())?;
    check_status(&restored_audit, "VGR_META_AUDITED")?;
    if restored_audit.get(SESSION)?.runs[RUN]
        .native_final_commit_digest
        .is_some()
    {
        return Err("audited-only recovery granted delivery".into());
    }
    println!("PHASE6_PRECOMMIT_RECOVERY_FAIL_CLOSED=PASS");
    let final_digest = native_closure::final_commit(&mut journal, &mut state, &closure)?;
    check_status(&state, "VGR_COMMITTED")?;
    if final_digest.len() != 64
        || state.get(SESSION)?.runs[RUN]
            .native_final_commit_digest
            .as_deref()
            != Some(final_digest.as_str())
    {
        return Err("final commit digest not durable".into());
    }
    if native_closure::final_commit(&mut journal, &mut state, &closure).is_ok() {
        return Err("duplicate final commit erroneously admitted".into());
    }
    let count = journal.entries().len();
    let disk = Journal::open(&journal_path)?;
    if disk.entries().len() != count {
        return Err("journal event count mismatch after reopening".into());
    }
    let reconstructed = State::reconstruct(disk.entries())?;
    check_status(&reconstructed, "VGR_COMMITTED")?;
    if reconstructed.get(SESSION)?.runs[RUN]
        .native_final_commit_digest
        .as_deref()
        != Some(final_digest.as_str())
    {
        return Err("replay final digest mismatch".into());
    }
    println!("PHASE6_DURABLE_REPLAY=PASS");
    // Corrupt only a *separate copy* of the completed journal. Never modify
    // the original journal or the user's workspace.
    let mut corrupt = fs::read(&journal_path)?;
    let needle = b"VGR_COMMITTED";
    let offset = corrupt
        .windows(needle.len())
        .position(|w| w == needle)
        .ok_or("missing committed marker in journal")?;
    corrupt[offset] = b'X';
    let tampered = base.join("tampered-journal.jsonl");
    fs::write(&tampered, corrupt)?;
    if Journal::open(&tampered).is_ok() {
        return Err("tampered journal replay accepted".into());
    }
    println!("PHASE6_TAMPERED_JOURNAL_REJECTED=PASS");
    println!("VGR_COMPLETE_NATIVE_COMMIT_WITH_MODAL=PASS");
    println!("PHASE6_SCOPE=READ_ONLY_ADMIN_ACCEPTANCE");
    println!("PRODUCTION_COMPLETE=NO");
    Ok(())
}
fn closure_context<'a>(
    root: &'a Path,
    action: &'a PreparedAction,
    contract: &'a FrozenContract,
    workspace: &'a WorkspaceIdentity,
    obligations: &'a ObligationRegistry,
    evidence: &'a EvidenceLedger,
    hard_evidence: &'a [(
        voder_core::types::RequirementId,
        voder_core::types::EvidenceId,
    )],
) -> ClosureContext<'a> {
    ClosureContext {
        session_id: SESSION,
        action,
        contract,
        workspace,
        obligations,
        evidence,
        hard_evidence,
        workspace_root: root,
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::args().nth(1).as_deref() == Some("--self-check") {
        let configured = sha256(b"/usr/bin/cat\0v1\0[\"src/input.txt\"]")?;
        if configured != VALIDATOR_COMMAND_SHA256 {
            return Err("validator command pin mismatch".into());
        }
        let frozen = contract();
        let _obligations = obligations(&frozen);
        println!("PHASE6_STATIC_ACCEPTANCE_CONTRACT=PASS");
        return Ok(());
    }
    let base = stage_dir()?;
    eprintln!("PHASE6_EVIDENCE_ROOT={}", base.display());
    let outcome = run(&base);
    if outcome.is_err() {
        eprintln!("PHASE6_RESULT=FAIL; evidence retained; no auto-retry");
    }
    outcome
}
