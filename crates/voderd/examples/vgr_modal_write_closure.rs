//! Real Modal native VGR WRITE acceptance: an authorized patch in an isolated git worktree.
//!
//! This is an administrator-run acceptance executable, NOT a user/model tool.
//! It validates write-action provenance and measured bytes, but is not a
//! substitute for crash injection, multi-session or independent cost accounting.
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voder_core::{
    action_commit::{ActionIntent, ExpectedChange, MutationKind, PreparedAction},
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
    patch::{FilePatch, PatchOperation},
    process::ProcessLimits,
    receipt::{self, CaptureContext, WorkspaceSnapshot},
    repository::Repository,
    sandbox::{SandboxAvailability, SandboxBackend, SandboxPolicy},
    sandbox_plugin::PluginSandbox,
    tool_executor::DurableDispatchPermit,
    tool_executor::ToolExecutor,
    tool_registry::{ToolKind, ToolRegistry},
    validator::{ValidatorCommand, ValidatorRegistry},
    worktree::OwnedWorktree,
};
use voderd::{
    journal::Journal,
    native_closure::{self, ClosureContext, MetaAuditResources},
    native_finalization,
    native_trusted_receipt::{self, TrustedReceiptRequest, TrustedReceiptResources},
    native_vgr::{NativePreparation, NativePrepareRequest},
    native_write_recovery::{reconcile_interrupted_write, Reconciliation},
    session::State,
};

const SESSION: &str = "vgr6-accept-session";
const RUN: &str = "vgr6-accept-run";
const ACTION: &str = "vgr6-accept-action";
const VALIDATOR_COMMAND_SHA256: &str =
    "b4f76ab162ad705a804ab4238bc189a0025847aee7043e4e09439b863f0510b5";
const CONTENT: &[u8] = b"VODER_VGR_REAL_MODAL_WRITE_OLD\n";
const NEW_CONTENT: &[u8] = b"VODER_VGR_REAL_MODAL_WRITE_VERIFIED\n";

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
    let digest = Digest::parse(sha256(b"vgr-pinned-write-contract-v1").unwrap()).unwrap();
    FrozenContract::compile(ContractDraft {
        anchor: TaskAnchor {
            task_id: id("vgr6-task"), identity: "vgr6-modal-acceptance".into(),
            goal: "Apply an authorized edit in a detached git worktree and verify exact content".into(),
            version: Version::new(1).unwrap(), accepted_facts: vec![],
            environment_constraints: vec!["detached git worktree; original source untouched".into()],
            evaluation_boundary: vec!["exact pinned changed bytes and verified Modal output".into()],
        },
        identity: ContractIdentity {
            contract_id: id("vgr6-contract"), version: Version::new(1).unwrap(),
            digest, toolchain: id("modal-sandbox"), environment: id("vgr6-accept-env"),
        },
        requirements: vec![Requirement { id: id("vgr6-requirement"), description: "pinned validator exits zero".into(),
            hard: true, severity: Severity::Critical }],
        validators: vec![ValidatorBinding { id: id("vgr6-validator"), version: id("v1"),
            requirement_ids: vec![id("vgr6-requirement")] }],
        trusted_sources: vec![id("vgr6-daemon")],
        tools: vec![ToolRule { id: id("vgr6-write"), effect: ToolEffect::Write, requires_approval: false }],
        read_scopes: vec![RepoScope::parse("src").unwrap()],
        write_scopes: vec![RepoScope::parse("src").unwrap()], forbidden_effects: vec![], required_inputs: vec![], sampling_rules: vec![],
    }).unwrap()
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
// An operator-only fault barrier. The acceptance runner observes a durable
// marker and sends SIGKILL to this child; the process does not recover itself.
fn fault_checkpoint(base: &Path, checkpoint: &str) -> Result<(), Box<dyn std::error::Error>> {
    if env::var("VODER_VGR_FAULT_PAUSE_AFTER").ok().as_deref() == Some(checkpoint) {
        let ready = base.join(format!("checkpoint-{checkpoint}.ready"));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(ready)?;
        use std::io::Write;
        file.write_all(checkpoint.as_bytes())?;
        file.sync_all()?;
        std::fs::File::open(base)?.sync_all()?;
        eprintln!("VGR_FAULT_CHECKPOINT={checkpoint}");
        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    }
    Ok(())
}

fn confirmed_recovery(base: &Path, root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let state_dir = base.join("daemon-state");
    let outcome = reconcile_interrupted_write(&state_dir, SESSION, RUN, ACTION)?;
    if outcome != Reconciliation::RolledBack || fs::read(root.join("src/input.txt"))? != CONTENT {
        return Err("recovery failed to restore the exact preimage".into());
    }
    let journal = Journal::open(&state_dir.join("events.jsonl"))?;
    let state = State::reconstruct(journal.entries())?;
    check_status(&state, "VGR_RECOVERED_ROLLBACK")?;
    if state.get(SESSION)?.runs[RUN]
        .native_final_commit_digest
        .is_some()
    {
        return Err("failed action obtained final commit".into());
    }
    if reconcile_interrupted_write(&state_dir, SESSION, RUN, ACTION).is_ok() {
        return Err("duplicate recovery was accepted".into());
    }
    println!("VGR_F07_REAL_MODAL_FAILURE_MEASURED_ROLLBACK=PASS");
    Ok(())
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
    let original = base.join("original-git-repository");
    fs::create_dir_all(original.join("src"))?;
    fs::write(original.join("src/input.txt"), CONTENT)?;
    let command = |argv: &[&str]| -> Result<(), Box<dyn std::error::Error>> {
        let output = std::process::Command::new("/usr/bin/git")
            .arg("-C")
            .arg(&original)
            .args(argv)
            .output()?;
        if !output.status.success() {
            return Err(
                format!("git {argv:?}: {}", String::from_utf8_lossy(&output.stderr)).into(),
            );
        }
        Ok(())
    };
    command(&["init", "-q"])?;
    command(&["add", "src/input.txt"])?;
    command(&[
        "-c",
        "user.name=Voder Acceptance",
        "-c",
        "user.email=acceptance@example.invalid",
        "commit",
        "-qm",
        "baseline",
    ])?;
    let repo = Repository::inspect(&original)?;
    let original_fingerprint = repo.fingerprint()?;
    let worktree =
        OwnedWorktree::create(&repo, &base.join("isolated-worktrees"), "verified-write")?;
    let root = worktree.path().canonicalize()?;
    // Daemon state must be outside the isolated worktree, not its ancestor:
    // trusted rollback forbids overlapping state/workspace roots.
    let state_dir = base.join("daemon-state");
    fs::create_dir(&state_dir)?;
    let journal_path = state_dir.join("events.jsonl");
    let store = ArtifactStore::open(&state_dir.join("artifacts"))?;
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
                    kind: "verified-write-patch".into(),
                    target_obligation_id: Some(id("vgr6-obligation")),
                    hypothesis_id: None,
                    tool_id: id("vgr6-write"),
                    parameter_key: id("vgr6-fixed-write-args"),
                    scope: RepoScope::parse("src").unwrap(),
                },
                contract: contract.identity().clone(),
                workspace: workspace.clone(),
                reservation_id: id("vgr6-reservation"),
                expected_changes: vec![ExpectedChange {
                    path: RepoScope::parse("src/input.txt").unwrap(),
                    kind: MutationKind::Modify,
                }],
                required_validators: vec![id("vgr6-validator")],
                rollback_ref: Some("durable-vgr-write-wal".into()),
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
    let dispatch_guard =
        DurableDispatchPermit::from_durable_journal(&journal_path, SESSION, &action)?;
    check_status(&state, "VGR_DISPATCHED")?;
    let node = PathBuf::from(env::var("VODER_NODE_BINARY")?);
    let cli = PathBuf::from(env::var("VODER_MODAL_CLI")?);
    let backend = PluginSandbox::new(node, cli, "modal")?;
    if !matches!(backend.probe(), SandboxAvailability::Available) {
        return Err("Modal sandbox unavailable".into());
    }
    let mut tools = ToolRegistry::new();
    tools.insert("vgr6-write", ToolKind::Patch, ToolEffect::Write)?;
    let executor = ToolExecutor {
        tools: &tools,
        sandbox: &backend,
        store: &store,
    };
    executor.apply_patch(
        &action,
        &dispatch_guard,
        &root,
        &FilePatch {
            relative: "src/input.txt".into(),
            operation: PatchOperation::Replace {
                old_sha256: sha256(CONTENT)?,
                new_bytes: NEW_CONTENT.to_vec(),
            },
        },
    )?;
    if repo.fingerprint()? != original_fingerprint {
        return Err("original repository was modified".into());
    }
    fault_checkpoint(base, "POST_PATCH")?;
    println!("VGR_LIVE_WRITE_ISOLATION=PASS");
    let policy = SandboxPolicy::new(
        root.clone(),
        ProcessLimits::new(Duration::from_secs(120), 32 * 1024, 512 * 1024 * 1024, 120)?,
    );
    if env::args().nth(1).as_deref() == Some("--fault-remote-timeout") {
        // Enforce an actual remote process deadline inside the pinned image.
        // Exit 124 is observed from the Modal receipt, not inferred from a
        // generic SDK/network/transport error.
        let timed = backend.execute(
            &policy,
            "/usr/bin/timeout",
            &["2s".into(), "/usr/bin/sleep".into(), "30".into()],
            &base.join("remote-timeout-capture"),
        )?;
        if timed.exit_code() != Some(124) || timed.timed_out() || timed.output_limited() {
            return Err("expected real remote /usr/bin/timeout exit code 124".into());
        }
        let timeout_receipt = receipt::capture(CaptureContext {
            run: id(RUN),
            action: id(ACTION),
            contract: contract.identity().clone(),
            prepared_before: workspace.clone(),
            snapshot: &before,
            process: &timed,
            store: &store,
        })?;
        preparation.observe(
            &mut journal,
            &mut state,
            SESSION,
            &action,
            &timeout_receipt,
            &store,
        )?;
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
                wall_time_ms: u64::try_from(timeout_receipt.elapsed_millis)?,
                ..BudgetAmount::default()
            },
        )?;
        check_status(&state, "VGR_BUDGET_SETTLED")?;
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
            return Err("timed-out remote action gained delivery".into());
        }
        println!("VGR_F07_REAL_MODAL_TIMEOUT_EXIT124=PASS");
        confirmed_recovery(base, &root)?;
        if repo.fingerprint()? != original_fingerprint {
            return Err("timeout fault modified original repository".into());
        }
        return Ok(());
    }
    if env::args().nth(1).as_deref() == Some("--fault-remote-exit") {
        let failure = backend.execute(
            &policy,
            "/usr/bin/false",
            &[],
            &base.join("remote-failure-capture"),
        )?;
        if failure.exit_code() != Some(1) || failure.timed_out() || failure.output_limited() {
            return Err("expected real Modal remote nonzero exit=1".into());
        }
        let failed_receipt = receipt::capture(CaptureContext {
            run: id(RUN),
            action: id(ACTION),
            contract: contract.identity().clone(),
            prepared_before: workspace.clone(),
            snapshot: &before,
            process: &failure,
            store: &store,
        })?;
        preparation.observe(
            &mut journal,
            &mut state,
            SESSION,
            &action,
            &failed_receipt,
            &store,
        )?;
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
                wall_time_ms: u64::try_from(failed_receipt.elapsed_millis)?,
                ..BudgetAmount::default()
            },
        )?;
        check_status(&state, "VGR_BUDGET_SETTLED")?;
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
            return Err("failed remote execution gained delivery".into());
        }
        println!("VGR_F07_REAL_MODAL_NONZERO_RECEIPT=PASS");
        confirmed_recovery(base, &root)?;
        if repo.fingerprint()? != original_fingerprint {
            return Err("fault test modified original repository".into());
        }
        return Ok(());
    }
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
    if store.read(&receipt.stdout_artifact)? != NEW_CONTENT
        || !store.read(&receipt.stderr_artifact)?.is_empty()
    {
        return Err("real Modal output differs from exact pinned fixture".into());
    }
    println!("VGR_REAL_MODAL_WRITE_ACTION=PASS");
    preparation.observe(&mut journal, &mut state, SESSION, &action, &receipt, &store)?;
    check_status(&state, "VGR_OBSERVED")?;
    fault_checkpoint(base, "POST_OBSERVED")?;
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
    fault_checkpoint(base, "POST_SETTLED")?;
    let after = receipt.report.after.clone();
    let mut validators = ValidatorRegistry::default();
    let command = ValidatorCommand {
        id: id("vgr6-validator"),
        version: id("v1"),
        program: "/usr/bin/grep".into(),
        args: vec![
            "-Fx".into(),
            "VODER_VGR_REAL_MODAL_WRITE_VERIFIED".into(),
            "src/input.txt".into(),
        ],
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
    fault_checkpoint(base, "POST_ADMITTED")?;
    println!("VGR_WRITE_CORE_SEALED_EVIDENCE=PASS");
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
    fault_checkpoint(base, "POST_META_AUDIT")?;
    println!("VGR_WRITE_INDEPENDENT_META_AUDIT=PASS");
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
    println!("VGR_WRITE_PRECOMMIT_REPLAY_FAIL_CLOSED=PASS");
    let final_digest = native_closure::final_commit(&mut journal, &mut state, &closure)?;
    check_status(&state, "VGR_COMMITTED")?;
    fault_checkpoint(base, "POST_COMMIT")?;
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
    println!("VGR_WRITE_DURABLE_REPLAY=PASS");
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
    println!("VGR_WRITE_TAMPERED_JOURNAL_REJECTED=PASS");
    if repo.fingerprint()? != original_fingerprint {
        return Err("original repository changed after VGR commit".into());
    }
    if fs::read(root.join("src/input.txt"))? != NEW_CONTENT {
        return Err("committed isolated worktree lost verified edit".into());
    }
    println!("VGR_REAL_MODAL_NATIVE_WRITE_COMMIT=PASS");
    println!("VGR_ACCEPTANCE_SCOPE=ISOLATED_WRITE_WITH_PINNED_CONTENT");
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
        let configured = sha256(b"/usr/bin/grep\0v1\0[\"-Fx\", \"VODER_VGR_REAL_MODAL_WRITE_VERIFIED\", \"src/input.txt\"]")?;
        if configured != VALIDATOR_COMMAND_SHA256 {
            return Err("validator command pin mismatch".into());
        }
        let frozen = contract();
        let _obligations = obligations(&frozen);
        println!("VGR_WRITE_STATIC_ACCEPTANCE_CONTRACT=PASS");
        return Ok(());
    }
    let base = stage_dir()?;
    eprintln!("VGR_WRITE_EVIDENCE_ROOT={}", base.display());
    let outcome = run(&base);
    if outcome.is_err() {
        eprintln!("VGR_WRITE_RESULT=FAIL; evidence retained; no auto-retry");
    }
    outcome
}
