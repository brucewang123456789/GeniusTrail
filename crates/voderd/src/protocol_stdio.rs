//! Stdio JSONL server. Single authoritative OS process/lock; all consequence
//! records are durable before clients see a successful acknowledgement.
use crate::{
    journal::Journal,
    native_vgr::{NativePreparation, NativePrepareRequest},
    provider_bridge::ProviderBridge,
    recovery, run_loop,
    session::State,
    DaemonError, Result,
};
use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{BufRead, Read, Write},
    path::Path,
};
use voder_protocol::{
    commands::Command,
    envelope::{Envelope, FrameKind, MAX_FRAME_BYTES},
    errors::{ErrorCode, ProtocolError},
    json::Value,
};
use voder_runtime::{
    artifact_store::{sha256, ArtifactStore},
    repository::Repository,
};

fn error(code: ErrorCode, message: &'static str) -> ProtocolError {
    ProtocolError::new(code, message)
}
fn object(items: impl IntoIterator<Item = (String, Value)>) -> Value {
    Value::object(items)
}
fn strv(s: impl Into<String>) -> Value {
    Value::String(s.into())
}
fn numeric(n: u64) -> Value {
    Value::Number(n.to_string())
}
fn internal(_: impl std::fmt::Display) -> ProtocolError {
    error(
        ErrorCode::Internal,
        "operation rejected; inspect daemon stderr",
    )
}
fn random_id() -> Result<String> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(format!(
        "s-{}",
        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
    ))
}
fn snapshot_request(req: &Envelope) -> voder_protocol::errors::Result<String> {
    let fingerprint = object([
        ("kind".into(), strv(&req.kind)),
        (
            "session".into(),
            strv(req.session_id.clone().unwrap_or_default()),
        ),
        ("payload".into(), req.payload.clone()),
    ]);
    sha256(fingerprint.encode().as_bytes()).map_err(internal)
}
fn entry(
    kind: &str,
    session: &str,
    key: &str,
    fingerprint: &str,
    response: Value,
    fields: Vec<(&str, Value)>,
) -> Value {
    let mut props = vec![
        ("type".into(), strv(kind)),
        ("session".into(), strv(session)),
        ("key".into(), strv(key)),
        ("fingerprint".into(), strv(fingerprint)),
        ("response".into(), response),
    ];
    for (name, v) in fields {
        props.push((name.into(), v));
    }
    object(props)
}

/// Trusted configuration, never deserialized from model/provider payload.
pub struct NativeSandboxCommand<'a> {
    pub session_id: &'a str,
    pub action: &'a voder_core::action_commit::PreparedAction,
    pub executor: &'a voder_runtime::tool_executor::ToolExecutor<'a>,
    pub policy: &'a voder_runtime::sandbox::SandboxPolicy,
    pub program: &'a str,
    pub arguments: &'a [String],
    pub capture_dir: &'a Path,
    pub snapshot: &'a voder_runtime::receipt::WorkspaceSnapshot,
}

pub struct Server {
    journal: Journal,
    state: State,
    native_vgr: NativePreparation,
    store: ArtifactStore,
    provider: Option<ProviderBridge>,
    state_root: std::path::PathBuf,
    lock: File,
    initialized: bool,
    seen: HashSet<String>,
    poisoned: bool,
}
impl Server {
    pub fn open(root: &Path) -> Result<Self> {
        if !root.is_absolute() {
            return Err(DaemonError::Invalid("state path must be absolute"));
        }
        if let Ok(m) = fs::symlink_metadata(root) {
            if !m.file_type().is_dir() {
                return Err(DaemonError::Invalid("state path is not a plain directory"));
            }
        }
        fs::create_dir_all(root)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        }
        let root = root.canonicalize()?;
        let p = root.join("daemon.lock");
        if let Ok(m) = fs::symlink_metadata(&p) {
            if !m.file_type().is_file() {
                return Err(DaemonError::Invalid("lock path is not regular file"));
            }
        }
        let mut options = OpenOptions::new();
        options.create(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
            options.custom_flags(0o400000);
        }
        let lock = options.open(&p)?;
        lock.try_lock()
            .map_err(|_| DaemonError::Conflict("another writer owns this state dir"))?;
        let journal = Journal::open(&root.join("events.jsonl"))?;
        let state = recovery::recover(&journal)?;
        let native_vgr = NativePreparation::recover(&journal)?;
        let provider = ProviderBridge::from_env()?;
        let store = ArtifactStore::open(&root.join("artifacts"))
            .map_err(|e| DaemonError::Integrity(e.to_string()))?;
        let mut server = Self {
            journal,
            state,
            native_vgr,
            store,
            provider,
            state_root: root,
            lock,
            initialized: false,
            seen: HashSet::new(),
            poisoned: false,
        };
        // Reconcile interrupted model calls in the verified journal, never retry silently.
        let interrupted: Vec<(String, String)> = server
            .state
            .sessions
            .values()
            .flat_map(|s| {
                s.runs
                    .values()
                    .filter(|r| r.status == "MODEL_PENDING")
                    .map(move |r| (s.id.clone(), r.id.clone()))
            })
            .collect();
        for (sid, rid) in interrupted {
            let reason = "MODEL_INTERRUPTED: unknown upstream outcome; explicit new task required";
            let blocked = run_loop::blocked_payload(&rid, reason);
            let body = entry(
                "task.model_failed",
                &sid,
                &format!("sys-recovery-{rid}"),
                "recovery",
                blocked,
                vec![("run_id", strv(&rid)), ("reason", strv(reason))],
            );
            server
                .persist(body)
                .map_err(|_| DaemonError::Integrity("recovery journal commit failed".into()))?;
        }
        Ok(server)
    }
    /// Trusted Rust-only VGR preparation. No client protocol command exposes this.
    /// A return value is NOT a dispatch permit or delivery authorization.
    pub fn prepare_native_vgr(
        &mut self,
        request: NativePrepareRequest<'_>,
    ) -> Result<voder_core::action_commit::PreparedAction> {
        if self.poisoned {
            return Err(DaemonError::Conflict("daemon poisoned: recovery required"));
        }
        let result = self
            .native_vgr
            .prepare(&mut self.journal, &mut self.state, request);
        if result.is_err() {
            // Fail closed until a trusted operator reconciles any ambiguous journal append.
            self.poisoned = true;
        }
        result
    }
    /// Trusted Rust-only dispatch. The receipt is checked against the fsynced
    /// authoritative journal, never granted from model/CLI self-attestation.
    /// Any ambiguous journal write poisons the process until reconciliation.
    pub fn dispatch_native_vgr(
        &mut self,
        session_id: &str,
        action: &voder_core::action_commit::PreparedAction,
    ) -> Result<voder_runtime::tool_executor::DurableDispatchPermit> {
        if self.poisoned {
            return Err(DaemonError::Conflict("daemon poisoned: recovery required"));
        }
        // Snapshot original prepared identity before writing any event.
        // A preparation replay or mismatch is a deterministic refusal.
        self.native_vgr
            .dispatch(&mut self.journal, &mut self.state, session_id, action)
            .inspect_err(|_| {
                self.poisoned = true;
            })?;
        voder_runtime::tool_executor::DurableDispatchPermit::from_durable_journal(
            &self.state_root.join("events.jsonl"),
            session_id,
            action,
        )
        .map_err(|e| {
            self.poisoned = true;
            DaemonError::Integrity(format!("durable dispatch proof rejected: {e}"))
        })
    }
    /// Execute a single Core-prepared tool in the configured Runtime backend,
    /// then persist the measured workspace/stdio receipt. Fails closed on any
    /// ambiguous external result; no automatic replay on crash.
    pub fn execute_native_sandbox_command(
        &mut self,
        args: NativeSandboxCommand<'_>,
    ) -> Result<voder_core::consequence::ConsequenceAssessment> {
        if self.poisoned {
            return Err(DaemonError::Conflict("daemon poisoned: recovery required"));
        }
        let NativeSandboxCommand {
            session_id,
            action,
            executor,
            policy,
            program,
            arguments,
            capture_dir,
            snapshot,
        } = args;
        let canonical = policy.worktree.canonicalize().map_err(DaemonError::Io)?;
        if snapshot.root() != canonical.as_path() {
            return Err(DaemonError::Conflict(
                "snapshot and sandbox workspace differ",
            ));
        }
        if snapshot
            .workspace(action.intent().workspace.revision.clone())
            .map_err(|e| DaemonError::Integrity(e.to_string()))?
            != action.intent().workspace
        {
            return Err(DaemonError::Conflict(
                "sandbox preimage differs from Core-prepared workspace",
            ));
        }
        // All external effects are after fsync of the dispatch event.
        let permit = self.dispatch_native_vgr(session_id, action)?;
        let result = (|| {
            let process = executor
                .sandboxed_command(action, &permit, policy, program, arguments, capture_dir)
                .map_err(|e| DaemonError::Integrity(e.to_string()))?;
            let receipt = voder_runtime::receipt::capture(voder_runtime::receipt::CaptureContext {
                run: action.intent().run_id.clone(),
                action: action.intent().action_id.clone(),
                contract: action.intent().contract.clone(),
                prepared_before: action.intent().workspace.clone(),
                snapshot,
                process: &process,
                store: &self.store,
            })
            .map_err(|e| DaemonError::Integrity(e.to_string()))?;
            self.native_vgr.observe(
                &mut self.journal,
                &mut self.state,
                session_id,
                action,
                &receipt,
                &self.store,
            )
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Trusted-only native budget settlement. Model wire commands have no route
    /// to this API. A rejected event poisons the writer until recovery.
    pub fn settle_native_budget(
        &mut self,
        session_id: &str,
        action: &voder_core::action_commit::PreparedAction,
        budget: &mut voder_core::budget::BudgetLedger,
        actual: voder_core::budget::BudgetAmount,
    ) -> Result<()> {
        if self.poisoned {
            return Err(DaemonError::Conflict("daemon poisoned: recovery required"));
        }
        crate::native_finalization::settle_observed_action(
            &mut self.journal,
            &mut self.state,
            session_id,
            action,
            budget,
            actual,
        )
        .inspect_err(|_| {
            self.poisoned = true;
        })
    }

    /// Trusted-only Core evidence admission, never reachable from model tools.
    pub fn admit_native_evidence(
        &mut self,
        request: crate::native_finalization::NativeEvidenceRequest<'_>,
    ) -> Result<voder_core::evidence::EvidenceAdmission> {
        if self.poisoned {
            return Err(DaemonError::Conflict("daemon poisoned: recovery required"));
        }
        crate::native_finalization::admit_core_verified_receipt(
            &mut self.journal,
            &mut self.state,
            request,
        )
        .inspect_err(|_| {
            self.poisoned = true;
        })
    }

    pub fn record_native_rollback(
        &mut self,
        session_id: &str,
        action: &voder_core::action_commit::PreparedAction,
        workspace_root: &Path,
    ) -> Result<()> {
        if self.poisoned {
            return Err(DaemonError::Conflict("daemon poisoned: recovery required"));
        }
        crate::native_finalization::record_measured_rollback(
            &mut self.journal,
            &mut self.state,
            session_id,
            action,
            workspace_root,
        )
        .inspect_err(|_| {
            self.poisoned = true;
        })
    }

    /// Trusted-only final safety review. Audit executes fresh pinned validators
    /// under a sealed Runtime sandbox, and journals the proof BEFORE commit.
    pub fn audit_native_closure<B: voder_runtime::sandbox::SandboxBackend>(
        &mut self,
        context: &crate::native_closure::ClosureContext<'_>,
        registry: &voder_runtime::validator::ValidatorRegistry,
        backend: &B,
        policy: &voder_runtime::sandbox::SandboxPolicy,
        capture_root: &Path,
    ) -> Result<String> {
        if self.poisoned {
            return Err(DaemonError::Conflict("daemon poisoned: recovery required"));
        }
        crate::native_closure::meta_audit(
            &mut self.journal,
            &mut self.state,
            context,
            &crate::native_closure::MetaAuditResources {
                registry,
                backend,
                policy,
                store: &self.store,
                capture_root,
            },
        )
        .inspect_err(|_| self.poisoned = true)
    }

    /// This is the ONLY native transition that may mark delivery_allowed=true.
    pub fn commit_native_closure(
        &mut self,
        context: &crate::native_closure::ClosureContext<'_>,
    ) -> Result<String> {
        if self.poisoned {
            return Err(DaemonError::Conflict("daemon poisoned: recovery required"));
        }
        crate::native_closure::final_commit(&mut self.journal, &mut self.state, context)
            .inspect_err(|_| self.poisoned = true)
    }

    pub fn event_count(&self) -> usize {
        self.journal.entries().len()
    }
    pub fn state(&self) -> &State {
        &self.state
    }
    pub fn is_locked(&self) -> bool {
        self.lock.metadata().is_ok()
    }
    fn persist(&mut self, body: Value) -> voder_protocol::errors::Result<()> {
        let record = self.journal.append(body).map_err(|e| {
            self.poisoned = true;
            internal(e)
        })?;
        if let Err(e) = self.state.apply(&record) {
            self.poisoned = true;
            return Err(internal(e));
        }
        Ok(())
    }
    fn idem(&self, req: &Envelope) -> voder_protocol::errors::Result<Option<Value>> {
        let key = req
            .idempotency_key
            .as_ref()
            .ok_or_else(|| error(ErrorCode::InvalidField, "idempotency key missing"))?;
        let scope = req.session_id.clone().unwrap_or_default();
        if let Some(cached) = self.state.dedup.get(&(scope, key.clone())) {
            if cached.fingerprint != snapshot_request(req)? {
                return Err(error(
                    ErrorCode::Conflict,
                    "idempotency key used for different request",
                ));
            }
            return Ok(Some(cached.response.clone()));
        }
        Ok(None)
    }
    fn dispatch(&mut self, req: &Envelope) -> voder_protocol::errors::Result<Value> {
        let command = Command::parse(req)?;
        match command {
            Command::Initialize(hello) => {
                if self.initialized {
                    return Err(error(
                        ErrorCode::InvalidTransition,
                        "connection already initialized",
                    ));
                }
                let negotiated = hello.negotiate()?;
                self.initialized = true;
                Ok(negotiated.as_payload())
            }
            _ if !self.initialized => {
                Err(error(ErrorCode::InvalidTransition, "initialize required"))
            }
            Command::SessionCreate { project_root } => {
                if let Some(value) = self.idem(req)? {
                    return Ok(value);
                }
                let root = Path::new(&project_root)
                    .canonicalize()
                    .map_err(|_| error(ErrorCode::NotFound, "project root missing"))?;
                let repo = Repository::inspect(&root).map_err(|_| {
                    error(
                        ErrorCode::InvalidPayload,
                        "project root must be a readable git repository",
                    )
                })?;
                if repo.root() != root {
                    return Err(error(
                        ErrorCode::InvalidPayload,
                        "noncanonical repository root",
                    ));
                }
                let id = random_id().map_err(internal)?;
                let response = object([
                    ("session_id".into(), strv(&id)),
                    ("status".into(), strv("READY")),
                ]);
                let body = entry(
                    "session.created",
                    &id,
                    req.idempotency_key.as_deref().unwrap_or(""),
                    &snapshot_request(req)?,
                    response.clone(),
                    vec![("project_root", strv(root.to_string_lossy()))],
                );
                self.persist(body)?;
                Ok(response)
            }
            Command::SessionResume => {
                let id = req.session_id.as_deref().unwrap_or("");
                let s = self
                    .state
                    .get(id)
                    .map_err(|_| error(ErrorCode::NotFound, "session not found"))?;
                Ok(object([
                    ("session_id".into(), strv(id)),
                    ("last_sequence".into(), numeric(s.last_sequence)),
                    ("status".into(), strv("READY")),
                ]))
            }
            Command::TaskSubmit {
                instruction,
                context_refs,
            } => {
                if let Some(value) = self.idem(req)? {
                    return Ok(value);
                }
                let id = req.session_id.as_deref().unwrap_or("");
                self.state
                    .get(id)
                    .map_err(|_| error(ErrorCode::NotFound, "session not found"))?;
                let plan = run_loop::evaluate_provider(
                    &context_refs,
                    |digest| {
                        digest
                            .strip_prefix("sha256:")
                            .is_some_and(|sha| self.store.read(sha).is_ok())
                    },
                    self.provider.is_some(),
                );
                let run = format!("r-{}", self.journal.entries().len() + 1);
                let digest = sha256(instruction.as_bytes()).map_err(internal)?;
                let key = req.idempotency_key.as_deref().unwrap_or("");
                let fingerprint = snapshot_request(req)?;
                if plan.gate != run_loop::Gate::ProviderReady {
                    let response = run_loop::blocked_payload(&run, plan.reason);
                    let body = entry(
                        "task.blocked",
                        id,
                        key,
                        &fingerprint,
                        response.clone(),
                        vec![
                            ("run_id", strv(&run)),
                            ("reason", strv(plan.reason)),
                            ("instruction_digest", strv(digest)),
                        ],
                    );
                    self.persist(body)?;
                    return Ok(response);
                }
                // Acknowledgement is idempotent and durable BEFORE any billable model call.
                // No model proposal is an authorized tool action or a VGR verdict.
                let acknowledged = object([
                    ("run_id".into(), strv(&run)),
                    ("status".into(), strv("MODEL_PENDING")),
                    ("delivery_allowed".into(), Value::Bool(false)),
                ]);
                let body = entry(
                    "task.model_requested",
                    id,
                    key,
                    &fingerprint,
                    acknowledged.clone(),
                    vec![
                        ("run_id", strv(&run)),
                        (
                            "reason",
                            strv("model invocation journaled; no tool dispatch permit"),
                        ),
                        ("instruction_digest", strv(&digest)),
                    ],
                );
                self.persist(body)?;
                let result = self
                    .provider
                    .as_ref()
                    .ok_or_else(|| error(ErrorCode::Internal, "provider disappeared"))?
                    .invoke(&self.state_root, &run, &instruction);
                match result {
                    Ok(proposal) => {
                        let proposal_digest = match self.store.save(&proposal.bytes) {
                            Ok(digest) => digest,
                            Err(err) => {
                                // External effect has already happened but evidence cannot be durably stored.
                                // Poison and stop rather than continuing with a pending authoritative run.
                                self.poisoned = true;
                                return Err(internal(err));
                            }
                        };
                        let reason = "model proposal stored as inert artifact; awaiting authorized VGR verification";
                        let response = object([
                            ("run_id".into(), strv(&run)),
                            ("status".into(), strv("PROPOSED")),
                            ("proposal_digest".into(), strv(&proposal_digest)),
                            ("delivery_allowed".into(), Value::Bool(false)),
                        ]);
                        let completion = entry(
                            "task.model_proposed",
                            id,
                            &format!("sys-proposal-{run}"),
                            &fingerprint,
                            response,
                            vec![
                                ("run_id", strv(&run)),
                                ("reason", strv(reason)),
                                ("proposal_digest", strv(&proposal_digest)),
                            ],
                        );
                        self.persist(completion)?;
                    }
                    Err(_) => {
                        let reason = "MODEL_FAILED: bridge rejected or upstream outcome unknown; no automatic retry";
                        let response = run_loop::blocked_payload(&run, reason);
                        let completion = entry(
                            "task.model_failed",
                            id,
                            &format!("sys-failed-{run}"),
                            &fingerprint,
                            response,
                            vec![("run_id", strv(&run)), ("reason", strv(reason))],
                        );
                        self.persist(completion)?;
                    }
                }
                Ok(acknowledged)
            }
            Command::TaskCancel { run_id } => {
                if let Some(value) = self.idem(req)? {
                    return Ok(value);
                }
                let id = req.session_id.as_deref().unwrap_or("");
                let s = self
                    .state
                    .get(id)
                    .map_err(|_| error(ErrorCode::NotFound, "session not found"))?;
                let run = s
                    .runs
                    .get(&run_id)
                    .ok_or_else(|| error(ErrorCode::NotFound, "run not found"))?;
                if !matches!(
                    run.status.as_str(),
                    "MODEL_PENDING" | "PROPOSED" | "BLOCKED"
                ) {
                    return Err(error(
                        ErrorCode::Conflict,
                        "run cannot be cancelled after VGR preparation",
                    ));
                }
                let response = object([
                    ("run_id".into(), strv(&run_id)),
                    ("status".into(), strv("CANCELLED")),
                ]);
                let body = entry(
                    "task.cancelled",
                    id,
                    req.idempotency_key.as_deref().unwrap_or(""),
                    &snapshot_request(req)?,
                    response.clone(),
                    vec![
                        ("run_id", strv(&run_id)),
                        ("reason", strv("cancelled by user")),
                    ],
                );
                self.persist(body)?;
                Ok(response)
            }
            Command::StateGet { run_id } => {
                let id = req.session_id.as_deref().unwrap_or("");
                let session = self
                    .state
                    .get(id)
                    .map_err(|_| error(ErrorCode::NotFound, "session not found"))?;
                if let Some(target) = &run_id {
                    if !session.runs.contains_key(target) {
                        return Err(error(ErrorCode::NotFound, "run not found"));
                    }
                } else if session.runs.len() > 512 {
                    return Err(error(
                        ErrorCode::InvalidTransition,
                        "too many runs: specify run_id",
                    ));
                }
                let runs = session
                    .runs
                    .values()
                    .filter(|r| run_id.as_ref().is_none_or(|s| s == &r.id))
                    .map(|r| {
                        let mut fields = vec![
                            ("run_id".into(), strv(&r.id)),
                            ("status".into(), strv(&r.status)),
                            ("reason".into(), strv(&r.reason)),
                        ];
                        if let Some(digest) = &r.proposal_digest {
                            fields.push(("proposal_digest".into(), strv(digest)));
                        }
                        fields.push((
                            "delivery_allowed".into(),
                            Value::Bool(r.status == "VGR_COMMITTED"),
                        ));
                        if let Some(digest) = &r.native_final_commit_digest {
                            fields.push(("final_commit_digest".into(), strv(digest)));
                        }
                        object(fields)
                    })
                    .collect();
                Ok(object([
                    ("session_id".into(), strv(id)),
                    ("last_sequence".into(), numeric(session.last_sequence)),
                    ("runs".into(), Value::Array(runs)),
                ]))
            }
            Command::ArtifactRead {
                digest,
                offset,
                limit,
            } => {
                let id = req.session_id.as_deref().unwrap_or("");
                self.state
                    .get(id)
                    .map_err(|_| error(ErrorCode::NotFound, "session not found"))?;
                // A proposal receipt is readable ONLY when this session's verified
                // journal binds it to a run. Raw artifact digests grant no access.
                let raw = digest
                    .strip_prefix("sha256:")
                    .ok_or_else(|| error(ErrorCode::InvalidPayload, "invalid artifact digest"))?;
                let session = self
                    .state
                    .get(id)
                    .map_err(|_| error(ErrorCode::NotFound, "session not found"))?;
                if !session
                    .runs
                    .values()
                    .any(|r| r.proposal_digest.as_deref() == Some(raw))
                {
                    return Err(error(ErrorCode::Unauthorized, "artifact not in session"));
                }
                let contents = self.store.read(raw).map_err(internal)?;
                let start = usize::try_from(offset)
                    .unwrap_or(usize::MAX)
                    .min(contents.len());
                let end = start.saturating_add(limit as usize).min(contents.len());
                let mut encoded = String::with_capacity((end - start) * 2);
                for byte in &contents[start..end] {
                    use std::fmt::Write as _;
                    write!(&mut encoded, "{byte:02x}").map_err(internal)?;
                }
                Ok(object([
                    ("digest".into(), strv(&digest)),
                    ("offset".into(), numeric(start as u64)),
                    ("length".into(), numeric((end - start) as u64)),
                    ("total_bytes".into(), numeric(contents.len() as u64)),
                    ("encoding".into(), strv("hex")),
                    ("data_hex".into(), strv(encoded)),
                    ("eof".into(), Value::Bool(end == contents.len())),
                ]))
            }
            Command::RunReplay {
                run_id,
                after_sequence,
                limit,
            } => {
                let id = req.session_id.as_deref().unwrap_or("");
                let session = self
                    .state
                    .get(id)
                    .map_err(|_| error(ErrorCode::NotFound, "session not found"))?;
                if !session.runs.contains_key(&run_id) {
                    return Err(error(ErrorCode::NotFound, "run not found"));
                }
                let mut rows: Vec<Value> = Vec::new();
                let mut used = 0usize;
                for e in self
                    .journal
                    .entries()
                    .iter()
                    .filter(|e| {
                        e.seq > after_sequence
                            && e.body.get("session").and_then(Value::as_str) == Some(id)
                            && e.body.get("run_id").and_then(Value::as_str) == Some(&run_id)
                    })
                    .take(limit as usize)
                {
                    let row = object([
                        ("sequence".into(), numeric(e.seq)),
                        ("event".into(), e.body.clone()),
                    ]);
                    let bytes = row.encode().len();
                    if used.saturating_add(bytes) > 512 * 1024 {
                        break;
                    }
                    used += bytes;
                    rows.push(row);
                }
                Ok(object([("events".into(), Value::Array(rows))]))
            }
            Command::RunExport { .. } => Err(error(
                ErrorCode::InvalidTransition,
                "session-scoped export not yet available",
            )),
            Command::ApprovalRespond { .. } => Err(error(
                ErrorCode::InvalidTransition,
                "no pending trusted approval",
            )),
        }
    }
    pub fn handle(&mut self, req: Envelope) -> Result<Envelope> {
        if self.poisoned {
            return Err(DaemonError::Integrity(
                "journal failed: daemon poisoned".into(),
            ));
        }
        req.validate()
            .map_err(|_| DaemonError::Invalid("invalid request envelope"))?;
        if req.frame_type != FrameKind::Request {
            return Err(DaemonError::Invalid("requests only"));
        }
        let id = req
            .request_id
            .as_ref()
            .ok_or(DaemonError::Invalid("missing request id"))?;
        if self.seen.len() >= 16384 {
            return Err(DaemonError::Invalid("request window full"));
        }
        if !self.seen.insert(id.clone()) {
            return Envelope::reject(
                &req,
                &error(ErrorCode::DuplicateOperation, "request id already seen"),
            )
            .map_err(|_| DaemonError::Invalid("encoding rejection"));
        }
        let reply = match self.dispatch(&req) {
            Ok(payload) => Envelope::respond_to(&req, payload),
            Err(e) => Envelope::reject(&req, &e),
        }
        .map_err(|_| DaemonError::Invalid("failed to encode response"))?;
        if self.poisoned {
            return Err(DaemonError::Integrity(
                "journal commit failed; no response emitted".into(),
            ));
        }
        Ok(reply)
    }
    pub fn serve(&mut self, mut input: impl BufRead, mut output: impl Write) -> Result<()> {
        loop {
            let mut line = Vec::new();
            let count = read_frame(&mut input, &mut line)?;
            if count == 0 {
                return Ok(());
            }
            let frame = Envelope::decode(&line)
                .map_err(|_| DaemonError::Invalid("malformed JSONL frame"))?;
            let response = self.handle(frame)?;
            output.write_all(
                &response
                    .encode_line()
                    .map_err(|_| DaemonError::Invalid("response encoding failed"))?,
            )?;
            output.flush()?;
        }
    }
}
fn read_frame(reader: &mut impl BufRead, output: &mut Vec<u8>) -> Result<usize> {
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            return if output.is_empty() {
                Ok(0)
            } else {
                Err(DaemonError::Invalid("unterminated frame"))
            };
        }
        let end = buf
            .iter()
            .position(|b| *b == b'\n')
            .map(|p| p + 1)
            .unwrap_or(buf.len());
        if output.len().saturating_add(end) > MAX_FRAME_BYTES {
            return Err(DaemonError::Invalid("oversized wire frame"));
        }
        let finished = buf[end - 1] == b'\n';
        output.extend_from_slice(&buf[..end]);
        reader.consume(end);
        if finished {
            return Ok(output.len());
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.lock.unlock();
    }
}
