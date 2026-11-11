//! Executes only a Core PreparedAction *and* a separate durable dispatch proof.
//! The daemon must persist PREPARED first. No model-controlled host shell escape hatch.
use crate::{
    artifact_store::ArtifactStore,
    durable_write,
    patch::{FilePatch, PatchResult},
    paths,
    process::ProcessResult,
    sandbox::{SandboxBackend, SandboxPolicy},
    tool_registry::{ToolKind, ToolRegistry},
    Result, RuntimeError,
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
use voder_core::action_commit::MutationKind;
use voder_core::{
    action_commit::PreparedAction,
    types::{ActionId, RunId},
};

#[derive(Debug)]
pub struct DurableDispatchPermit {
    run: RunId,
    action: ActionId,
    acknowledged: bool,
    used: AtomicBool,
    journal_path: PathBuf,
    session_id: String,
}
impl DurableDispatchPermit {
    /// Only a trusted Rust caller holding the Daemon's exclusive journal lock
    /// may request this capability. No CLI/model payload can instantiate it.
    /// The last journal record must be a fsynced, identity-bound dispatch.
    pub fn from_durable_journal(
        path: &Path,
        session_id: &str,
        action: &PreparedAction,
    ) -> Result<Self> {
        crate::dispatch_proof::verify(path, session_id, action)?;
        Ok(Self {
            run: action.intent().run_id.clone(),
            action: action.intent().action_id.clone(),
            acknowledged: true,
            used: AtomicBool::new(false),
            journal_path: path.canonicalize()?,
            session_id: session_id.to_owned(),
        })
    }
    fn matches(&self, a: &PreparedAction) -> bool {
        self.acknowledged && self.run == a.intent().run_id && self.action == a.intent().action_id
    }
}
pub struct ToolExecutor<'a> {
    pub tools: &'a ToolRegistry,
    pub sandbox: &'a dyn SandboxBackend,
    pub store: &'a ArtifactStore,
}
impl ToolExecutor<'_> {
    fn gate(&self, a: &PreparedAction, p: &DurableDispatchPermit, k: ToolKind) -> Result<()> {
        if !p.matches(a) {
            return Err(RuntimeError::Unauthorized("missing durable dispatch proof"));
        }
        self.tools.authorize(a, k)?;
        if p.used.swap(true, Ordering::SeqCst) {
            return Err(RuntimeError::Unauthorized(
                "durable dispatch permit already consumed",
            ));
        }
        Ok(())
    }
    pub fn read(
        &self,
        a: &PreparedAction,
        p: &DurableDispatchPermit,
        workspace: &Path,
        relative: &str,
    ) -> Result<String> {
        self.gate(a, p, ToolKind::ReadFile)?;
        if !a.intent().descriptor.scope.contains(
            &voder_core::contract::RepoScope::parse(relative.to_string())
                .map_err(|_| RuntimeError::UnsafePath(relative.into()))?,
        ) {
            return Err(RuntimeError::Unauthorized("read out of committed scope"));
        }
        let path = paths::checked_existing(workspace, relative)?;
        let data = fs::read(path)?;
        self.store.save(&data)
    }
    pub fn apply_patch(
        &self,
        a: &PreparedAction,
        p: &DurableDispatchPermit,
        workspace: &Path,
        change: &FilePatch,
    ) -> Result<PatchResult> {
        self.gate(a, p, ToolKind::Patch)?;
        let rel = voder_core::contract::RepoScope::parse(change.relative.to_string())
            .map_err(|_| RuntimeError::UnsafePath(change.relative.clone()))?;
        let kind = match &change.operation {
            crate::patch::PatchOperation::Create { .. } => MutationKind::Create,
            crate::patch::PatchOperation::Replace { .. } => MutationKind::Modify,
            crate::patch::PatchOperation::Delete { .. } => MutationKind::Delete,
        };
        if !a
            .intent()
            .expected_changes
            .iter()
            .any(|c| c.path == rel && c.kind == kind)
        {
            return Err(RuntimeError::Unauthorized("patch not in expected changes"));
        }
        let journal_parent = p
            .journal_path
            .parent()
            .ok_or(RuntimeError::Unauthorized("dispatch journal has no parent"))?;
        let wal_root = journal_parent.join("voder-write-transactions");
        let key = format!("{}:{}:{}", p.session_id, p.run, p.action);
        durable_write::apply(workspace, &wal_root, &key, change, self.store)
    }
    pub fn apply_patches(&self,a:&PreparedAction,p:&DurableDispatchPermit,workspace:&Path,changes:&[FilePatch])->Result<Vec<PatchResult>> {
        self.gate(a,p,ToolKind::Patch)?;
        if changes.len()!=a.intent().expected_changes.len() {return Err(RuntimeError::Unauthorized("batch must match ALL expected changes"));}
        let mut seen=std::collections::BTreeSet::new();
        for change in changes {
            let rel=voder_core::contract::RepoScope::parse(change.relative.clone()).map_err(|_|RuntimeError::UnsafePath(change.relative.clone()))?;
            if !seen.insert(change.relative.clone())||!matches!(&change.operation,crate::patch::PatchOperation::Replace{..})||!a.intent().expected_changes.iter().any(|e|e.path==rel&&e.kind==MutationKind::Modify) {return Err(RuntimeError::Unauthorized("batch outside exact expected changes"));}
        }
        let parent=p.journal_path.parent().ok_or(RuntimeError::Unauthorized("journal parent"))?;
        durable_write::apply_batch(workspace,&parent.join("voder-write-transactions"),&format!("{}:{}:{}",p.session_id,p.run,p.action),changes,self.store)
    }
    pub fn sandboxed_command(
        &self,
        a: &PreparedAction,
        p: &DurableDispatchPermit,
        policy: &SandboxPolicy,
        program: &str,
        args: &[String],
        capture: &Path,
    ) -> Result<ProcessResult> {
        self.gate(a, p, ToolKind::SandboxedCommand)?;
        self.sandbox.execute(policy, program, args, capture)
    }
}
