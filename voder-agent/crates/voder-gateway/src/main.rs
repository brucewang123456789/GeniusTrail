//! Trusted gateway: operator-pinned single-file replacement in an isolated git worktree.
//! One-shot process; model input can propose bytes, never a contract, validator or execution permit.
//! NO model-accessible shell, commit, rollback, approval, or policy replacement endpoint.
use std::{env, fs, io::{self, Read}, fs::OpenOptions, path::{Path, PathBuf}, time::Duration};
use voder_core::{
 action_commit::{ActionIntent, ExpectedChange, MutationKind, PreparedAction},
 budget::{BudgetAmount, BudgetLedger},
 contract::{ContractDraft, FrozenContract, RepoScope, Requirement, TaskAnchor, ToolRule, ValidatorBinding},
 evidence::EvidenceLedger,
 obligation::{ObligationRegistry, ObligationSpec}, retry::ActionDescriptor,
 types::{ContractIdentity, Digest, EvidenceRelation, Severity, ToolEffect, Version, WorkspaceIdentity,
         RequirementId, EvidenceId},
};
use voder_protocol::json::{self, Value};
use voder_runtime::{
 artifact_store::{sha256, ArtifactStore}, patch::{FilePatch, PatchOperation},
 process::ProcessLimits, receipt::{self,CaptureContext,WorkspaceSnapshot},
 repository::Repository, paths, sandbox::{SandboxAvailability,SandboxBackend,SandboxPolicy},
 sandbox_plugin::PluginSandbox, tool_executor::{DurableDispatchPermit,ToolExecutor},
 tool_registry::{ToolKind,ToolRegistry}, validator::{ValidatorCommand,ValidatorRegistry},
 worktree::OwnedWorktree,
};
use voderd::{
 journal::Journal, native_closure::{self,ClosureContext,MetaAuditResources},
 native_finalization, native_trusted_receipt::{self,TrustedReceiptRequest,TrustedReceiptResources},
 native_vgr::{NativePreparation,NativePrepareRequest}, native_write_recovery::reconcile_interrupted_write, session::State,
};

type AnyResult<T> = Result<T,Box<dyn std::error::Error>>;
fn id<'a,T:TryFrom<&'a str>>(s:&'a str)->T where <T as TryFrom<&'a str>>::Error:std::fmt::Debug { T::try_from(s).unwrap() }
fn txt(s:impl Into<String>)->Value {Value::String(s.into())}
fn entry(kind:&str, session:&str, key:&str, fields: Vec<(&str,Value)>)->Value {
 let mut pairs=vec![("type".into(),txt(kind)), ("session".into(),txt(session)),
 ("key".into(),txt(key)), ("fingerprint".into(),txt("voder-gateway-v1")),
 ("response".into(),Value::object([]))];
 pairs.extend(fields.into_iter().map(|(k,v)|(k.into(),v)));
 Value::object(pairs)
}
fn s<'a>(o:&'a Value,k:&str)->AnyResult<&'a str> {Ok(o.get(k).and_then(Value::as_str).ok_or(format!("missing {k}"))?)}
fn exactly_keys(o:&Value, keys:&[&str])->AnyResult<()> {
 let Value::Object(map)=o else{return Err("expected JSON object".into())};
 if map.len()!=keys.len() || map.keys().any(|k|!keys.contains(&k.as_str())){return Err("unknown/missing input fields".into())}
 Ok(())
}
fn clean_id(x:&str)->bool { !x.is_empty() && x.len()<=64 && x.bytes().all(|b|b.is_ascii_alphanumeric()||matches!(b,b'-'|b'_'|b'.')) && x!="." && x!=".." }
fn clean_rel(x:&str)->bool {
 !x.is_empty() && x.len()<=240 && x.split('/').all(|c| {
  !c.is_empty() && !c.starts_with('.') && !c.ends_with('.') &&
  c.bytes().all(|b|b.is_ascii_alphanumeric()||matches!(b,b'-'|b'_'|b'.')) &&
  RepoScope::parse(c).is_ok()
 })
}
fn source_scope(target:&str)->String {
 // Root file gets an exact file scope, never @repository. Nested behavior remains unchanged.
 target.rsplit_once('/').map(|(parent,_)|parent.to_owned()).unwrap_or_else(||target.to_owned())
}

struct Policy {repo:PathBuf,target:String,targets:Vec<String>,scope:String,program:String,args:Vec<String>,validator_sha:String,raw_sha:String}
#[derive(Clone)]
struct Change {relative:String,before_sha:String,content:Vec<u8>}
struct Proposal {session:String,action:String,run:String,relative:String,before_sha:String,content:Vec<u8>,idempotency:String,changes:Vec<Change>,proposal_sha:String}
fn policy(path:&Path)->AnyResult<Policy>{
 let raw=fs::read(path)?; if raw.len()>16384 {return Err("policy too large".into())}
 let j=json::parse(&raw)?;
 let batch=s(&j,"schema")?=="voder.gateway.policy.v2";
 if batch {exactly_keys(&j,&["schema","repository_path","target_paths","validator_program","validator_args","validator_sha256"])?;} else {
 exactly_keys(&j,&["schema","repository_path","target_path","validator_program","validator_args","validator_sha256"])?;
 if s(&j,"schema")?!="voder.gateway.policy.v1" {return Err("unsupported policy version".into())}}

 let repo=PathBuf::from(s(&j,"repository_path")?);
 if !repo.is_absolute(){return Err("repository must be absolute".into())}
 let targets=if batch {
 let Value::Array(paths)=j.get("target_paths").ok_or("missing targets")? else{return Err("invalid target list".into())};
 if !(2..=16).contains(&paths.len()) {return Err("batch policy needs 2..16 targets".into())}
 paths.iter().map(|x|x.as_str().map(str::to_owned).ok_or("target must be string")).collect::<Result<Vec<_>,_>>()?
 }else{vec![s(&j,"target_path")?.to_owned()]};
 if targets.iter().any(|x|!clean_rel(x))||targets.windows(2).any(|x|x[0]>=x[1]) {return Err("targets must be safe, unique and sorted".into())}
 let target=targets[0].clone();
 let scope=if batch {"@repository".to_owned()}else{source_scope(&target)};
 let program=s(&j,"validator_program")?.to_owned();
 if !program.starts_with("/usr/bin/") && !program.starts_with("/bin/"){return Err("validator program outside permitted executables".into())}
 let Value::Array(items)=j.get("validator_args").ok_or("missing validator_args")? else {return Err("validator args must be array".into())};
 if items.len()>16{return Err("too many validator arguments".into())}
 let args=items.iter().map(|v|v.as_str().map(str::to_owned).ok_or("invalid validator arg")).collect::<Result<Vec<_>,_>>()?;
 if args.iter().any(|a|a.len()>512||a.contains('\0')){return Err("invalid validator arg".into())}
 let validator_sha=s(&j,"validator_sha256")?.to_owned();
 if validator_sha.len()!=64||!validator_sha.bytes().all(|b|b.is_ascii_hexdigit()){return Err("bad validator pin".into())}
 let digest=sha256(format!("{}\0{}\0{:?}",program,"v1",args).as_bytes())?;
 if digest!=validator_sha {return Err("operator validator digest mismatch".into())}
 Ok(Policy{repo,target,targets,scope,program,args,validator_sha,raw_sha:sha256(&raw)?})
}
fn parse_proposal(raw:&[u8])->AnyResult<Proposal> {
 let j=json::parse(raw)?;let batch=s(&j,"schema")?=="voder.gateway.request.v2";
 if batch {exactly_keys(&j,&["schema","session_id","action_id","idempotency_key","kind","patches"])?;
 if s(&j,"kind")?!="patch_batch" {return Err("unsupported batch kind".into())}}
 else {exactly_keys(&j,&["schema","session_id","action_id","idempotency_key","kind","relative_path","old_sha256","new_content"])?;
 if s(&j,"schema")?!="voder.gateway.request.v1"||s(&j,"kind")?!="patch" {return Err("unsupported request".into())}}
 let session=s(&j,"session_id")?.to_owned();let action=s(&j,"action_id")?.to_owned();let idem=s(&j,"idempotency_key")?.to_owned();
 if !clean_id(&session)||!clean_id(&action)||!clean_id(&idem) {return Err("invalid identity".into())}
 let values=if batch {let Value::Array(a)=j.get("patches").ok_or("missing patches")? else{return Err("invalid patches".into())};
 if !(2..=16).contains(&a.len()) {return Err("batch needs 2..16 files".into())} a.iter().collect::<Vec<_>>()}else{vec![&j]};
 let mut changes=Vec::new();
 for v in values {
 if batch {exactly_keys(v,&["relative_path","old_sha256","new_content"])?;}
 let relative=s(v,"relative_path")?.to_owned();let before_sha=s(v,"old_sha256")?.to_owned();let content=s(v,"new_content")?.as_bytes().to_vec();
 if !clean_rel(&relative)||before_sha.len()!=64||!before_sha.bytes().all(|c|c.is_ascii_hexdigit())||content.is_empty() {return Err("invalid batch member".into())}
 changes.push(Change{relative,before_sha,content});
 }
 if changes.windows(2).any(|x|x[0].relative>=x[1].relative) {return Err("batch size, order or duplicate".into())}
 let first=changes[0].clone();
 let binding=changes.iter().map(|c|Ok(format!("{}\0{}\0{}",c.relative,c.before_sha,sha256(&c.content)?))).collect::<AnyResult<Vec<_>>>()?.join("\0");
 let proposal_sha=if changes.len()==1 {sha256(&first.content)?}else{sha256(binding.as_bytes())?};
 Ok(Proposal{session,action:action.clone(),run:format!("run-{action}"),relative:first.relative,before_sha:first.before_sha,content:first.content,idempotency:idem,changes,proposal_sha})
}
fn proposal()->AnyResult<Proposal>{let mut raw=Vec::new();io::stdin().read_to_end(&mut raw)?;parse_proposal(&raw)}
fn frozen(p:&Policy)->AnyResult<FrozenContract>{
 let identity=ContractIdentity{contract_id:id("voder-gateway-contract"),version:Version::new(1)?,
 digest:Digest::parse(p.raw_sha.clone())?,toolchain:id("modal-sandbox"),environment:id("voder-gateway-environment")};
 Ok(FrozenContract::compile(ContractDraft{
 anchor:TaskAnchor{task_id:id("gateway-patch-task"),identity:"operator-scoped-model-patch".into(),
 goal:"Apply one proposed replacement to a detached checkout, then independently validate".into(),
 version:Version::new(1)?,accepted_facts:vec![],
 environment_constraints:vec!["clean git input and isolated worktree".into()],
 evaluation_boundary:vec!["operator-pinned sandbox validator and exact measured bytes".into()]},
 identity,
 requirements:vec![Requirement{id:id("gateway-requirement"),description:"pinned validator must succeed".into(),hard:true,severity:Severity::Critical}],
 validators:vec![ValidatorBinding{id:id("gateway-validator"),version:id("v1"),requirement_ids:vec![id("gateway-requirement")]}],
 trusted_sources:vec![id("gateway-native-validator")],
 tools:vec![ToolRule{id:id("gateway-patch"),effect:ToolEffect::Write,requires_approval:false}],
 read_scopes:vec![RepoScope::parse(p.scope.clone())?],
 write_scopes:vec![RepoScope::parse(p.scope.clone())?],forbidden_effects:vec![],required_inputs:vec![],sampling_rules:vec![],
 }).map_err(|errors| format!("contract compilation failed: {errors:?}"))?)
}
fn obligations(contract:&FrozenContract, p:&Policy)->AnyResult<ObligationRegistry>{
 Ok(ObligationRegistry::compile(contract,vec![ObligationSpec{
 id:id("gateway-obligation"),claim_id:"pinned-validator-exit-zero".into(),
 requirement_id:Some(id("gateway-requirement")),threatened_judgement:"delivery".into(),
 expected_observation_type:"validator-process-exit".into(),
 admission_rule_id:"voder-validator-exit-zero-v1".into(),
 scope:RepoScope::parse(p.scope.clone())?, priority:10,
 stop_condition:"independently rerun pinned validator".into(),source:id("gateway-native-validator"),critical:true,
 closure_relation:EvidenceRelation::Supports,dependencies:vec![]
 }])?)
}
fn closure<'a>(r:&'a Proposal,root:&'a Path,action:&'a PreparedAction,contract:&'a FrozenContract,
 workspace:&'a WorkspaceIdentity,obligations:&'a ObligationRegistry,evidence:&'a EvidenceLedger,
 hard:&'a [(RequirementId,EvidenceId)])->ClosureContext<'a>{
 ClosureContext{session_id:&r.session,action,contract,workspace,obligations,evidence,hard_evidence:hard,workspace_root:root}
}
fn status(state:&State,r:&Proposal,expected:&str)->AnyResult<()>{
 let actual=&state.get(&r.session)?.runs.get(&r.run).ok_or("missing VGR run")?.status;
 if actual!=expected{return Err(format!("VGR state mismatch: {actual} != {expected}").into())} Ok(())
}
fn run(base:&Path,p:&Policy,r:&Proposal,relative:&str)->AnyResult<Value>{
 // Caller created a fresh exclusive invocation directory outside the source repo.
 let repo=Repository::inspect(&p.repo)?;
 if !repo.clean()?{return Err("dirty original repository; model must not erase changes".into())}
 let original=repo.fingerprint()?;
 let owned=OwnedWorktree::create(&repo,&base.join("owned-worktrees"),"candidate")?;
 let root=owned.path().canonicalize()?;
 for change in &r.changes {
 let actual_before=fs::read(paths::checked_existing(&root,&change.relative)?)?;
 if sha256(&actual_before)?!=change.before_sha {return Err("stale model preimage".into())}
 if actual_before==change.content {return Err("no-op member rejected".into())}
 }
 let state_dir=base.join("daemon-state");fs::create_dir(&state_dir)?;
 #[cfg(unix)] {use std::os::unix::fs::PermissionsExt;
     fs::set_permissions(&state_dir,fs::Permissions::from_mode(0o700))?;}
 // Exactly one trusted native writer for this journal; observer can only open after exit.
 let mut lockopts=OpenOptions::new();
 lockopts.create(true).write(true);
 #[cfg(unix)] {use std::os::unix::fs::OpenOptionsExt;
     lockopts.mode(0o600).custom_flags(0o400000);}
 let lock=lockopts.open(state_dir.join("daemon.lock"))?;
 lock.try_lock()?;
 let journal_path=state_dir.join("events.jsonl");
 let store=ArtifactStore::open(&state_dir.join("artifacts"))?;
 let mut journal=Journal::open(&journal_path)?;
 journal.append(entry("session.created",&r.session,"gateway-session-created",vec![
 ("project_root",txt(root.to_string_lossy().to_string()))]))?;
 journal.append(entry("task.model_requested",&r.session,"gateway-model-requested",vec![
 ("run_id",txt(&r.run)),("reason",txt("trusted adapter model tool call")),
 ("instruction_digest",txt(sha256(r.idempotency.as_bytes())?))]))?;
 journal.append(entry("task.model_proposed",&r.session,"gateway-model-proposed",vec![
 ("run_id",txt(&r.run)),("reason",txt("model proposal inert until VGR authorizes")),
 ("proposal_digest",txt(&r.proposal_sha))]))?;
 let mut state=State::reconstruct(journal.entries())?;
 let mut preparation=NativePreparation::recover(&journal)?;
 // Trusted sandbox availability must be verified BEFORE durable VGR preparation.
 // A missing host dependency must not strand a newly prepared native action.
 // Sandbox and tool registry are trusted host configuration, not model arguments.
 let node=PathBuf::from(env::var("VODER_NODE_BINARY")?);
 let cli=PathBuf::from(env::var("VODER_MODAL_CLI")?);
 let backend=PluginSandbox::new(node,cli,"modal")?;
 if !matches!(backend.probe(),SandboxAvailability::Available){return Err("Modal sandbox not available".into())}
 let mut tool_registry=ToolRegistry::new();
 tool_registry.insert("gateway-patch",ToolKind::Patch,ToolEffect::Write)?;
 let executor=ToolExecutor{tools:&tool_registry,sandbox:&backend,store:&store};
 let contract=frozen(p)?;
 let mut obligations=obligations(&contract,p)?;
 let mut evidence=EvidenceLedger::new();
 let hard=vec![(id("gateway-requirement"),id("gateway-evidence"))];
 let before=WorkspaceSnapshot::measure(&root)?;
 let workspace=before.workspace(id("gateway-revision"))?;
 let amount=BudgetAmount{steps:1,tool_calls:1,wall_time_ms:3600000,..BudgetAmount::default()};
 let mut budget=BudgetLedger::new(amount);
 let new_content_digest=r.proposal_sha.clone();
 budget.reserve(id("gateway-reservation"),amount)?;
 let action=preparation.prepare(&mut journal,&mut state,NativePrepareRequest{
 session_id:&r.session,contract:&contract,workspace:&workspace,verified_inputs:&[],budget:&budget,
 intent:ActionIntent{
 run_id:id(&r.run),action_id:id(&r.action),
 descriptor:ActionDescriptor{kind:"verified-model-patch".into(),target_obligation_id:Some(id("gateway-obligation")),
 hypothesis_id:None,tool_id:id("gateway-patch"),parameter_key:id(&format!("params-{}",&new_content_digest[..24])),
 scope:RepoScope::parse(p.scope.clone())?},
 contract:contract.identity().clone(),workspace:workspace.clone(),reservation_id:id("gateway-reservation"),
 expected_changes:r.changes.iter().map(|c|Ok(ExpectedChange{path:RepoScope::parse(c.relative.clone())?,kind:MutationKind::Modify})).collect::<AnyResult<Vec<_>>>()?,
 required_validators:vec![id("gateway-validator")],rollback_ref:Some("voder-durable-write-wal".into())},
 approval:None})?;
 status(&state,r,"VGR_PREPARED")?;
 preparation.dispatch(&mut journal,&mut state,&r.session,&action)?;
 let permit=DurableDispatchPermit::from_durable_journal(&journal_path,&r.session,&action)?;
 let changes=r.changes.iter().map(|c|FilePatch{relative:c.relative.clone(),operation:PatchOperation::Replace{old_sha256:c.before_sha.clone(),new_bytes:c.content.clone()}}).collect::<Vec<_>>();
 if changes.len()==1 {executor.apply_patch(&action,&permit,&root,&changes[0])?;}else{executor.apply_patches(&action,&permit,&root,&changes)?;}
 if repo.fingerprint()?!=original {return Err("original repository changed during isolated write".into())}
 let policy=SandboxPolicy::new(root.clone(),ProcessLimits::new(Duration::from_secs(3600),2097152,536870912,3600)?);
 let process=backend.execute(&policy,"/usr/bin/sha256sum",&std::iter::once("--".to_owned()).chain(r.changes.iter().map(|c|c.relative.clone())).collect::<Vec<_>>(),&base.join("capture-original"))?;
 if process.exit_code()!=Some(0)||process.timed_out()||process.output_limited(){return Err("sandboxed observation failed".into())}
 let receipt=receipt::capture(CaptureContext{run:id(&r.run),action:id(&r.action),contract:contract.identity().clone(),
 prepared_before:workspace.clone(),snapshot:&before,process:&process,store:&store})?;
 if store.read(&receipt.stdout_artifact)?!=r.changes.iter().map(|c|Ok(format!("{}  {}\n",sha256(&c.content)?,c.relative))).collect::<AnyResult<Vec<_>>>()?.concat().as_bytes()||!store.read(&receipt.stderr_artifact)?.is_empty(){return Err("observed bytes differ".into())}
 preparation.observe(&mut journal,&mut state,&r.session,&action,&receipt,&store)?;
 status(&state,r,"VGR_OBSERVED")?;
 native_finalization::settle_observed_action(&mut journal,&mut state,&r.session,&action,&mut budget,
 BudgetAmount{steps:1,tool_calls:1,wall_time_ms:u64::try_from(receipt.elapsed_millis)?,..BudgetAmount::default()})?;
 status(&state,r,"VGR_BUDGET_SETTLED")?;
 let after=receipt.report.after.clone();
 let mut validators=ValidatorRegistry::default();
 validators.register(ValidatorCommand{id:id("gateway-validator"),version:id("v1"),program:p.program.clone(),args:p.args.clone()})?;
 native_trusted_receipt::execute_verify_and_admit(&mut journal,&mut state,TrustedReceiptRequest{
 session_id:&r.session,action:&action,contract:&contract,observed_workspace:&after,
 obligation_id:id("gateway-obligation"),observation_id:id("gateway-observation"),evidence_id:id("gateway-evidence"),
 obligations:&mut obligations,ledger:&mut evidence,expected_validator_command_sha256:&p.validator_sha},
 &TrustedReceiptResources{validators:&validators,backend:&backend,policy:&policy,store:&store,capture_dir:&base.join("evidence-capture")})?;
 status(&state,r,"VGR_EVIDENCE_ADMITTED")?;
 let cx=closure(r,&root,&action,&contract,&after,&obligations,&evidence,&hard);
 fs::create_dir(base.join("meta-audit"))?;
 native_closure::meta_audit(&mut journal,&mut state,&cx,
 &MetaAuditResources{registry:&validators,backend:&backend,policy:&policy,store:&store,capture_root:&base.join("meta-audit")})?;
 status(&state,r,"VGR_META_AUDITED")?;
 let commit_digest=native_closure::final_commit(&mut journal,&mut state,&cx)?;
 status(&state,r,"VGR_COMMITTED")?;
 let restored=State::reconstruct(Journal::open(&journal_path)?.entries())?;
 status(&restored,r,"VGR_COMMITTED")?;
 if restored.get(&r.session)?.runs[&r.run].native_final_commit_digest.as_deref()!=Some(commit_digest.as_str()){
 return Err("durable final commit mismatch".into())}
 if repo.fingerprint()?!=original {return Err("original repository changed".into())}
 for change in &r.changes {if fs::read(paths::checked_existing(&root,&change.relative)?)?!=change.content {return Err("post-commit batch mismatch".into())}}
 let journal_digest=sha256(&fs::read(&journal_path)?)?;
 Ok(Value::object([
 ("authority".into(),txt("VODER_NATIVE_VGR")),("status".into(),txt("COMMITTED")),
 ("session_id".into(),txt(&r.session)),("run_id".into(),txt(&r.run)),("action_id".into(),txt(&r.action)),
 ("journal_digest".into(),txt(journal_digest)),("commit_digest".into(),txt(commit_digest)),
 ("workspace_path".into(),txt(root.to_string_lossy().to_string()))]))
}
fn main()->std::process::ExitCode{
 match gateway_main(){Ok(())=>std::process::ExitCode::SUCCESS,Err(e)=>{eprintln!("VODER_TRUSTED_GATEWAY=FAIL: {e}");std::process::ExitCode::FAILURE}}
}
fn gateway_main()->AnyResult<()> {
 let args:Vec<String>=env::args().collect();
 if args.len()==3 && args[1]=="--check-policy" {
   let p=policy(Path::new(&args[2]))?;
   let repo=p.repo.canonicalize()?;
   for target in &p.targets {
     let file=paths::checked_existing(&repo,target)?;
     let meta=fs::symlink_metadata(file)?;
     if !meta.is_file() {return Err("target must be regular file".into())}
   }
   println!("{}",Value::object([
     ("schema".into(),txt("voder.gateway.preflight.v1")),
     ("path_contract".into(),txt("voder.source-path.v1")),
     ("status".into(),txt("PASS")),
     ("targets".into(),Value::Array(p.targets.iter().map(|x|txt(x)).collect())),
     ("policy_sha256".into(),txt(p.raw_sha)),
   ]).encode());
   return Ok(());
 }

 if args.len()==6 && args[1]=="--recover" {
   // Operator-only repair: the adapter never emits --recover from model tools.
   let state=PathBuf::from(&args[2]);
   if !state.is_absolute(){return Err("absolute recovery state required".into())}
   let recovered=reconcile_interrupted_write(&state,&args[3],&args[4],&args[5])?;
   eprintln!("VODER_NATIVE_RECOVERY={recovered:?}");
   return Ok(());
 }
 if args.len()!=5 || args[1]!="--policy"||args[3]!="--attempt-root"{return Err("expected --policy FILE --attempt-root DIR".into())}
 let p=policy(Path::new(&args[2]))?;
 let r=proposal()?;
 let relative=&r.relative;
 if r.changes.iter().map(|c|c.relative.clone()).collect::<Vec<_>>()!=p.targets {return Err("model-selected target list differs from exact operator policy".into())}
 let base=PathBuf::from(&args[4]);
 if !base.is_absolute(){return Err("attempt root must be absolute".into())}
 let parent=base.parent().ok_or("attempt root has no parent")?.canonicalize()?;
 let original=p.repo.canonicalize()?;
 if parent.starts_with(&original)||original.starts_with(&parent){return Err("attempt parent overlaps repository".into())}
 if fs::symlink_metadata(&base).is_ok(){return Err("attempt already exists (no replay)".into())}
 fs::create_dir(&base)?; // exclusive fresh allocation, no automatic replay
 #[cfg(unix)] {use std::os::unix::fs::PermissionsExt;
     fs::set_permissions(&base,fs::Permissions::from_mode(0o700))?;}
 let outcome=run(&base,&p,&r,relative);
 let result=match outcome {
   Ok(value)=>value,
   Err(e)=>{
     let state=base.join("daemon-state");
     if state.join("events.jsonl").exists() {
       // Reconstruct authoritative state before deciding whether write
       // reconciliation is even applicable. Never retry or invent rollback.
       let phase=Journal::open(&state.join("events.jsonl"))
         .and_then(|journal| State::reconstruct(journal.entries()))
         .and_then(|state| {
           let session=state.get(&r.session)?;
           let run=session.runs.get(&r.run)
             .ok_or(voderd::DaemonError::Invalid("missing gateway run"))?;
           Ok(run.status.clone())
         });
       match phase {
         Ok(phase) if phase=="PROPOSED" || phase=="VGR_PREPARED" =>
           eprintln!("VODER_FAILURE_RECOVERY=PRE_DISPATCH_NO_WRITE; NO_AUTOMATIC_REPLAY; OPERATOR_REVIEW_REQUIRED"),
         Ok(_) => match reconcile_interrupted_write(&state,&r.session,&r.run,&r.action) {
           Ok(x)=>eprintln!("VODER_FAILURE_RECOVERY={x:?}"),
           Err(err)=>eprintln!("VODER_FAILURE_RECOVERY=NEEDS_OPERATOR_ATTENTION: {err}"),
         },
         Err(err)=>eprintln!("VODER_FAILURE_RECOVERY=NEEDS_OPERATOR_ATTENTION: STATE_REPLAY_FAILED: {err}"),
       }
     }
     return Err(e);
   }
 };
 println!("{}",result.encode());
 Ok(())
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn bound_all_request_identifiers(){
  for v in ["", "..", "a/b", "a\\b", "a\n", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"] {assert!(!clean_id(v));}
  for v in ["sess-123", "action.123", "voder_123"] {assert!(clean_id(v));}
 }
 #[test]
 fn patch_target_must_be_scoped_and_nontraversing(){
  for v in ["../tmp/x", "src/../x", "src/.git", "/etc/passwd", "a\\b", "", "src//x"]{
    assert!(!clean_rel(v),"unsafe accepted {v}");
  }
  assert!(clean_rel("src/input.txt"));
  assert!(clean_rel("file.txt"));
 }
 #[test]
 fn untrusted_wire_must_not_add_validator_fields(){
  let raw=br#"{"schema":"voder.gateway.request.v1","kind":"patch","session_id":"s","action_id":"a","idempotency_key":"k","relative_path":"src/input.txt","old_sha256":"0000000000000000000000000000000000000000000000000000000000000000","new_content":"hello","validator_program":"/bin/true"}"#;
  let v=json::parse(raw).unwrap();
  assert!(exactly_keys(&v,&["schema","kind","session_id","action_id","idempotency_key","relative_path","old_sha256","new_content"]).is_err());
 }
}

#[cfg(test)] mod batch_request_tests {use super::*;
 fn request(paths:&[&str])->String{let entries=paths.iter().map(|p|format!(r#"{{"relative_path":"{}","old_sha256":"{}","new_content":"fixed"}}"#,p,"0".repeat(64))).collect::<Vec<_>>().join(",");format!(r#"{{"schema":"voder.gateway.request.v2","kind":"patch_batch","session_id":"s","action_id":"a","idempotency_key":"i","patches":[{}]}}"#,entries)}
 #[test]fn batch_cross_directory_sorted_request_is_one_action(){let r=parse_proposal(request(&["lib/a","src/b"]).as_bytes()).unwrap();assert_eq!(r.changes.len(),2);assert_eq!(r.action,"a");}
 #[test]fn batch_duplicate_unsorted_and_traversal_refused(){for paths in [&["src/a","src/a"][..],&["src/b","src/a"][..],&["../escape","src/a"][..],&["src/a"][..]]{assert!(parse_proposal(request(paths).as_bytes()).is_err());}}
}

#[cfg(test)] mod root_path_tests {use super::*;
 #[test] fn root_scope_is_exact_and_nested_scope_unchanged(){
  let root=RepoScope::parse(source_scope("index.js")).unwrap();
  assert_eq!(root.as_str(),"index.js");
  assert!(!root.contains(&RepoScope::parse("other.js").unwrap()));
  assert_eq!(source_scope("src/pkg/a.py"),"src/pkg");
 }
 #[test] fn shared_host_native_path_vectors(){
  let value=json::parse(include_bytes!("path-contract-vectors.json")).unwrap();
  for (key,expected) in [("valid",true),("invalid",false)] {
   let Value::Array(items)=value.get(key).unwrap() else{panic!("vectors missing")};
   for path in items {let p=path.as_str().unwrap();assert_eq!(clean_rel(p),expected,"path={p}");}
  }
 }
 #[test] fn root_proposal_is_exact_and_bounded(){
  let raw=format!(r#"{{"schema":"voder.gateway.request.v1","kind":"patch","session_id":"s","action_id":"a","idempotency_key":"i","relative_path":"index.js","old_sha256":"{}","new_content":"fixed"}}"#,"0".repeat(64));
  assert_eq!(parse_proposal(raw.as_bytes()).unwrap().relative,"index.js");
 }
}

#[cfg(test)] mod mixed_root_batch_test {use super::*;
 #[test] fn root_and_nested_members_share_one_exact_sorted_proposal(){
  let raw=format!(r#"{{"schema":"voder.gateway.request.v2","kind":"patch_batch","session_id":"s","action_id":"a","idempotency_key":"i","patches":[{{"relative_path":"index.js","old_sha256":"{}","new_content":"x"}},{{"relative_path":"src/a.js","old_sha256":"{}","new_content":"y"}}]}}"#,"0".repeat(64),"0".repeat(64));
  let r=parse_proposal(raw.as_bytes()).unwrap();assert_eq!(r.changes.len(),2);assert_eq!(r.changes[0].relative,"index.js");
 }
}

#[cfg(test)] mod file_flex_regressions { use super::*;
 #[test] fn proposals_above_old_wire_and_per_file_caps_are_accepted() {
  let content="x".repeat(3*1024*1024);
  let raw=format!(r#"{{"schema":"voder.gateway.request.v1","kind":"patch","session_id":"s","action_id":"a","idempotency_key":"i","relative_path":"index.js","old_sha256":"{}","new_content":"{}"}}"#,"0".repeat(64),content);
  let proposal=parse_proposal(raw.as_bytes()).unwrap();assert_eq!(proposal.changes[0].content.len(),content.len());
 }
 #[test] fn batch_exceeds_old_aggregate_without_relaxing_exact_schema() {
  let content="x".repeat(700_000);
  let member=|path|format!(r#"{{"relative_path":"{}","old_sha256":"{}","new_content":"{}"}}"#,path,"0".repeat(64),content);
  let raw=format!(r#"{{"schema":"voder.gateway.request.v2","kind":"patch_batch","session_id":"s","action_id":"a","idempotency_key":"i","patches":[{},{}]}}"#,member("a.js"),member("b.js"));
  assert_eq!(parse_proposal(raw.as_bytes()).unwrap().changes.len(),2);
  assert!(parse_proposal(raw.replace("b.js","a.js").as_bytes()).is_err());
 }
}
