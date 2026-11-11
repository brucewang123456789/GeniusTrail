'use strict';
/** Reviewed task scope in the selected directory; VGR still owns candidate delivery. */
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const {argumentPin,pinnedManagedNode,cacheManagedNode}=require('./native-profile.cjs');
const {parsePlan}=require('./task-plan.cjs');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const fail=message=>{throw Object.assign(Error(message),{code:'TASK_WORKSPACE_BLOCKED'});};
async function canonical(p,{directory=false,privateFile=false}={}){
 if(!path.isAbsolute(p)||await fs.realpath(p)!==p)fail('Noncanonical task path: '+p);
 for(let q=p;;q=path.dirname(q)){const s=await fs.lstat(q);if(s.isSymbolicLink())fail('Linked task path');if(q===path.dirname(q))break;}
 const s=await fs.lstat(p);if(directory?!s.isDirectory():!s.isFile())fail('Unexpected task file type');
 if((s.mode&0o022)!==0)fail('Task path is writable by other users: '+p);
 if(privateFile&&((s.mode&0o077)!==0||(process.getuid&&s.uid!==process.getuid())))fail('Private host record required');
 return s;
}
function git(root,...args){return execFileSync('/usr/bin/git',['-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false','-C',root,...args],{env:{PATH:'/usr/bin:/bin',HOME:root,GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null',GIT_TERMINAL_PROMPT:'0'},timeout:10000,maxBuffer:1048576,stdio:'pipe'}).toString();}
async function freshFile(p){await fs.mkdir(path.dirname(p),{recursive:true,mode:0o700});await canonical(path.dirname(p),{directory:true});}
async function writeNew(p,b){const f=await fs.open(p,'wx',0o600);try{await f.writeFile(b);await f.sync();}finally{await f.close();}}
async function validatorFor(scope){return (await fs.readFile(path.join(__dirname,'../resources/task-check.py'),'utf8')).replace('CONFIG = None  # Replaced with the reviewed, host-validated scope.','CONFIG = '+JSON.stringify({files:scope.files,checks:scope.checks}));}
async function snapshot(root,files){const out={};for(const f of files){const p=path.join(root,f);try{await canonical(p);const raw=await fs.readFile(p);if(raw.includes(0))fail('Existing target is not UTF-8 text: '+f);new TextDecoder('utf-8',{fatal:true}).decode(raw);out[f]=hash(raw);}catch(e){if(e.code!=='ENOENT')throw e;let parent=path.dirname(p);while(parent!==root){try{await canonical(parent,{directory:true});break;}catch(e){if(e.code!=='ENOENT')throw e;}parent=path.dirname(parent);}out[f]=null;}}return out;}
/** Determine the owning repository before approval or any preimage write. */
async function resolveTaskScope(selectedRoot,scope){
 await canonical(selectedRoot,{directory:true});
 const owners=[];
 for(const file of scope.files){
  let dir=selectedRoot,owner=selectedRoot;
  for(const component of ['',...file.split('/').slice(0,-1)]){
   if(component)dir=path.join(dir,component);
   try{
    await canonical(dir,{directory:true});
    const marker=path.join(dir,'.git'),meta=await fs.lstat(marker);
    if(meta.isSymbolicLink())fail('Linked Git metadata is not supported.');
    const top=git(dir,'rev-parse','--show-toplevel').trim();
    if(top!==dir)fail('Git metadata does not match task directory: '+dir);
    owner=dir;
   }catch(e){if(e.code!=='ENOENT')throw e;}
  }
  owners.push(owner);
 }
 const unique=[...new Set(owners)];
 if(unique.length!==1)fail('The requested files cross independent Git repositories. Use one task per repository; no files were written.');
 const root=unique[0];
 if(root===selectedRoot)return {root,scope};
 const prefix=path.relative(selectedRoot,root).split(path.sep).join('/')+'/';
 if(prefix.startsWith('../')||scope.files.some(f=>!f.startsWith(prefix)))fail('Task repository leaves selected directory.');
 // Rewrite only literal relative path argv. Opaque evaluated shell/program text
 // cannot be safely rebased and is rejected before any baseline is created.
 if(scope.checks.some(argv=>argv.some(a=>['-c','-e','--eval','--command'].includes(a))))fail('Task commands contain opaque evaluated text. Open '+root+' and resubmit; no files were written.');
 const checks=scope.checks.map(argv=>argv.map((arg,i)=>i&&arg.startsWith(prefix)?arg.slice(prefix.length):arg));
 const files=scope.files.map(f=>f.slice(prefix.length));
 return {root,scope:parsePlan({status:'DRAFT_UNVERIFIED',verified:false,draft:JSON.stringify({...scope,files,checks})})};
}
async function planTask({agentRoot,storageRoot,root,scope}){
 scope=parsePlan({status:'DRAFT_UNVERIFIED',verified:false,draft:JSON.stringify(scope)});
 if(scope.action!=='execute')fail('Executable task scope required');
 const resolved=await resolveTaskScope(root,scope);root=resolved.root;scope=resolved.scope;
 await canonical(root,{directory:true});await canonical(agentRoot,{directory:true});
 const host=path.resolve(agentRoot,'..'),cfg={binary:path.join(host,'.voder-build/release/voder-gateway'),daemonBinary:path.join(host,'.voder-build/release/voderd'),modalCli:path.join(agentRoot,'packages/voder-sandbox-modal/src/cli.ts')};
 // Match the existing deployed host layout (agentRoot usually ends in voder-agent).
 cfg.modalCli=path.join(host,'packages/voder-sandbox-modal/src/cli.ts');
 for(const p of Object.values(cfg))await canonical(p);
 const nodeInfo=await pinnedManagedNode(host,{details:true});
 try{if((await fs.lstat(path.join(root,'.git'))).isSymbolicLink())fail('Linked Git metadata is not supported.');}catch(e){if(e.code!=='ENOENT')throw e;}
 let hasGit=false;try{const top=git(root,'rev-parse','--show-toplevel').trim();if(top!==root)fail('Open the repository root, not a nested task folder.');hasGit=true;}catch(e){if(e.code==='TASK_WORKSPACE_BLOCKED')throw e;}

 const before=await snapshot(root,scope.files),id=crypto.randomUUID(),validatorName='.voder-task-check-'+id+'.py';
 return Object.freeze({id,root,scope,before,hasGit,cfg,nodeInfo,validatorName,validator:await validatorFor(scope),storageRoot:path.resolve(storageRoot)});
}
async function createTask(plan){
 if(!/^[a-f0-9-]{36}$/.test(plan.id)||plan.validatorName!=='.voder-task-check-'+plan.id+'.py')fail('Invalid prepared task identity');
 await canonical(plan.root,{directory:true});
 if(JSON.stringify(await snapshot(plan.root,plan.scope.files))!==JSON.stringify(plan.before))fail('Target changed after scope approval');
 if(!plan.hasGit)git(plan.root,'init','-q');

 for(const file of plan.scope.files)if(plan.before[file]===null){const p=path.join(plan.root,file);await freshFile(p);await writeNew(p,'Voder task preimage; awaiting verified candidate.\n');}
 await writeNew(path.join(plan.root,plan.validatorName),plan.validator);
 // Snapshot only the reviewed current target bytes and the immutable validator.
 // Existing user edits become the approved baseline; unrelated staged files are untouched.
 const added=[...plan.scope.files,plan.validatorName];
 // A caller's staged files must never be swept into this host baseline commit.
 git(plan.root,'add','--',...added);
 const indexed=new Set(git(plan.root,'ls-files','-z','--',...added).split('\0').filter(Boolean));
 if(added.some(f=>!indexed.has(f)))fail('Git did not index every approved preimage. Nested repository or ignore rule detected; no baseline commit attempted.');
 git(plan.root,'-c','user.name=Voder','-c','user.email=task@voder.invalid','commit','--only','-qm','Voder approved task preimages', '--',...added);
 const dir=path.join(plan.storageRoot,'task-profiles',plan.id);await fs.mkdir(dir,{recursive:true,mode:0o700});await canonical(dir,{directory:true,privateFile:true});
 const nodeBinary=await cacheManagedNode(plan.nodeInfo,plan.storageRoot),attemptStore=path.join(dir,'attempt-store');await fs.mkdir(attemptStore,{mode:0o700});await fs.mkdir(path.join(attemptStore,'attempts'),{mode:0o700});
 const args=['/usr/local/bin/python3.10','-B',plan.validatorName],files=plan.scope.files;
 const policy={schema:files.length>1?'voder.gateway.policy.v2':'voder.gateway.policy.v1',repository_path:plan.root,...(files.length>1?{target_paths:files}:{target_path:files[0]}),validator_program:'/usr/bin/env',validator_args:args,validator_sha256:argumentPin('/usr/bin/env',args)};
 const cfg={...plan.cfg,nodeBinary,attemptStore,policyFile:path.join(dir,'policy.json'),targetPath:files[0],...(files.length>1?{targetPaths:files}:{})};
 const profile=path.join(dir,'gateway.json'),recordPath=path.join(dir,'project.json');
 await writeNew(cfg.policyFile,JSON.stringify(policy));await writeNew(profile,JSON.stringify(cfg));
 const record={schema:'voder.task-project.v1',id:plan.id,root:plan.root,profile,targets:files,checks:plan.scope.checks,validatorName:plan.validatorName,validator_sha256:hash(plan.validator),config_sha256:hash(JSON.stringify(cfg)),policy_sha256:hash(JSON.stringify(policy))};
 await writeNew(recordPath,JSON.stringify(record));return validateTask(recordPath,plan.storageRoot);
}
async function validateTask(recordPath,storageRoot){
 const base=path.join(path.resolve(storageRoot),'task-profiles');
 if(path.basename(recordPath)!=='project.json'||path.dirname(path.dirname(recordPath))!==base)fail('Foreign task record');
 await canonical(path.dirname(recordPath),{directory:true,privateFile:true});await canonical(recordPath,{privateFile:true});
 const r=JSON.parse(await fs.readFile(recordPath,'utf8'));
 if(r.schema!=='voder.task-project.v1'||!/^[a-f0-9-]{36}$/.test(r.id)||path.basename(path.dirname(recordPath))!==r.id||r.profile!==path.join(path.dirname(recordPath),'gateway.json')||r.validatorName!=='.voder-task-check-'+r.id+'.py')fail('Task record identity mismatch');
 const scope=parsePlan({status:'DRAFT_UNVERIFIED',verified:false,draft:JSON.stringify({action:'execute',answer:'',files:r.targets,checks:r.checks})});
 if(JSON.stringify(scope.files)!==JSON.stringify(r.targets))fail('Task targets changed');
 await canonical(r.root,{directory:true});await canonical(r.profile,{privateFile:true});
 const raw=await fs.readFile(r.profile,'utf8');if(hash(raw)!==r.config_sha256)fail('Task config changed');const cfg=JSON.parse(raw);
 if(cfg.policyFile!==path.join(path.dirname(recordPath),'policy.json')||cfg.attemptStore!==path.join(path.dirname(recordPath),'attempt-store')||cfg.targetPath!==r.targets[0]||JSON.stringify(cfg.targetPaths??[cfg.targetPath])!==JSON.stringify(r.targets))fail('Task profile scope changed');
 await canonical(cfg.policyFile,{privateFile:true});const praw=await fs.readFile(cfg.policyFile,'utf8');if(hash(praw)!==r.policy_sha256)fail('Task policy changed');const p=JSON.parse(praw),args=['/usr/local/bin/python3.10','-B',r.validatorName];
 if(p.repository_path!==r.root||p.schema!==(r.targets.length>1?'voder.gateway.policy.v2':'voder.gateway.policy.v1')||JSON.stringify(p.target_paths??[p.target_path])!==JSON.stringify(r.targets)||p.validator_program!=='/usr/bin/env'||JSON.stringify(p.validator_args)!==JSON.stringify(args)||p.validator_sha256!==argumentPin('/usr/bin/env',args))fail('Task policy scope mismatch');
 for(const k of ['binary','daemonBinary','modalCli','nodeBinary'])await canonical(cfg[k]);await canonical(cfg.attemptStore,{directory:true,privateFile:true});
 await canonical(path.join(r.root,r.validatorName));const validator=await fs.readFile(path.join(r.root,r.validatorName));if(hash(validator)!==r.validator_sha256||hash(await validatorFor(scope))!==r.validator_sha256)fail('Task validator changed');
 for(const f of r.targets)await canonical(path.join(r.root,f));
 return Object.freeze({...r,recordPath,nodeBinary:cfg.nodeBinary});
}
module.exports={planTask,createTask,validateTask,validatorFor,snapshot,hash,resolveTaskScope};
