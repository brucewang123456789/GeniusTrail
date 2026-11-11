'use strict';
/** Operator-controlled native profiles. No model action, repository mutation or implicit widening. */
const fs=require('node:fs/promises');
const {constants:C}=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const SHA=/^[a-f0-9]{64}$/i;
const IMAGE='docker.io/library/python@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370';
const digest=v=>crypto.createHash('sha256').update(v).digest('hex');
const deny=(code,message)=>{throw Object.assign(new Error(message),{code});};
const argumentPin=(program,args)=>digest(`${program}\0v1\0[${args.map(x=>JSON.stringify(x)).join(', ')}]`);
const json=(p)=>fs.readFile(p,'utf8').then(s=>JSON.parse(s));
async function privateFile(p,max=16384){
 const s=await fs.lstat(p);
 if(!s.isFile()||s.isSymbolicLink()||s.size>max||(s.mode&0o077)!==0||(process.getuid&&s.uid!==process.getuid()))deny('CHAT_PROFILE_NOT_PRIVATE','Native profile must be a private regular file owned by the current user (0600).');
 return s;
}
async function regularFile(p){const s=await fs.lstat(p);if(!s.isFile()||s.isSymbolicLink())deny('CHAT_UNSAFE_FILE','Native component must be a regular non-symlink file.');return s;}
/**
 * A12 uses the resolved Node executable, not the version-manager symlink.
 * Only resolve this ONE known managed tool location. User workspace files,
 * gateway, daemon, policy and secrets retain their no-symlink rules.
 * Fail closed if the resolved executable leaves the Voder-managed tools tree.
 */
async function pinnedManagedNode(host,options={}){
 const tools=path.resolve(host,'.voder-tools');
 const requested=path.join(tools,'node/bin/node');
 let anchor,real;
 try{anchor=await fs.realpath(tools);real=await fs.realpath(requested);}
 catch(e){deny('CHAT_NODE_RESOLVE_FAILED','Voder managed Node executable is missing, dangling or not resolvable ('+String(e.code||'IO_ERROR')+').');}
 if(anchor!==tools)deny('CHAT_NODE_ROOT_SYMLINK','The Voder managed tools directory must be a real directory, not a symlink.');
 const permitted=[path.join(tools,'node'),path.join(tools,'node-versions')];
 if(!permitted.some(root=>real.startsWith(root+path.sep)))
  deny('CHAT_NODE_OUTSIDE_TRUSTED_ROOT','The managed Node link points outside Voder managed Node installations.');
 // Validate the entire canonical chain beneath .voder-tools to prevent a
 // writable or redirected version directory from supplying a different binary.
 const rel=path.relative(tools,real);
 const parts=rel.split(path.sep);
 let foreignOwner=false;
 let dir=tools;
 for(const segment of ['.',...parts.slice(0,-1)]){
  if(segment!=='.')dir=path.join(dir,segment);
  let st;try{st=await fs.lstat(dir);}catch{deny('CHAT_NODE_UNSAFE_DIRECTORY','A managed Node directory disappeared during authorization.');}
  if(!st.isDirectory()||st.isSymbolicLink()||(st.mode&0o022)!==0)
   deny('CHAT_NODE_UNSAFE_DIRECTORY','Managed Node directory is linked, not a directory, or writable by group/others: '+dir);
  if(process.getuid&&st.uid!==process.getuid()&&st.uid!==0)foreignOwner=true;
 }
 let st;try{st=await fs.lstat(real);}catch{deny('CHAT_NODE_RESOLVE_FAILED','Managed Node target disappeared during authorization.');}
 if(!st.isFile()||st.isSymbolicLink()||(st.mode&0o111)===0||(st.mode&0o022)!==0)
  deny('CHAT_NODE_UNSAFE_EXECUTABLE','Managed Node target must be a non-group-writable executable regular file.');
 if(process.getuid&&st.uid!==process.getuid()&&st.uid!==0)foreignOwner=true;
 // Reject a changed link between discovery and final authorization.
 if(await fs.realpath(requested)!==real)deny('CHAT_NODE_CHANGED','Managed Node symlink changed during authorization; retry only after inspecting the installation.');
 return options.details?{path:real,foreignOwner}:real;
}
/**
 * A foreign-owned installation (e.g. UID 1001 under a root AutoDL session)
 * is not inherently malicious, but its owner CAN replace executable contents.
 * Therefore do not execute it directly as root. Snapshot through an O_NOFOLLOW
 * file descriptor into a private, content-addressed per-user cache; the gateway
 * receives only that current-user-owned executable. Neither chown nor widening
 * of the source permissions is necessary.
 */
async function cacheManagedNode(info,storageRoot){
 if(!info.foreignOwner)return info.path;
 const base=path.join(path.resolve(storageRoot),'.voder-chat-node-cache');
 await fs.mkdir(base,{recursive:true,mode:0o700});
 const dirStat=await fs.lstat(base);
 if(!dirStat.isDirectory()||dirStat.isSymbolicLink()||(dirStat.mode&0o077)!==0||(process.getuid&&dirStat.uid!==process.getuid()))
  deny('CHAT_NODE_CACHE_UNSAFE','Node snapshot cache must be a private directory owned by the active user.');
 const source=await fs.open(info.path,C.O_RDONLY|C.O_NOFOLLOW);
 let tmp;
 try{
  const before=await source.stat();
  if(!before.isFile()||(before.mode&0o022)!==0||(before.mode&0o111)===0)
   deny('CHAT_NODE_UNSAFE_EXECUTABLE','Source Node executable changed during snapshot.');
  tmp=path.join(base,'.pending-'+crypto.randomUUID());
  const dest=await fs.open(tmp,C.O_WRONLY|C.O_CREAT|C.O_EXCL|C.O_NOFOLLOW,0o700);
  const h=crypto.createHash('sha256');let total=0;
  try{
   const buf=Buffer.allocUnsafe(1024*1024);
   for(;;){const {bytesRead}=await source.read(buf,0,buf.length,null);if(bytesRead===0)break;
    h.update(buf.subarray(0,bytesRead));let written=0;
    while(written<bytesRead){const r=await dest.write(buf,written,bytesRead-written,null);if(r.bytesWritten===0)throw Error('NODE_SNAPSHOT_WRITE_STALLED');written+=r.bytesWritten;}
    total+=bytesRead;
   }
   await dest.sync();await dest.chmod(0o500);
  }finally{await dest.close();}
  const after=await source.stat();
  if(before.dev!==after.dev||before.ino!==after.ino||before.size!==after.size||before.mtimeMs!==after.mtimeMs||before.ctimeMs!==after.ctimeMs||total!==after.size)
   deny('CHAT_NODE_CHANGED','Managed Node changed while copying; snapshot rejected.');
  const hash=h.digest('hex'),file=path.join(base,hash+'.node');
  try{await fs.link(tmp,file);}catch(e){if(e.code!=='EEXIST')throw e;}
  await validateCachedNode(file,base);
  return file;
 }finally{await source.close();if(tmp)await fs.rm(tmp,{force:true}).catch(()=>{});}
}
async function validateCachedNode(file,base){
 if(path.dirname(file)!==base||!/^[a-f0-9]{64}\.node$/.test(path.basename(file)))deny('CHAT_NODE_CACHE_UNSAFE','Unexpected Node cache filename.');
 const ds=await fs.lstat(base),s=await fs.lstat(file);
 if(!ds.isDirectory()||ds.isSymbolicLink()||(ds.mode&0o077)!==0||(process.getuid&&ds.uid!==process.getuid())||
    !s.isFile()||s.isSymbolicLink()||(s.mode&0o077)!==0||(s.mode&0o100)===0||(process.getuid&&s.uid!==process.getuid()))
  deny('CHAT_NODE_CACHE_UNSAFE','Cached Node executable is not exclusively owned and executable by the active user.');
 const fd=await fs.open(file,C.O_RDONLY|C.O_NOFOLLOW);
 try{
  const actual=crypto.createHash('sha256');
  for await(const chunk of fd.createReadStream({autoClose:false}))actual.update(chunk);
  if(actual.digest('hex')!==path.basename(file).slice(0,64))deny('CHAT_NODE_CACHE_HASH_MISMATCH','Cached Node bytes no longer match their pinned digest.');
 }finally{await fd.close();}
}

async function workspacePath(folder){const root=path.resolve(folder.uri.fsPath),real=await fs.realpath(root);if(root!==real)deny('CHAT_WORKSPACE_SYMLINK','Native workspace must use its canonical path.');return root;}
function sourceRelative(root,selected){
 const abs=path.resolve(selected),r=path.relative(root,abs).replaceAll(path.sep,'/');
 if(!r||r.startsWith('../')||r==='..'||path.isAbsolute(r)||r.split('/').some(s=>!s||s==='.'||s==='..')||r.includes('\\')||r.includes('\0'))deny('CHAT_SCOPE_OUTSIDE_WORKSPACE','Choose a file within the opened workspace.');
 return r;
}
function tracked(root,rel){
 try{
  const checked=execFileSync('/usr/bin/git',['-C',root,'show','HEAD:'+rel],{timeout:4000,stdio:'pipe',maxBuffer:131073});
  const current=require('node:fs').readFileSync(path.join(root,rel));
  return current.equals(checked);
 }catch{return false;}
}

async function validateProfile(file,folder){
 if(typeof file!=='string'||!path.isAbsolute(file))deny('CHAT_PROFILE_PATH','Native gateway must be an absolute file path.');
 await privateFile(file,8192);const cfg=await json(file),keys=['binary','daemonBinary','policyFile','attemptStore','nodeBinary','modalCli','targetPath'];
 if(!cfg||typeof cfg!=='object'||Array.isArray(cfg)||keys.some(k=>typeof cfg[k]!=='string'||(k!=='targetPath'&&!path.isAbsolute(cfg[k])))||Object.keys(cfg).some(k=>![...keys,'targetPaths'].includes(k)))deny('CHAT_PROFILE_FORMAT','Invalid Voder native gateway file.');
 if(cfg.targetPaths!==undefined)deny('CHAT_BATCH_REQUIRES_OPERATOR_PROFILE','Batch policies must be separately approved and are not supported by this Chat onboarding.');
 const root=await workspacePath(folder);
 await privateFile(cfg.policyFile);const p=await json(cfg.policyFile);
 if(p.schema!=='voder.gateway.policy.v1'||p.repository_path!==root||p.target_path!==cfg.targetPath||typeof p.validator_program!=='string'||!Array.isArray(p.validator_args)||p.validator_args.some(x=>typeof x!=='string')||!SHA.test(p.validator_sha256)||argumentPin(p.validator_program,p.validator_args)!==p.validator_sha256)deny('CHAT_PROFILE_POLICY_MISMATCH','Policy must pin this exact workspace, target file and validator command.');
 if(!/^\/(usr\/bin|bin)\//.test(p.validator_program))deny('CHAT_VALIDATOR_PROGRAM','Validator program must be an allowed OS executable.');
 const abs=path.resolve(root,cfg.targetPath);
 if(sourceRelative(root,abs)!==cfg.targetPath)deny('CHAT_PROFILE_SCOPE','Gateway target path must be canonical.');
 await regularFile(abs);if(!tracked(root,cfg.targetPath))deny('CHAT_TARGET_NOT_TRACKED','Target must already exist, be committed in Git HEAD and have unchanged content. New files or dirty targets are unsupported by this VGR delivery version.');
 for(const f of ['binary','daemonBinary','modalCli'])await regularFile(cfg[f]);
 const cacheBase=path.join(path.resolve(path.dirname(file),'../../..'),'.voder-chat-node-cache');
 if(path.dirname(cfg.nodeBinary)===cacheBase)await validateCachedNode(cfg.nodeBinary,cacheBase);
 else await regularFile(cfg.nodeBinary);
 const attempt=await fs.lstat(cfg.attemptStore);if(!attempt.isDirectory()||(attempt.mode&0o077)!==0)deny('CHAT_ATTEMPT_STORE_UNSAFE','Native attempt store must be a private directory.');
 return {path:file,target:cfg.targetPath,validator:p.validator_args.join(' '),workspace:root};
}
async function createProfile({folder,agentRoot,storageRoot,targetFile,testFile}){
 const root=await workspacePath(folder),target=sourceRelative(root,targetFile),test=sourceRelative(root,testFile);
 for(const r of [target,test]){await regularFile(path.join(root,r));if(!tracked(root,r))deny('CHAT_FILE_NOT_TRACKED','Native target and validator must be committed, existing and clean Git files (needed for independent clone verification).');}
 if(target===test)deny('CHAT_TEST_TARGET_CONFLICT','Independent validator test must be a different file from the editable target.');
 if(!test.endsWith('.py'))deny('CHAT_VALIDATOR_TYPE','Managed profiles currently support Python validator scripts only; select an existing gateway for other languages.');
 const host=path.resolve(agentRoot,'..'),binary=path.join(host,'.voder-build/release/voder-gateway'),daemonBinary=path.join(host,'.voder-build/release/voderd'),modalCli=path.join(host,'packages/voder-sandbox-modal/src/cli.ts');
 for(const f of [binary,daemonBinary,modalCli])await regularFile(f);
 // Match the real A12 evaluator: store the canonical trusted Node binary, never the symlink.
 const nodeBinary=await cacheManagedNode(await pinnedManagedNode(host,{details:true}),storageRoot);
 const program='/usr/bin/env',args=['/usr/local/bin/python3.10','-B',test];
 const id=crypto.randomUUID(),dir=path.join(storageRoot,'native-profiles',digest(root).slice(0,24),id);await fs.mkdir(dir,{recursive:true,mode:0o700});
 const store=path.join(dir,'attempt-store');await fs.mkdir(store,{mode:0o700});await fs.mkdir(path.join(store,'attempts'),{mode:0o700});
 const policy={schema:'voder.gateway.policy.v1',repository_path:root,target_path:target,validator_program:program,validator_args:args,validator_sha256:argumentPin(program,args)};
 const cfg={binary,daemonBinary,policyFile:path.join(dir,'policy.json'),attemptStore:store,nodeBinary,modalCli,targetPath:target};
 const write=async(p,data)=>{const h=await fs.open(p,'wx',0o600);try{await h.writeFile(JSON.stringify(data)+'\n');await h.sync();}finally{await h.close();}};
 await write(cfg.policyFile,policy);await write(path.join(dir,'gateway.json'),cfg);
 return validateProfile(path.join(dir,'gateway.json'),folder);
}
module.exports={IMAGE,argumentPin,validateProfile,createProfile,sourceRelative,tracked,pinnedManagedNode,cacheManagedNode};
