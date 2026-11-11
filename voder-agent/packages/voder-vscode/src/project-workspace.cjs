'use strict';
/** Separate managed-project scope. Existing native-profile rules are deliberately unchanged. */
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const {argumentPin,pinnedManagedNode,cacheManagedNode}=require('./native-profile.cjs');
const TARGETS=Object.freeze(['app.js','index.html','style.css']);
const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
const fail=message=>{throw Error('WEB_PROJECT: '+message);};
function isProjectGoal(goal){
 if(typeof goal!=='string')return false;
 const intent=goal.split(/[\n。.!?；;]/,1)[0];
 if(/(?:只读|不要(?:创建|写入|修改)|do not (?:create|write)|read.only)/i.test(intent))return false;
 return /(?:新建|创建)(?:一个)?(?:独立|全新|新的)?(?:的)?(?:网页|web|HTML)(?:动画)?项目|create\s+(?:a\s+)?(?:new\s+|isolated\s+|new isolated\s+)(?:web|html)\s+project/i.test(intent);
}
async function canonical(p,dir=false,privateMode=false){
 if(!path.isAbsolute(p)||await fs.realpath(p)!==p)fail('noncanonical path');
 // Do not accept symbolic-link ancestors even when their ultimate target happens to match.
 for(let q=p;;q=path.dirname(q)){const s=await fs.lstat(q);if(s.isSymbolicLink())fail('symlink path');if(q===path.dirname(q))break;}
 const s=await fs.lstat(p);
 if(dir?!s.isDirectory():!s.isFile())fail('unexpected file type');
 if(privateMode&&((s.mode&0o077)!==0||(process.getuid&&s.uid!==process.getuid())))fail('private owner scope required');
 if(!privateMode&&(s.mode&0o022)!==0)fail('writable trusted path');
 return s;
}
async function write(p,content,mode=0o600){const h=await fs.open(p,'wx',mode);try{await h.writeFile(content);await h.sync();}finally{await h.close();}}
function git(root,...args){return execFileSync('/usr/bin/git',['-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false','-C',root,...args],{env:{PATH:'/usr/bin:/bin',HOME:root,GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null',GIT_TERMINAL_PROMPT:'0'},timeout:10000,maxBuffer:1048576,stdio:'pipe'});}
async function planProject({agentRoot,storageRoot}){
 const host=path.resolve(agentRoot,'..');await canonical(agentRoot,true);await canonical(host,true);
 const cfg={binary:path.join(host,'.voder-build/release/voder-gateway'),daemonBinary:path.join(host,'.voder-build/release/voderd'),modalCli:path.join(host,'packages/voder-sandbox-modal/src/cli.ts')};
 for(const f of Object.values(cfg))await canonical(f);
 const info=await pinnedManagedNode(host,{details:true});
 const validator=await fs.readFile(path.join(__dirname,'../resources/web-project-check.py'));
 const id=crypto.randomUUID(),parent=path.join(host,'voder-projects'),root=path.join(parent,'web-'+id);
 try{await canonical(parent,true,true);}catch(e){if(e.code!=='ENOENT')throw e;}
 return Object.freeze({id,parent,root,storageRoot:path.resolve(storageRoot),cfg:Object.freeze(cfg),nodeInfo:Object.freeze(info),validator:validator.toString('utf8'),validatorDigest:digest(validator)});
}
async function createProject(plan){
 // This function is called only after exact inline approval; root is never caller chosen.
 if(!/^[a-f0-9-]{36}$/.test(plan.id)||plan.root!==path.join(plan.parent,'web-'+plan.id))fail('invalid reserved identity');
 await fs.mkdir(plan.parent,{mode:0o700}).catch(e=>{if(e.code!=='EEXIST')throw e;});await canonical(plan.parent,true,true);
 await fs.mkdir(plan.root,{mode:0o700});await canonical(plan.root,true,true);
 // No overwrites, no deletion on failure: preserve partially created evidence rather than risking another project.
 for(const name of TARGETS)await write(path.join(plan.root,name),'/* Voder managed blank preimage; awaiting generation. */\n');
 await write(path.join(plan.root,'verify_web.py'),plan.validator);
 git(plan.root,'init','-q');git(plan.root,'add','--',...TARGETS,'verify_web.py');git(plan.root,'-c','user.name=Voder','-c','user.email=project@voder.invalid','commit','-qm','Managed blank web project baseline');
 const nodeBinary=await cacheManagedNode(plan.nodeInfo,plan.storageRoot);
 const dir=path.join(plan.storageRoot,'web-project-profiles',plan.id);await fs.mkdir(dir,{recursive:true,mode:0o700});await canonical(dir,true,true);
 const attemptStore=path.join(dir,'attempt-store');await fs.mkdir(attemptStore,{mode:0o700});await fs.mkdir(path.join(attemptStore,'attempts'),{mode:0o700});
 const program='/usr/bin/env',args=['/usr/local/bin/python3.10','-B','verify_web.py'];
 const policy={schema:'voder.gateway.policy.v2',repository_path:plan.root,target_paths:[...TARGETS],validator_program:program,validator_args:args,validator_sha256:argumentPin(program,args)};
 const cfg={...plan.cfg,nodeBinary,policyFile:path.join(dir,'policy.json'),attemptStore,targetPath:TARGETS[0],targetPaths:[...TARGETS]};
 const profile=path.join(dir,'gateway.json');await write(cfg.policyFile,JSON.stringify(policy));await write(profile,JSON.stringify(cfg));
 const record={schema:'voder.web-project.v1',id:plan.id,root:plan.root,profile,validator_sha256:plan.validatorDigest,config_sha256:digest(JSON.stringify(cfg)),policy_sha256:digest(JSON.stringify(policy))};
 await write(path.join(dir,'project.json'),JSON.stringify(record));
 return validateProject(path.join(dir,'project.json'),plan.storageRoot);
}
async function validateProject(recordPath,storageRoot){
 const base=path.join(path.resolve(storageRoot),'web-project-profiles');
 if(path.basename(recordPath)!=='project.json'||path.dirname(path.dirname(recordPath))!==base)fail('foreign project record');
 await canonical(path.dirname(recordPath),true,true);await canonical(recordPath,false,true);
 const r=JSON.parse(await fs.readFile(recordPath,'utf8'));
 if(r.schema!=='voder.web-project.v1'||!/^[a-f0-9-]{36}$/.test(r.id)||path.basename(path.dirname(recordPath))!==r.id||path.basename(r.root)!=='web-'+r.id||path.basename(path.dirname(r.root))!=='voder-projects'||r.profile!==path.join(path.dirname(recordPath),'gateway.json'))fail('record scope');
 await canonical(path.dirname(r.root),true,true);await canonical(r.root,true,true);await canonical(r.profile,false,true);
 const raw=await fs.readFile(r.profile,'utf8');if(digest(raw)!==r.config_sha256)fail('profile changed');const cfg=JSON.parse(raw);
 if(cfg.targetPath!==TARGETS[0]||JSON.stringify(cfg.targetPaths)!==JSON.stringify(TARGETS)||cfg.policyFile!==path.join(path.dirname(recordPath),'policy.json')||cfg.attemptStore!==path.join(path.dirname(recordPath),'attempt-store'))fail('profile scope');
 await canonical(cfg.policyFile,false,true);const pRaw=await fs.readFile(cfg.policyFile,'utf8');if(digest(pRaw)!==r.policy_sha256)fail('policy changed');const p=JSON.parse(pRaw);
 const args=['/usr/local/bin/python3.10','-B','verify_web.py'];
 if(p.schema!=='voder.gateway.policy.v2'||p.repository_path!==r.root||JSON.stringify(p.target_paths)!==JSON.stringify(TARGETS)||p.validator_program!=='/usr/bin/env'||JSON.stringify(p.validator_args)!==JSON.stringify(args)||p.validator_sha256!==argumentPin('/usr/bin/env',args))fail('policy scope');
 for(const k of ['binary','daemonBinary','modalCli','nodeBinary'])await canonical(cfg[k]);
 await canonical(cfg.attemptStore,true,true);
 await canonical(path.join(r.root,'verify_web.py'));if(digest(await fs.readFile(path.join(r.root,'verify_web.py')))!==r.validator_sha256||r.validator_sha256!==digest(await fs.readFile(path.join(__dirname,'../resources/web-project-check.py'))))fail('validator changed');
 for(const f of TARGETS)await canonical(path.join(r.root,f));
 return Object.freeze({...r,recordPath,targets:TARGETS,nodeBinary:cfg.nodeBinary});
}
module.exports={TARGETS,isProjectGoal,planProject,createProject,validateProject,digest};
