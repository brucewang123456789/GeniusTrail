'use strict';
/** Separate managed-project scope. Existing native-profile rules are deliberately unchanged. */
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const {argumentPin,pinnedManagedNode,cacheManagedNode}=require('./native-profile.cjs');
const CORE=Object.freeze(['app.js','index.html','style.css']);
const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
const fail=message=>{throw Error('WEB_PROJECT: '+message);};
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
function parseMultiGoal(goal){
 if(typeof goal!=='string'||!/^\s*(?:Create a new multi-file web project|创建一个独立多文件网页项目)/i.test(goal))return null;
 const match=goal.match(/(?:^|\n)Files:\s*([^\n]+)/i);
 if(!match)throw Error('MULTI_FILES_REQUIRED: add Files: app.js, index.html, style.css, src/helpers.js');
 const targets=match[1].split(',').map(x=>x.trim()).sort();
 if(targets.length<4||targets.length>16||new Set(targets).size!==targets.length||CORE.some(x=>!targets.includes(x))||targets.some(x=>x.length>160||x.split('/').some(y=>!(/^[A-Za-z0-9_-][A-Za-z0-9._-]*$/).test(y)||y.endsWith('.')||/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(y))||!(/\.(?:js|mjs|cjs|css|html|json|py)$/).test(x)||x==='verify_web.py'||x.startsWith('node_modules/')||x.startsWith('.')))throw Error('MULTI_FILES_SCOPE: select 4..16 distinct HTML/CSS/JS/JSON/Python paths including app.js, index.html and style.css');
 if(targets.some((x,i)=>targets.some((y,j)=>i!==j&&y.startsWith(x+'/'))))throw Error('MULTI_FILE_DIRECTORY_CONFLICT');
 return Object.freeze(targets);
}
async function planValidator(TARGETS){
 const original=await fs.readFile(path.join(__dirname,'../resources/web-project-check.py'),'utf8');
 return Buffer.from(original.replace("if __name__ == '__main__': check()", "if __name__ == '__main__':\n    import json, ast\n    total = 0\n    for relative in json.loads("+JSON.stringify(JSON.stringify(TARGETS))+"):\n        raw = Path(relative).read_bytes()\n        assert 1 <= len(raw) and b'\\x00' not in raw, 'invalid auxiliary source'\n        total += len(raw)\n        text = raw.decode('utf-8')\n        if relative.endswith('.json'): json.loads(text)\n        if relative.endswith('.py'): ast.parse(text)\n    check()"));
}
function operations(TARGETS){
 if(!Array.isArray(TARGETS)||TARGETS.length<4||TARGETS.length>16)fail('invalid multi-file targets');
async function planProject({agentRoot,storageRoot}){
 const host=path.resolve(agentRoot,'..');await canonical(agentRoot,true);await canonical(host,true);
 const cfg={binary:path.join(host,'.voder-build/release/voder-gateway'),daemonBinary:path.join(host,'.voder-build/release/voderd'),modalCli:path.join(host,'packages/voder-sandbox-modal/src/cli.ts')};
 for(const f of Object.values(cfg))await canonical(f);
 const info=await pinnedManagedNode(host,{details:true});
 const validator=await planValidator(TARGETS);
 const id=crypto.randomUUID(),parent=path.join(host,'voder-multi-projects'),root=path.join(parent,'multi-'+id);
 try{await canonical(parent,true,true);}catch(e){if(e.code!=='ENOENT')throw e;}
 return Object.freeze({id,parent,root,storageRoot:path.resolve(storageRoot),cfg:Object.freeze(cfg),nodeInfo:Object.freeze(info),validator:validator.toString('utf8'),validatorDigest:digest(validator)});
}
async function createProject(plan){
 // This function is called only after exact inline approval; root is never caller chosen.
 if(!/^[a-f0-9-]{36}$/.test(plan.id)||plan.root!==path.join(plan.parent,'multi-'+plan.id))fail('invalid reserved identity');
 await fs.mkdir(plan.parent,{mode:0o700}).catch(e=>{if(e.code!=='EEXIST')throw e;});await canonical(plan.parent,true,true);
 await fs.mkdir(plan.root,{mode:0o700});await canonical(plan.root,true,true);
 // No overwrites, no deletion on failure: preserve partially created evidence rather than risking another project.
 for(const name of TARGETS){await fs.mkdir(path.dirname(path.join(plan.root,name)),{recursive:true,mode:0o700});await write(path.join(plan.root,name),name.endsWith('.py')?'# Voder managed blank preimage\n':name.endsWith('.json')?'{}\n':'/* Voder managed blank preimage; awaiting generation. */\n');}
 await write(path.join(plan.root,'verify_web.py'),plan.validator);
 git(plan.root,'init','-q');git(plan.root,'add','--',...TARGETS,'verify_web.py');git(plan.root,'-c','user.name=Voder','-c','user.email=project@voder.invalid','commit','-qm','Managed blank web project baseline');
 const nodeBinary=await cacheManagedNode(plan.nodeInfo,plan.storageRoot);
 const dir=path.join(plan.storageRoot,'multi-project-profiles',plan.id);await fs.mkdir(dir,{recursive:true,mode:0o700});await canonical(dir,true,true);
 const attemptStore=path.join(dir,'attempt-store');await fs.mkdir(attemptStore,{mode:0o700});await fs.mkdir(path.join(attemptStore,'attempts'),{mode:0o700});
 const program='/usr/bin/env',args=['/usr/local/bin/python3.10','-B','verify_web.py'];
 const policy={schema:'voder.gateway.policy.v2',repository_path:plan.root,target_paths:[...TARGETS],validator_program:program,validator_args:args,validator_sha256:argumentPin(program,args)};
 const cfg={...plan.cfg,nodeBinary,policyFile:path.join(dir,'policy.json'),attemptStore,targetPath:TARGETS[0],targetPaths:[...TARGETS]};
 const profile=path.join(dir,'gateway.json');await write(cfg.policyFile,JSON.stringify(policy));await write(profile,JSON.stringify(cfg));
 const record={schema:'voder.multi-web-project.v1',targets:[...TARGETS],id:plan.id,root:plan.root,profile,validator_sha256:plan.validatorDigest,config_sha256:digest(JSON.stringify(cfg)),policy_sha256:digest(JSON.stringify(policy))};
 await write(path.join(dir,'project.json'),JSON.stringify(record));
 return validateProject(path.join(dir,'project.json'),plan.storageRoot);
}
async function validateProject(recordPath,storageRoot){
 const base=path.join(path.resolve(storageRoot),'multi-project-profiles');
 if(path.basename(recordPath)!=='project.json'||path.dirname(path.dirname(recordPath))!==base)fail('foreign project record');
 await canonical(path.dirname(recordPath),true,true);await canonical(recordPath,false,true);
 const r=JSON.parse(await fs.readFile(recordPath,'utf8'));
 if(r.schema!=='voder.multi-web-project.v1'||!/^[a-f0-9-]{36}$/.test(r.id)||path.basename(path.dirname(recordPath))!==r.id||path.basename(r.root)!=='multi-'+r.id||path.basename(path.dirname(r.root))!=='voder-multi-projects'||r.profile!==path.join(path.dirname(recordPath),'gateway.json'))fail('record scope');
 await canonical(path.dirname(r.root),true,true);await canonical(r.root,true,true);await canonical(r.profile,false,true);
 const raw=await fs.readFile(r.profile,'utf8');if(digest(raw)!==r.config_sha256)fail('profile changed');const cfg=JSON.parse(raw);
 if(cfg.targetPath!==TARGETS[0]||JSON.stringify(cfg.targetPaths)!==JSON.stringify(TARGETS)||cfg.policyFile!==path.join(path.dirname(recordPath),'policy.json')||cfg.attemptStore!==path.join(path.dirname(recordPath),'attempt-store'))fail('profile scope');
 await canonical(cfg.policyFile,false,true);const pRaw=await fs.readFile(cfg.policyFile,'utf8');if(digest(pRaw)!==r.policy_sha256)fail('policy changed');const p=JSON.parse(pRaw);
 const args=['/usr/local/bin/python3.10','-B','verify_web.py'];
 if(p.schema!=='voder.gateway.policy.v2'||p.repository_path!==r.root||JSON.stringify(p.target_paths)!==JSON.stringify(TARGETS)||p.validator_program!=='/usr/bin/env'||JSON.stringify(p.validator_args)!==JSON.stringify(args)||p.validator_sha256!==argumentPin('/usr/bin/env',args))fail('policy scope');
 for(const k of ['binary','daemonBinary','modalCli','nodeBinary'])await canonical(cfg[k]);
 await canonical(cfg.attemptStore,true,true);
 await canonical(path.join(r.root,'verify_web.py'));if(digest(await fs.readFile(path.join(r.root,'verify_web.py')))!==r.validator_sha256||r.validator_sha256!==digest((await planValidator(TARGETS))))fail('validator changed');
 for(const f of TARGETS)await canonical(path.join(r.root,f));
 return Object.freeze({...r,recordPath,targets:TARGETS,nodeBinary:cfg.nodeBinary});
}
return {planProject,createProject,validateProject};
}
async function validateExtended(recordPath,storageRoot){
 const r=JSON.parse(await fs.readFile(recordPath,'utf8'));
 const targets=parseMultiGoal('Create a new multi-file web project\nFiles: '+(Array.isArray(r.targets)?r.targets.join(', '):''));
 return operations(targets).validateProject(recordPath,storageRoot);
}
module.exports={parseMultiGoal,operations,validateExtended};
