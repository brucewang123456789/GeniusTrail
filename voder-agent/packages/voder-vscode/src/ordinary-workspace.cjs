'use strict';
/** Preserve shared/nested/dirty selected workspaces. Never chmod or initialize the user's parent repository. */
const fs=require('node:fs/promises'),{constants:C}=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),{execFileSync}=require('node:child_process');
const fail=(code,message)=>{throw Object.assign(Error(message),{code});};
const OMIT=new Set(['.git','node_modules','.venv','venv','__pycache__','.voder-tools','.voder-build']);
async function directory(p,privateMode=false){
 if(!path.isAbsolute(p)||await fs.realpath(p)!==p)fail('GENERAL_WORKSPACE','Canonical workspace path required');
 for(let q=p;;q=path.dirname(q)){if((await fs.lstat(q)).isSymbolicLink())fail('GENERAL_WORKSPACE','Linked workspace ancestors are not supported');if(q===path.dirname(q))break;}
 const s=await fs.lstat(p);if(!s.isDirectory())fail('GENERAL_WORKSPACE','Workspace must be a directory');
 if(privateMode&&((s.mode&0o077)!==0||(process.getuid&&s.uid!==process.getuid())))fail('GENERAL_WORKSPACE','Private workspace storage must be owned by this user and mode 0700');return s;
}
async function ordinaryWorkspace({root,storageRoot,readOnly=false}){
 const st=await directory(root);if(readOnly)return {root,isolated:false};
 const probe=(...args)=>execFileSync('/usr/bin/git',['-c','core.hooksPath=/dev/null','-C',root,...args],{encoding:'utf8',timeout:10000,stdio:['ignore','pipe','pipe'],env:{PATH:'/usr/bin:/bin',GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null',GIT_TERMINAL_PROMPT:'0',GIT_OPTIONAL_LOCKS:'0'}});
 let nested=false,repository=false,dirty=false;
 try{const top=probe('rev-parse','--show-toplevel').trim();repository=true;nested=top!==root;}catch{}
 // Match the native gateway's clean-worktree requirement, including untracked
 // files. Snapshot dirty inputs privately; never stage or erase the user's edits.
 if(repository&&!nested)dirty=probe('status','--porcelain=v1','--untracked-files=all').trim()!=='';
 if(!(st.mode&0o022)&&!nested&&!dirty)return {root,isolated:false};
 const reason=st.mode&0o022?'SHARED_PERMISSIONS':nested?'NESTED_REPOSITORY':'DIRTY_REPOSITORY';
 if(storageRoot===root||storageRoot.startsWith(root+path.sep))fail('WORKSPACE_COPY_RECURSIVE','Select a project outside private Voder storage');
 const storage=await directory(storageRoot);if((storage.mode&0o022)!==0||(process.getuid&&storage.uid!==process.getuid()))fail('GENERAL_WORKSPACE','Workspace storage must be owned by this user and not writable by other users');
 const parent=path.join(storageRoot,'working-copies');await fs.mkdir(parent,{mode:0o700}).catch(e=>{if(e.code!=='EEXIST')throw e;});await directory(parent,true);
 const target=path.join(parent,crypto.randomUUID());await fs.mkdir(target,{mode:0o700});
 let count=0,bytes=0;const records=[];
 async function copy(source,dest,relative=''){
  const before=await fs.lstat(source);if(before.isSymbolicLink())fail('WORKSPACE_COPY_SYMLINK','Workspace contains a symlink; original retained, no model request started: '+relative);
  if(before.isDirectory()){
   if(relative)await fs.mkdir(dest,{mode:0o700});
   for(const name of (await fs.readdir(source)).sort())if(!OMIT.has(name))await copy(path.join(source,name),path.join(dest,name),relative?relative+'/'+name:name);
   const after=await fs.lstat(source);if(after.ino!==before.ino||after.dev!==before.dev||after.mtimeMs!==before.mtimeMs)fail('WORKSPACE_COPY_CHANGED','Workspace changed while copying; no model request started');
   return;
  }
  if(!before.isFile()||before.nlink>1)fail('WORKSPACE_COPY_UNSAFE','Only independent regular workspace files can be copied: '+relative);
  if(++count>4096)fail('WORKSPACE_COPY_LIMIT','Selected workspace exceeds private-copy limits; select the relevant project folder. No model request started');
  const h=await fs.open(source,C.O_RDONLY|C.O_NOFOLLOW|C.O_NONBLOCK),out=await fs.open(dest,'wx',before.mode&0o111?0o700:0o600);
  const digest=crypto.createHash('sha256');let copied=0;
  try{const a=await h.stat();if(a.ino!==before.ino||a.dev!==before.dev||!a.isFile())fail('WORKSPACE_COPY_CHANGED','Workspace entry changed');
   const buffer=Buffer.alloc(64*1024);for(;;){const {bytesRead}=await h.read(buffer,0,buffer.length,null);if(!bytesRead)break;const chunk=buffer.subarray(0,bytesRead);digest.update(chunk);await out.writeFile(chunk);copied+=bytesRead;}
   const b=await h.stat();if(a.size!==b.size||a.mtimeMs!==b.mtimeMs||a.ctimeMs!==b.ctimeMs||copied!==before.size)fail('WORKSPACE_COPY_CHANGED','Workspace file changed while copying');await out.sync();
  }finally{await h.close();await out.close();}
  bytes+=copied;records.push({path:relative,sha256:digest.digest('hex')});
 }
 try{await copy(root,target);await directory(root);
 // Establish the copied bytes as the baseline ONLY in this newly created private repository.
 // Force-add preserves copied ignored files too; no original Git index or hooks are reused.
 const git=(...args)=>execFileSync('/usr/bin/git',['-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false','-C',target,...args],{encoding:'utf8',timeout:10000,stdio:['ignore','pipe','pipe'],env:{PATH:'/usr/bin:/bin',GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null',GIT_TERMINAL_PROMPT:'0'}});
 git('init','-q');git('add','-f','--all','--','.');
 git('-c','user.name=Voder','-c','user.email=task@voder.invalid','commit','--allow-empty','-qm','Voder private working copy baseline');
 if(git('status','--porcelain').trim())fail('WORKSPACE_COPY_CHANGED','Private workspace baseline is not clean; no model request started');
await fs.writeFile(path.join(parent,path.basename(target)+'.json'),JSON.stringify({schema:'voder.workspace-copy.v1',source:root,root:target,reason,files:records}),{flag:'wx',mode:0o600});}
 catch(e){e.message+='; partial private copy preserved at '+target;throw e;}
 return {root:target,originalRoot:root,isolated:true,reason,files:count,bytes};
}
module.exports={ordinaryWorkspace};
