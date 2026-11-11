'use strict';
/** Session launch authorization, not a model-generated execution plan. */
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const {pinnedManagedNode,cacheManagedNode}=require('./native-profile.cjs');
const fail=m=>{throw Object.assign(Error(m),{code:'GENERAL_WORKSPACE'});};
async function real(p,{file=false,privateFile=false}={}){
 if(typeof p!=='string'||!path.isAbsolute(p)||await fs.realpath(p)!==p)fail('Canonical absolute path required');
 for(let q=p;;q=path.dirname(q)){if((await fs.lstat(q)).isSymbolicLink())fail('Linked paths are not supported');if(q===path.dirname(q))break;}
 const s=await fs.lstat(p);if(file?!s.isFile():!s.isDirectory())fail('Unexpected path type');
 if(privateFile&&((s.mode&0o077)!==0||(process.getuid&&s.uid!==process.getuid())))fail('Private session authorization required');return s;
}
async function launchGeneral({root,agentRoot,storageRoot,memoryRoot=root,approvalMode='preapproved',readOnly=false}){
 if(!['interactive','preapproved'].includes(approvalMode)||typeof readOnly!=='boolean')fail('Invalid host authorization mode');
 await real(root);await real(agentRoot);await real(storageRoot);await real(memoryRoot);
 const host=path.dirname(agentRoot),infrastructure={binary:path.join(host,'.voder-build/release/voder-gateway'),daemonBinary:path.join(host,'.voder-build/release/voderd'),modalCli:path.join(host,'packages/voder-sandbox-modal/src/cli.ts')};
 if(approvalMode==='preapproved')for(const p of Object.values(infrastructure))await real(p,{file:true});
 const nodeBinary=await cacheManagedNode(await pinnedManagedNode(host,{details:true}),storageRoot),id=crypto.randomUUID();
 const dir=path.join(storageRoot,'general-sessions',id);await fs.mkdir(dir,{recursive:true,mode:0o700});await real(dir,{privateFile:true});
 const config={approvalMode,readOnly,memoryRoot,schema:'voder.general-session.v1',id,root,agentRoot,storageRoot,nodeBinary,infrastructure};
 const profile=path.join(dir,'session.json');await fs.writeFile(profile,JSON.stringify(config),{flag:'wx',mode:0o600});return {...config,profile};
}
async function readLaunch(profile){
 await real(profile,{file:true,privateFile:true});const c=JSON.parse(await fs.readFile(profile,'utf8'));
 if(c.schema!=='voder.general-session.v1'||!/^[a-f0-9-]{36}$/.test(c.id)||profile!==path.join(c.storageRoot,'general-sessions',c.id,'session.json'))fail('Session authorization identity differs');
 if(c.approvalMode!==undefined&&!['interactive','preapproved'].includes(c.approvalMode)||c.readOnly!==undefined&&typeof c.readOnly!=='boolean')fail('Invalid session authorization mode');
 await real(path.dirname(profile),{privateFile:true});await real(c.root);await real(c.agentRoot);await real(c.nodeBinary,{file:true});await real(c.memoryRoot??c.root);
 const host=path.dirname(c.agentRoot),expected={binary:path.join(host,'.voder-build/release/voder-gateway'),daemonBinary:path.join(host,'.voder-build/release/voderd'),modalCli:path.join(host,'packages/voder-sandbox-modal/src/cli.ts')};
 if(JSON.stringify(c.infrastructure)!==JSON.stringify(expected))fail('Infrastructure selection differs');if((c.approvalMode??'preapproved')==='preapproved')for(const p of Object.values(expected))await real(p,{file:true});return {...c,profile};
}
module.exports={launchGeneral,readLaunch,real};
