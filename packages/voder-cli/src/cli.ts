#!/usr/bin/env node
import {readFile, cp, mkdir, rename, lstat, stat, readdir, open} from 'node:fs/promises';
import {existsSync, lstatSync} from 'node:fs';
import {join, resolve, isAbsolute} from 'node:path';
import {randomBytes} from 'node:crypto';
import {spawn} from 'node:child_process';
import {
  VENDORS,RUNTIMES,validVendor,validRuntime,validModel,validateEndpoint,parseOpts,requireOpts,strOpt,
  loadState,saveState,storeCredential,storeOpenCredential,removeCredential,credentialExists,ensurePrivate,repoRoot,assertWorkspace,
  assertInstalled,pluginAvailable,modelEnvironment,runProviderDoctor,printHelp,bridgePath,nodeExecutable,safeStateDir,
  sha256,type State,type OpenProfile,type Selection,
} from './core.js';
const P=process;
function usage(msg:string):never{throw new Error(msg)}
async function getSecret():Promise<string>{
 if(P.stdin.isTTY){
   const input=P.stdin;const out=P.stderr;const chars:string[]=[];
   out.write('API key (hidden): ');input.setRawMode(true);input.resume();input.setEncoding('utf8');
   try{
     return await new Promise<string>((resolvePromise,reject)=>{
       function cleanup(){input.removeListener('data',onData);input.removeListener('end',onEnd);}
       function onEnd(){cleanup();reject(new Error('input ended'))}
       function onData(chunk:string){
         for(const c of chunk){
           if(c==='\u0003'){cleanup();reject(new Error('interrupted'));return;}
           if(c==='\r'||c==='\n'){cleanup();out.write('\n');resolvePromise(chars.join(''));return;}
           if(c==='\u007f'){chars.pop();continue;}
           if(chars.length>=16000){cleanup();reject(new Error('secret too long'));return;}
           chars.push(c);
         }
       }
       input.on('data',onData);input.once('end',onEnd);
     });
   }finally{input.setRawMode(false);input.pause();}
 }
 throw new Error('interactive key entry requires TTY; pass --stdin to read from a pipe');
}
async function secretFromStdin():Promise<string>{
 let s='';for await(const c of P.stdin){s+=String(c);if(s.length>16002)throw new Error('key too long');}
 return s.replace(/[\r\n]+$/u,'');
}
const aliasValid=(s:string)=>/^[a-z][a-z0-9_-]{0,47}$/u.test(s) && !['cloud','open'].includes(s);
function ensurePlugin(id:string):'cloud'|'open'{if(id==='provider-cloud'||id==='cloud')return 'cloud';if(id==='provider-open'||id==='open')return 'open';return usage('plugin name must be provider-cloud or provider-open')}
async function installPlugin(root:string,id:'cloud'|'open'):Promise<void>{
 if(await pluginAvailable(root,id))return;
 const base=join(root,'packages');const destination=join(base,`voder-provider-${id}`);
 if(existsSync(destination))throw new Error('plugin directory already exists but is incomplete; refusing overwrite');
 const vendor=join(root,'packages/voder-cli/vendor',`voder-provider-${id}`);
 if(!existsSync(join(vendor,'dist/index.js')))throw new Error('trusted plugin distribution is missing');
 // Manifest binds the package source and build output to the installed release.
 const man=JSON.parse(await readFile(join(root,'packages/voder-cli/vendor/MANIFEST.json'),'utf8')) as Record<string,string>;
 for(const rel of [`voder-provider-${id}/src/index.ts`,`voder-provider-${id}/dist/index.js`]){
   const file=join(root,'packages/voder-cli/vendor',rel);
   if(sha256(await readFile(file))!==man[rel])throw new Error('plugin vendor integrity verification failed');
 }
 const tmp=join(base,`.voder-plugin-${id}-${randomBytes(8).toString('hex')}`);
 await cp(vendor,tmp,{recursive:true,errorOnExist:true,force:false});
 try{await rename(tmp,destination);}catch(e){await (await import('node:fs/promises')).rm(tmp,{recursive:true,force:true});throw e;}
}
async function commandPlugin(args:string[],root:string):Promise<void>{
 const [verb,name,...extra]=args;if(extra.length)usage('unexpected arguments');
 const state=await loadState();
 if(verb==='list'){
  if(name)usage('plugin list accepts no arguments');
  for(const id of ['cloud','open'] as const){console.log(`provider-${id}: ${state.plugins[id]?'enabled':'disabled'} / ${(await pluginAvailable(root,id))?'files-present':'not-installed'}`);}
  return;
 }
 if(!verb||!name)usage('usage: voder plugin install|remove provider-cloud|provider-open');
 const id=ensurePlugin(name);
 if(verb==='install'){
   await installPlugin(root,id);state.plugins[id]=true;await saveState(state);
   console.log(`PLUGIN_INSTALL=PASS provider-${id}`);
 }else if(verb==='remove'){
   // Deliberate soft uninstall: never remove any code, receipts or settings without backup.
   state.plugins[id]=false;if(state.active?.kind===id)state.active=null;
   await saveState(state);console.log(`PLUGIN_DISABLED=PASS provider-${id} (source preserved)`);
 }else usage('usage: voder plugin install|remove|list');
}
async function commandModel(args:string[],root:string):Promise<void>{
 const [verb,...rest]=args;const state=await loadState();
 if(verb==='login'){
   const {pos,opts}=parseOpts(rest);requireOpts(opts,['stdin']);
   if(pos.length!==1||!validVendor(pos[0]!))usage(`model login requires vendor: ${VENDORS.join(', ')}`);
   const key=opts.stdin===true?await secretFromStdin():await getSecret();
   await storeCredential(pos[0]!,key);console.log(`MODEL_LOGIN=PASS vendor=${pos[0]}`);return;
 }
 if(verb==='logout'){
   if(rest.length!==1)usage('model logout VENDOR|open:ALIAS');
   await removeCredential(rest[0]!);
   if(state.active?.kind==='cloud'&&state.active.vendor===rest[0]){state.active=null;await saveState(state);}
   if(state.active?.kind==='open'&&`open:${state.active.alias}`===rest[0]){state.active=null;await saveState(state);}
   console.log(`MODEL_LOGOUT=PASS ${rest[0]}`);return;
 }
 if(verb==='auth'){
   const {pos,opts}=parseOpts(rest);requireOpts(opts,['stdin']);
   if(pos.length!==1||!pos[0]?.startsWith('open:')||!aliasValid(pos[0].slice(5)))usage('model auth open:ALIAS [--stdin]');
   const alias=pos[0].slice(5);if(!state.openProfiles[alias])usage('unknown inference profile');
   const key=opts.stdin===true?await secretFromStdin():await getSecret();
   await storeOpenCredential(alias,key);console.log(`MODEL_AUTH=PASS open:${alias}`);return;
 }
 if(verb==='add'){
   const {pos,opts}=parseOpts(rest);requireOpts(opts,['runtime','endpoint','model','allow-loopback-http','replace']);
   if(pos.length!==1||!aliasValid(pos[0]!))usage('invalid alias: expected lowercase [a-z][a-z0-9_-]*');
   const runtime=strOpt(opts,'runtime'),endpoint=strOpt(opts,'endpoint'),model=strOpt(opts,'model');
   if(!runtime||!validRuntime(runtime)||!endpoint||!model||!validModel(model))usage('model add requires --runtime, --endpoint and --model');
   const allowLoopbackHttp=opts['allow-loopback-http']===true;
   const next={runtime,endpoint:validateEndpoint(endpoint,allowLoopbackHttp),model,allowLoopbackHttp};
   const old=state.openProfiles[pos[0]!];
   if(old&&JSON.stringify(old)!==JSON.stringify(next)&&opts.replace!==true)usage('existing inference profile differs; add --replace to change it');
   state.openProfiles[pos[0]!]=next;
   await saveState(state);console.log(`MODEL_PROFILE_ADDED=PASS ${pos[0]}`);return;
 }
 if(verb==='use'){
   const {pos,opts}=parseOpts(rest);requireOpts(opts,['protocol','endpoint']);
   if(pos.length!==1)usage('model use needs one model reference');
   const spec=pos[0]!;
   if(spec.startsWith('open:')){
     if(Object.keys(opts).length)usage('open model use accepts no protocol or endpoint override');
     const alias=spec.slice(5);if(!state.openProfiles[alias])usage('unknown open model profile');
     if(!state.plugins.open)usage('install provider-open first');assertInstalled(root,'open');
     state.active={kind:'open',alias};
   }else{
     const ix=spec.indexOf('/');if(ix<1||ix===spec.length-1)usage('cloud reference must be vendor/model');
     const vendor=spec.slice(0,ix),model=spec.slice(ix+1);
     if(!validVendor(vendor)||!validModel(model))usage('unknown vendor or invalid model ID');
     if(!state.plugins.cloud)usage('install provider-cloud first');assertInstalled(root,'cloud');
     const protocol=strOpt(opts,'protocol')??'chat';
     if(protocol!=='chat'&&protocol!=='responses')usage('protocol must be chat|responses');
     if(protocol==='responses'&&vendor!=='openai'&&vendor!=='custom')usage('Responses requires OpenAI or custom');
     if(!await credentialExists(vendor))usage(`missing key: voder model login ${vendor}`);
     const endpoint=strOpt(opts,'endpoint');if(vendor==='custom'&&!endpoint)usage('custom provider requires --endpoint HTTPS_URL');
     state.active={kind:'cloud',vendor,model,protocol,...(endpoint?{endpoint:validateEndpoint(endpoint)}:{})};
   }
   await saveState(state);console.log(`MODEL_SELECTED=PASS ${spec}`);return;
 }
 if(verb==='status'){
   if(rest.length)usage('model status has no arguments');
   const a=state.active;
   console.log(`MODEL_ACTIVE=${a?(a.kind==='cloud'?`${a.vendor}/${a.model} (${a.protocol})`:`open:${a.alias}`):'NONE'}`);
   console.log(`PLUGINS=cloud:${state.plugins.cloud?'enabled':'disabled'},open:${state.plugins.open?'enabled':'disabled'}`);
   for(const [alias,p]of Object.entries(state.openProfiles))console.log(`OPEN_PROFILE=${alias} (${p.runtime})`);
   return;
 }
 usage('model commands: login|add|use|status');
}
async function commandDoctor(args:string[],root:string):Promise<void>{
 const {pos,opts}=parseOpts(args);requireOpts(opts,['live']);if(pos.length)usage('doctor takes no positional args');
 const s=await loadState();
 if(!s.active)throw new Error('no active model; run voder model use');
 await assertWorkspace(root);const env=await modelEnvironment(s,root);
 console.log('VODER_CLI_PREFLIGHT=PASS');
 if(opts.live){
   console.log('VODER_LIVE_MODEL_REQUEST=RUNNING (may incur provider fees)');
   const r=await runProviderDoctor(root,env);
   console.log(`VODER_LIVE_MODEL_REQUEST=PASS provider=${r.providerId} model=${JSON.stringify(r.model)}`);
   console.log('VGR_EXECUTION_AUTHORITY=UNCHANGED');
 }else console.log('VODER_LIVE_MODEL_REQUEST=NOT_RUN (use --live)');
}
async function commandDaemon(args:string[],root:string):Promise<void>{
 const [verb,...rest]=args;if(verb!=='start')usage('daemon command: start');
 const {pos,opts}=parseOpts(rest);requireOpts(opts,['state-dir']);if(pos.length)usage('unexpected daemon arguments');
 const s=await loadState();await assertWorkspace(root);
 const env=await modelEnvironment(s,root);const daemon=join(root,'target/release/voderd');
 if(!existsSync(daemon))throw new Error('build Rust daemon first: cargo build -p voderd --release --locked');
 const state=strOpt(opts,'state-dir')??safeStateDir(root);if(!isAbsolute(state))throw new Error('state directory must be absolute');
 const statePath=resolve(state);
 if(existsSync(statePath)&&lstatSync(statePath).isSymbolicLink())throw new Error('daemon state directory symlink forbidden');
 await mkdir(statePath,{recursive:true,mode:0o700});const st=await stat(statePath);
 if((st.mode&0o077)!==0)throw new Error('daemon state dir is accessible to other users');
 const child=spawn(daemon,['--state-dir',statePath],{
   cwd:root,stdio:'inherit',env:{...P.env,...env,VODER_PROVIDER_NODE:nodeExecutable(),VODER_PROVIDER_CLI:bridgePath(root),NODE_OPTIONS:''}
 });
 const onSig=(sig:NodeJS.Signals)=>{child.kill(sig)};
 P.on('SIGINT',onSig);P.on('SIGTERM',onSig);
 const rc=await new Promise<number>((rs,rj)=>{child.on('error',rj);child.on('exit',(code)=>rs(code??1));});
 P.off('SIGINT',onSig);P.off('SIGTERM',onSig);P.exitCode=rc;
}
export async function main(argv:string[]):Promise<void>{
 const [sub,...rest]=argv;
 if(!sub||sub==='help'||sub==='--help'){console.log(printHelp());return;}
 if(Number(P.versions.node.split('.')[0])<22)throw new Error('Node.js >=22 required');
 const root=repoRoot();
 if(sub==='plugin')return commandPlugin(rest,root);
 if(sub==='model')return commandModel(rest,root);
 if(sub==='doctor')return commandDoctor(rest,root);
 if(sub==='daemon')return commandDaemon(rest,root);
 throw new Error('unknown command: use voder help');
}
if(import.meta.url===new URL(`file://${P.argv[1]}`).href){
 P.umask(0o077);
 main(P.argv.slice(2)).catch((error:unknown)=>{console.error(`VODER_CLI=BLOCKED: ${error instanceof Error?error.message:'unknown error'}`);P.exitCode=1;});
}
