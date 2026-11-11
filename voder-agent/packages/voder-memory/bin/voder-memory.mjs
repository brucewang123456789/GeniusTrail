#!/usr/bin/env node
import {readFile} from 'node:fs/promises';
import {exportMemoryFile,importMemoryFile,renderMemoryContext} from '../src/index.mjs';
const usage=`Voder portable memory (local files; no network or model needed)
  voder memory export --input entries.json --select id1,id2 --output carry.vmem.json --profile PROJECT --confirm
  voder memory inspect --file carry.vmem.json
  voder memory render --file carry.vmem.json  # outputs explicit user-data context for copy/paste
  voder memory verify --file carry.vmem.json

Entries JSON: {"entries":[{"id":"stable-slug","kind":"project","scope":"project","title":"...","content":"..."}]}
Kind: preference|project|decision|constraint|open_item|reference; scope: user|project|session.
Only selected IDs are exported; --confirm signifies explicit user approval. Never export credentials.\n`;
function argumentsFor(rest){
 const opts={};
 for(let i=0;i<rest.length;i++){
  const a=rest[i];if(!/^--[a-z]+$/.test(a)||Object.hasOwn(opts,a.slice(2)))throw Error('Unknown or duplicate memory option');
  const key=a.slice(2);
  if(key==='confirm'){opts[key]=true;continue;}
  if(!['input','select','output','profile','file'].includes(key)||!rest[i+1]||rest[i+1].startsWith('--'))throw Error('Missing or unknown memory option: '+a);
  opts[key]=rest[++i];
 }
 return opts;
}
export async function run(args){
 const [op,...rest]=args;
 if(!op||op==='help'||op==='--help'){process.stdout.write(usage);return;}
 const opt=argumentsFor(rest);
 if(op==='export'){
  if(Object.keys(opt).some(k=>!['input','select','output','profile','confirm'].includes(k))||!opt.input||!opt.select||!opt.output||opt.confirm!==true)throw Error('Export requires --input --select --output --confirm');
  const raw=await readFile(opt.input);
  if(raw.length>131072)throw Error('Entries input exceeds 128KiB');
  const payload=JSON.parse(raw.toString('utf8'));
  if(!payload||Object.keys(payload).some(k=>k!=='entries')||!Array.isArray(payload.entries))throw Error('Expected {"entries":[...]}');
  const receipt=await exportMemoryFile({entries:payload.entries,selectedIds:opt.select.split(','),outputPath:opt.output,profile:opt.profile,userConfirmed:true});
  process.stdout.write(JSON.stringify(receipt,null,2)+'\n');return;
 }
 if(['inspect','verify','render'].includes(op)){
  if(Object.keys(opt).length!==1||!opt.file)throw Error(op+' requires --file PATH');
  const b=await importMemoryFile(opt.file);
  if(op==='render')process.stdout.write(renderMemoryContext([b])+'\n');
  else if(op==='inspect')process.stdout.write(JSON.stringify({schema:b.schema,profile:b.profile,sha256:b.sha256,entries:b.entries.map(e=>({id:e.id,kind:e.kind,scope:e.scope,title:e.title}))},null,2)+'\n');
  else process.stdout.write(`MEMORY_VERIFY=PASS entries=${b.entries.length} sha256=${b.sha256} (integrity, NOT authenticity)\n`);
  return;
 }
 throw Error('Invalid command.\n'+usage);
}
if(import.meta.url===new URL(`file://${process.argv[1]}`).href){
 process.umask(0o077);
 run(process.argv.slice(2)).catch(e=>{process.stderr.write(`VODER_MEMORY=BLOCKED ${e?.code??'INVALID_REQUEST'}: ${e.message}\n`);process.exitCode=1;});
}
