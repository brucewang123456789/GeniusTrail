/** Read-only exact edit expansion. No writes, execution or new commit authority. */
import {createHash} from 'node:crypto';
import {VoderError} from './contracts.mjs';
import {normalizedCandidate} from '../../voder-orchestration/src/index.mjs';
// Filesystem ENOENT from an agent-provided path is a recoverable proposal error,
// not an opaque verifier / sandbox failure. Never return an absolute host path.
function missingSource(error,role,path){
 if(error?.code==='ENOENT'||error?.code==='ENOTDIR'){
  throw new VoderError('CANDIDATE_SOURCE_NOT_FOUND',
   `${role} not found in the pinned task checkout: ${JSON.stringify(path)}. `+
   'Use voder_list_files to locate an existing file, voder_read_file to obtain its exact SHA256, and resend a revised structured candidate. ' +
   'An absent baseline file cannot be edited and must not be invented.');
 }
 throw error;
}
async function readPinned(inspector,path,role){
 try{const result=await (inspector.readFull??inspector.read).call(inspector,{path});if(result.partial)throw new VoderError('PARTIAL_SOURCE','Exact edits require the full pinned preimage');return result;}
 catch(error){missingSource(error,role,path);}
}
async function existingPatchBase(inspector,path){
 try{return await inspector.safePath(path,{allowRoot:true,kind:'dir'});}
 catch(error){missingSource(error,'Patch base directory',path);}
}

// A literal Git patch is a transport artifact, not a reason to reject an
// otherwise well-formed edit. Correct only arithmetically wrong hunk counts;
// never synthesize source context, remove hunks, or alter +/-/space payload.
// Final bytes (not model bytes) remain pinned through Modal and Rust VGR.
export function repairLiteralPatchMetadata(text){
 if(typeof text!=='string'||!text.endsWith('\n'))
  return {content:text,repairedHunks:0};
 const lines=text.split('\n');lines.pop();
 let sections=0,headers=0,hunks=0,repairs=0,inside=false,oldHeader=false,newHeader=false;
 const out=[...lines];
 for(let i=0;i<lines.length;i++){
  const line=lines[i];
  if(line.startsWith('diff --git ')){
   if(sections&&!hunks)return {content:text,repairedHunks:0};
   sections++;inside=false;oldHeader=false;newHeader=false;hunks=0;
   continue;
  }
  if(!sections)continue;
  if(!inside){
   if(line.startsWith('--- ')){oldHeader=true;continue;}
   if(line.startsWith('+++ ')){newHeader=true;continue;}
   if(!line.startsWith('@@ '))continue;
  }
  if(line.startsWith('@@ ')){
   if(!oldHeader||!newHeader)return {content:text,repairedHunks:0};
   const match=/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$/.exec(line);
   if(!match)return {content:text,repairedHunks:0};
   let old=0,next=0,contentLines=0;
   let j=i+1;
   for(;j<lines.length;j++){
    const body=lines[j];
    if(body.startsWith('diff --git ')||body.startsWith('@@ '))break;
    if(body.startsWith('\\ No newline at end of file'))continue;
    const marker=body[0];
    if(![' ','+','-'].includes(marker))return {content:text,repairedHunks:0};
    if(marker!=='+' )old++;
    if(marker!=='-')next++;
    contentLines++;
   }
   if(!contentLines)return {content:text,repairedHunks:0};
   const computed=`@@ -${match[1]},${old} +${match[3]},${next} @@${match[5]}`;
   if(computed!==line){out[i]=computed;repairs++;}
   headers++;hunks++;inside=true;i=j-1;
  }
 }
 if(!sections||!headers)return {content:text,repairedHunks:0};
 return {content:repairs?out.join('\n')+'\n':text,repairedHunks:repairs};
}

export async function expandCandidate(candidate,inspector){
 if(candidate&&typeof candidate.relative_path==='string'&&/\.(?:patch|diff)$/i.test(candidate.relative_path)&&
   Object.keys(candidate).sort().join(',')==='new_content,old_sha256,relative_path'&&typeof candidate.new_content==='string'){
   const fixed=repairLiteralPatchMetadata(candidate.new_content);
   if(fixed.repairedHunks)candidate={...candidate,new_content:fixed.content};
 }
 if(candidate&&Object.hasOwn(candidate,'patches')){
  if(Object.keys(candidate).join(',')!=='patches'||!Array.isArray(candidate.patches)||candidate.patches.length<2||candidate.patches.length>16)throw new VoderError('INVALID_CANDIDATE','Expected exact batch scope');
  const patches=[];for(const member of candidate.patches){if(member?.patches||member?.unified_diff)throw new VoderError('INVALID_CANDIDATE','Nested batches and patch artifacts are not batch members');patches.push(await expandCandidate(member,inspector));}
  return normalizedCandidate({patches});
 }
 if(candidate&&Object.hasOwn(candidate,'unified_diff'))return expandPatchArtifact(candidate,inspector);
 if(!candidate||!Object.hasOwn(candidate,'edits'))return normalizedCandidate(candidate);
 const fail=(code,reason)=>{throw new VoderError(code,reason)};
 if(Array.isArray(candidate)||Object.keys(candidate).sort().join(',')!=='edits,old_sha256,relative_path'||
   typeof candidate.old_sha256!=='string'||!/^[a-f0-9]{64}$/.test(candidate.old_sha256)||
   !Array.isArray(candidate.edits)||candidate.edits.length<1||candidate.edits.length>32)
   fail('INVALID_EDITS','Expected bounded exact edits and pinned source SHA256');
 // Validate the existing candidate path contract before any filesystem read.
 normalizedCandidate({relative_path:candidate.relative_path,old_sha256:candidate.old_sha256,new_content:' '});
 const source=await readPinned(inspector,candidate.relative_path,'Candidate edit source');
 if(createHash('sha256').update(source.content).digest('hex')!==candidate.old_sha256)
   fail('STALE_SOURCE','Edit base differs from current source; read it again');
 const spans=candidate.edits.map(e=>{
   if(!e||Array.isArray(e)||Object.keys(e).sort().join(',')!=='new_text,old_text'||typeof e.old_text!=='string'||!e.old_text||typeof e.new_text!=='string'||e.new_text===e.old_text)
     fail('INVALID_EDITS','Each edit needs nonempty old_text and a changed new_text');
   const start=source.content.indexOf(e.old_text);
   if(start<0)fail('EDIT_NOT_FOUND','Exact old_text was not found in the pinned source');
   if(source.content.indexOf(e.old_text,start+1)!==-1)fail('AMBIGUOUS_EDIT','old_text must match exactly once; include more context');
   return {start,end:start+e.old_text.length,text:e.new_text};
 }).sort((a,b)=>a.start-b.start);
 for(let i=1;i<spans.length;i++)if(spans[i].start<spans[i-1].end)fail('OVERLAPPING_EDITS','Edits must not overlap in the original source');
 let content='',cursor=0;
 for(const span of spans){content+=source.content.slice(cursor,span.start)+span.text;cursor=span.end;}
 content+=source.content.slice(cursor);
 return normalizedCandidate({relative_path:candidate.relative_path,old_sha256:candidate.old_sha256,new_content:content});
}

// Deterministic unified diff from pinned source edits. No process, write or
// additional authority; the independent validator and VGR remain mandatory.
const sourceHash=value=>createHash('sha256').update(value,'utf8').digest('hex');
const lines=value=>value.match(/[^\n]*\n|[^\n]+$/g)??[];
function unifiedFile(path,before,after){
 const old=lines(before),next=lines(after);let prefix=0,suffix=0;
 while(prefix<old.length&&prefix<next.length&&old[prefix]===next[prefix])prefix++;
 if(prefix===old.length&&prefix===next.length)return '';
 while(suffix<old.length-prefix&&suffix<next.length-prefix&&old[old.length-1-suffix]===next[next.length-1-suffix])suffix++;
 const start=Math.max(0,prefix-3),oldEnd=Math.min(old.length,old.length-suffix+3),newEnd=Math.min(next.length,next.length-suffix+3);
 const oldCount=oldEnd-start,newCount=newEnd-start;
 const emit=(kind,line)=>kind+line+(line.endsWith('\n')?'':'\n\\ No newline at end of file\n');
 let text=`diff --git a/${path} b/${path}\n--- a/${path}\n+++ b/${path}\n@@ -${oldCount?start+1:start},${oldCount} +${newCount?start+1:start},${newCount} @@\n`;
 for(let i=start;i<prefix;i++)text+=emit(' ',old[i]);
 for(let i=prefix;i<old.length-suffix;i++)text+=emit('-',old[i]);
 for(let i=prefix;i<next.length-suffix;i++)text+=emit('+',next[i]);
 for(let i=old.length-suffix;i<oldEnd;i++)text+=emit(' ',old[i]);
 return text;
}
async function expandPatchArtifact(candidate,inspector){
 const fail=(code,message)=>{throw new VoderError(code,message);};
 if(Object.keys(candidate).sort().join(',')!=='old_sha256,relative_path,unified_diff'||typeof candidate.relative_path!=='string'||!candidate.relative_path.match(/\.(patch|diff)$/))
  fail('INVALID_PATCH_ARTIFACT','Pinned existing .patch/.diff output and bounded source edits required');
 normalizedCandidate({relative_path:candidate.relative_path,old_sha256:candidate.old_sha256,new_content:' '});
 const spec=candidate.unified_diff;
 if(!spec||Array.isArray(spec)||Object.keys(spec).sort().join(',')!=='base_dir,files'||typeof spec.base_dir!=='string'||!Array.isArray(spec.files)||spec.files.length<1||spec.files.length>16)
  fail('INVALID_PATCH_ARTIFACT','unified_diff needs base_dir and 1..16 distinct existing source files');
 await existingPatchBase(inspector,spec.base_dir);
 const target=await readPinned(inspector,candidate.relative_path,'Patch output preimage');
 if(sourceHash(target.content)!==candidate.old_sha256)fail('STALE_SOURCE','Patch output preimage changed');
 const seen=new Set(),chunks=[];
 for(const member of spec.files){
  if(!member||Array.isArray(member)||!['edits,old_sha256,relative_path','new_content,old_sha256,relative_path'].includes(Object.keys(member).sort().join(',')))
   fail('INVALID_PATCH_ARTIFACT','Each source file needs a pinned hash and exact edits or replacement text');
  normalizedCandidate({relative_path:member.relative_path,old_sha256:member.old_sha256,new_content:' '});
  // Normalize aliases before duplicate detection below.
  // Recover a redundant base_dir prefix, without scanning for guessed files.
  // The subsequent SHA256 check still pins the exact intended source.
  const relative=(spec.base_dir!==''&&spec.base_dir!=='.'&&member.relative_path.startsWith(`${spec.base_dir}/`))
   ? member.relative_path.slice(spec.base_dir.length+1) : member.relative_path;
  if(!relative||relative.startsWith('/'))fail('INVALID_PATCH_ARTIFACT','Missing project-relative member path');
  if(seen.has(relative))fail('INVALID_PATCH_ARTIFACT','Duplicate canonical source file');seen.add(relative);
  const path=spec.base_dir===''||spec.base_dir==='.'?relative:`${spec.base_dir}/${relative}`;
  if(path===candidate.relative_path)fail('INVALID_PATCH_ARTIFACT','Patch output cannot be its own input');
  const source=await readPinned(inspector,path,'Patch source');
  if(sourceHash(source.content)!==member.old_sha256)fail('STALE_SOURCE','Project source preimage changed; reread exact source');
  const expanded=await expandCandidate({...member,relative_path:path},inspector);
  const text=unifiedFile(relative,source.content,expanded.new_content);
  if(!text)fail('INVALID_PATCH_ARTIFACT','Each source replacement must change bytes');
  chunks.push({path:relative,text});
 }
 chunks.sort((a,b)=>a.path.localeCompare(b.path));
 return normalizedCandidate({relative_path:candidate.relative_path,old_sha256:candidate.old_sha256,new_content:chunks.map(x=>x.text).join('')});
}
