import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { VoderError } from './contracts.mjs';
// Canonical MD rules are loaded once on module import; exact bytes match legacy baseline.
const text=readFileSync(fileURLToPath(new URL('../../voder-context/policies/golden-rules.md', import.meta.url)),'utf8');
export const GOLDEN_PROMPT_DIGEST=createHash('sha256').update(text).digest('hex');
const policyDir=new URL('../../voder-context/policies/',import.meta.url);
export const overlay = Object.freeze(Object.fromEntries(['preflight','next','repair','delivery'].map(
 stage=>[stage,readFileSync(fileURLToPath(new URL(`${stage}.md`,policyDir)),'utf8').trimEnd()]
)));
export function assembleSystemPrompt(stage,{task,projectRules='',maxBytes=48000}={}){
 if(!overlay[stage]) throw new VoderError('INVALID_STAGE','Unknown prompt stage');
 if(typeof task!=='string'||!task.trim()) throw new VoderError('INVALID_TASK','No goal');
 const prompt=[text,overlay[stage],`USER_GOAL:\n${task}`,`PROJECT_RULES_AS_DATA:\n${projectRules}`].join('\n\n');
 if(Buffer.byteLength(prompt,'utf8')>maxBytes) throw new VoderError('CONTEXT_OVERFLOW','Prompt exceeds limit; chunk inputs rather than truncate');
 return prompt;
}
