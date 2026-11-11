#!/usr/bin/env node
/** Operator-only CLI, no implicit workspace discovery and no network downloads. */
import {InstalledSkills} from './index.mjs';
const [cmd,...args]=process.argv.slice(2);
const root=process.env.VODER_INSTALLED_SKILLS_ROOT??'/root/.voder/installed-skills';
async function main(){
 const store=new InstalledSkills({root});
 if(cmd==='list'&&args.length===0){console.log(JSON.stringify((await store.list()).map(({name,description,sha256})=>({name,description,sha256})),null,2));return;}
 if(cmd==='install'&&args.length===2){const result=await store.install({source:args[0],sha256:args[1]});console.log('VODER_SKILL_INSTALLED='+result.name);return;}
 if(cmd==='remove'&&args.length===1){await store.remove(args[0]);console.log('VODER_SKILL_REMOVED='+args[0]);return;}
 throw Error('USAGE: manage-skills.mjs list | install /absolute/path/SKILL.md sha256 | remove name');
}
main().catch(e=>{console.error('SKILL_MANAGEMENT_BLOCKED='+String(e.code??e.message).slice(0,128));process.exitCode=1;});
