import {writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {VoderSDK,VoderControlPlane,VoderJournal} from '../src/index.mjs';
const cwd=process.argv[2], journalDir=join(cwd,'private');
const goal='repair the fixture safely';
const native={async executeAuthorized(input){
 await writeFile(join(cwd,'native-started.json'),JSON.stringify({session_id:input.session_id,action_id:input.action_id}));
 await new Promise(()=>{});
}};
const modelHost={async createSession({tools}){
 const action=tools.find(t=>t.name==='voder_authorized_action');
 return {async prompt(){await action.execute({kind:'patch',relative_path:'src/input.txt',old_sha256:'0'.repeat(64),new_content:'new-content'});},dispose(){},async abort(){}};
}};
const sdk=new VoderSDK({cwd,native,modelHost});
const journal=await new VoderJournal({dir:journalDir,workspace:cwd}).initialize();
const controller=new VoderControlPlane({sdk,journal});
const {sessionId}=await controller.createSession({autoContinue:true});
await writeFile(join(cwd,'session-id'),sessionId);
await controller.submitTask(sessionId,{goal});
setInterval(()=>{},1000);
