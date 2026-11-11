'use strict';
/** Display-only scope diagnostics. Never select another file, replace a goal, or grant authority. */
const {parsePaths}=require('./native-intent.cjs');
const {startNotice}=require('./native-start.cjs');
function scopeFeedback(root,goal,error){
 const notice=startNotice(error);let requested=[];
 if(notice.type!=='input_required')return {...notice,goal,choices:[]};
 try{requested=parsePaths(goal);}catch{}
 const reasons={
  CHAT_SCOPE_FILE_NOT_TRACKED:'At least one requested file is not tracked in this project. Open the project containing that file, or correct its path.',
  CHAT_SCOPE_FILE_MISSING:'A requested tracked file is missing from this project. Restore it or correct the path.',
  CHAT_SCOPE_AMBIGUOUS_PATH:'A filename matches multiple paths in this project. Specify its full repository-relative path.',
  CHAT_SCOPE_AMBIGUOUS_TARGET:'More than one source file was named. Specify one existing source file for this write mode.',
  CHAT_SCOPE_AMBIGUOUS_TEST:'More than one independent validator matches. Specify the intended test path.',
  CHAT_SCOPE_UNSAFE_FILE:'The requested path is not an ordinary file. Symbolic links cannot be authorized.',
  CHAT_SCOPE_PLACEHOLDER:'Replace the example filename with a real path in the selected project.',
  CHAT_SCOPE_GIT_UNAVAILABLE:'This project has no readable Git file index. Open the intended Git project.'
 };
 return {...notice,goal,choices:[],scopeContext:{workspace:root,requested,reason:reasons[notice.code]||'Specify the intended existing source and independent validator in this project. Your original request is preserved.'}};
}
module.exports={scopeFeedback};
