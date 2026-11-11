'use strict';
/** Deterministic, fail-closed authorization scope discovery; never executes model or code. */
const fs=require('node:fs');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const TOKEN=/(?<![\w.\/-])(?:[A-Za-z0-9_.-]+\/)*[A-Za-z0-9_.-]+\.py(?![\w./-])/g;
function error(code,message){return Object.assign(Error(message),{code});}
function gitFiles(root){
 let data;
 try{data=execFileSync('/usr/bin/git',['-C',root,'ls-files','-z','--','*.py'],{encoding:'utf8',timeout:5000,maxBuffer:4*1024*1024});}
 catch{throw error('CHAT_SCOPE_GIT_UNAVAILABLE','Cannot determine tracked Python files in this workspace. Configure a native VGR profile explicitly.');}
 return [...new Set(data.split('\0').filter(p=>p&&!p.startsWith('/')&&!p.split('/').some(s=>!s||s==='.'||s==='..')&&p.endsWith('.py')))].sort();
}
function parsePaths(goal){
 if(typeof goal!=='string'||Buffer.byteLength(goal,'utf8')>16384||goal.includes('\0'))throw error('CHAT_INVALID_GOAL','Invalid chat task.');
 return [...new Set([...goal.matchAll(TOKEN)].map(m=>m[0].replace(/^\.\//,'')))].slice(0,24);
}
function isTestFile(file){return /(?:^|\/)(?:test_[^/]+|[^/]+_test)\.py$/.test(file)||/(?:^|\/)tests?\//.test(file);}
/**
 * Returns null for insufficient explicit scope; never guesses a new file or chooses
 * between ambiguous similarly-named source/validator files.
 * fileList is injected only by unit tests; production always queries git itself.
 */
function resolveNativeIntent(root,goal,fileList){
 const tracked=fileList===undefined?gitFiles(root):fileList;
 const mentioned=parsePaths(goal);
 if(!mentioned.length)return null;
 const mapped=[];
 for(const token of mentioned){
  const exact=tracked.includes(token)?[token]:tracked.filter(p=>path.posix.basename(p)===token);
  if(exact.length>1)throw error('CHAT_SCOPE_AMBIGUOUS_PATH','File '+token+' matches multiple tracked paths; specify its repository-relative path.');
  if(exact.length===1) mapped.push(exact[0]);
  // Do not silently ignore explicitly named absent files: a new file cannot be authorized.
  else throw error('CHAT_SCOPE_FILE_NOT_TRACKED','File '+token+' is not tracked in this workspace. Native VGR cannot create a new file or authorize an absent target.');
 }
 const named=[...new Set(mapped)];
 const source=named.filter(p=>!isTestFile(p)),tests=named.filter(isTestFile);
 if(source.length!==1)throw error('CHAT_SCOPE_AMBIGUOUS_TARGET',source.length?'Multiple source files named; native VGR currently supports one scoped existing target.':'No modifiable source file named; specify the existing source file.');
 const target=source[0];let test=null;
 if(tests.length===1)test=tests[0];
 else if(tests.length>1)throw error('CHAT_SCOPE_AMBIGUOUS_TEST','Multiple validator scripts named; specify one independently tracked Python test.');
 else{
  const dirname=path.posix.dirname(target),stem=path.posix.basename(target,'.py');
  const matches=tracked.filter(p=>isTestFile(p)&&(
   path.posix.basename(p)==='test_'+stem+'.py'||path.posix.basename(p)===stem+'_test.py'));
  const sameDir=matches.filter(p=>path.posix.dirname(p)===dirname);
  if(sameDir.length===1)test=sameDir[0];
  else if(matches.length===1)test=matches[0];
  else if(matches.length>1)throw error('CHAT_SCOPE_AMBIGUOUS_TEST','Several plausible tests found. Name one Python test file explicitly.');
 }
 if(!test)return null;
 if(test===target)throw error('CHAT_TEST_TARGET_CONFLICT','The validator cannot be the editable source file.');
 // Do not follow symlinks as a way of discovering scope; createProfile repeats all checks.
 for(const relative of [target,test]){
  const full=path.join(root,relative);
  let st;try{st=fs.lstatSync(full);}catch{throw error('CHAT_SCOPE_FILE_MISSING','Selected tracked file is missing: '+relative);}
  if(!st.isFile()||st.isSymbolicLink())throw error('CHAT_SCOPE_UNSAFE_FILE','Selected file is not an ordinary file: '+relative);
 }
 return {target,test,targetFile:path.join(root,target),testFile:path.join(root,test)};
}
module.exports={resolveNativeIntent,parsePaths,isTestFile};
