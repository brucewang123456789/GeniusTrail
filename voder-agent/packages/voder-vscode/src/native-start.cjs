'use strict';
/** UI preparation only. No approval grant, task dispatch, profile write or model call. */
const {execFileSync}=require('node:child_process');
const {resolveNativeIntent}=require('./native-intent.cjs');
const placeholder=goal=>typeof goal==='string'&&/(?:实际(?:源码|源代码|测试)文件名|你的(?:源码|测试)文件名)\.py|(?:actual_source|actual_test|your_file)\.py/i.test(goal);
function inputError(code,message){return Object.assign(Error(message),{code});}
function nativeInputChoices(root){
 try{
  const files=execFileSync('/usr/bin/git',['-C',root,'ls-files','-z','--','*.py'],{encoding:'utf8',timeout:5000,maxBuffer:4*1024*1024}).split('\0').filter(Boolean);
  const choices=[];
  for(const file of files.slice(0,128)){
   if(!/^[A-Za-z0-9_./-]+\.py$/.test(file))continue;
   try{const scope=resolveNativeIntent(root,file,files);if(scope&&!choices.some(x=>x.target===scope.target))choices.push({target:scope.target,test:scope.test,goal:'修复'+scope.target+'中的问题，使'+scope.test+'通过；只能修改'+scope.target+'，不修改测试。'});}catch{}
   if(choices.length>=6)break;
  }
  return choices;
 }catch{return [];}
}
function startNotice(error){
 const code=String(error?.code||'CHAT_START_BLOCKED');
 if(code==='CHAT_NATIVE_DECLINED'||code==='CHAT_NATIVE_NOT_APPROVED')return {type:'not_started',status:'NOT_STARTED',code,message:'你已取消本次授权；任务未开始，没有调用模型或沙箱。'};
 const messages={CHAT_SCOPE_GIT_UNAVAILABLE:'当前目录没有可用的Git追踪清单，请打开实际项目仓库。',CHAT_SCOPE_FILE_NOT_TRACKED:'消息中的文件未被当前仓库追踪，请使用下方真实文件或填写正确的仓库相对路径。',CHAT_SCOPE_AMBIGUOUS_PATH:'存在同名文件，请填写完整的仓库相对路径。',CHAT_SCOPE_AMBIGUOUS_TARGET:'当前一次任务只能修改一个已有源码文件，请明确目标文件。',CHAT_SCOPE_AMBIGUOUS_TEST:'存在多个可能的测试文件，请明确使用哪个独立测试。',CHAT_SCOPE_FILE_MISSING:'选中的Git追踪文件当前不存在，请核对文件路径。',CHAT_SCOPE_UNSAFE_FILE:'选中的文件不是普通文件或属于软链接，请选择真实源码文件。',CHAT_INVALID_GOAL:'请输入任务内容，最长16KiB。'};
 if(code==='CHAT_INVALID_GOAL'||code.startsWith('CHAT_SCOPE_'))return {type:'input_required',status:'WAITING_FOR_INPUT',code,message:messages[code]||String(error.message||'请指定当前项目中的真实源码文件和独立测试文件。')};
 if(['CHAT_FILE_NOT_TRACKED','CHAT_TARGET_NOT_TRACKED'].includes(code))return {type:'not_started',status:'START_BLOCKED',code,message:'源码和独立测试必须已提交Git且没有未提交修改；当前文件不符合启动条件，任务未开始。'};
 return {type:'not_started',status:'START_BLOCKED',code,message:'启动检查未通过，任务尚未开始：'+String(error?.message||'Unknown preflight error').slice(0,350)};
}
module.exports={placeholder,inputError,nativeInputChoices,startNotice};
