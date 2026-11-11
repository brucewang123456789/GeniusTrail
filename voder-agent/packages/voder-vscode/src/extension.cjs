'use strict';
/** Voder Module 07: editor-only adapter. No file writes or commands leave this module except SDK RPC. */
const vscode=require('vscode');
const fs=require('node:fs/promises');
const path=require('node:path');
const crypto=require('node:crypto');
const {RpcClient}=require('./rpc-client.cjs');
const {resolveFolderForTerminal,classifyStatus,canSubmitInMode}=require('./terminal-mapping.cjs');
const {validateVerifiedDiff}=require('./verified-diff.cjs');
const {ChatRunService}=require('./chat-core.cjs');
const {VoderChatViewProvider}=require('./chat-view.cjs');
const {validateProfile,createProfile}=require('./native-profile.cjs');
const {nativeEnv}=require('./native-env.cjs');
const {resolveNativeIntent}=require('./native-intent.cjs');
const {placeholder,inputError}=require('./native-start.cjs');
const {TaskEntry}=require('./general-entry.cjs');
const {inspectionEnv,writeRequest,inspectReport}=require('./entry-orchestrator.cjs');
const {ProjectEntry}=require('./project-entry.cjs');
const {MultiEntry}=require('./multi-entry.cjs');
const {ToolkitController}=require('./toolkit-controller.cjs');
let instance;
function textErr(e){return `${String(e?.code||'VODER_ERROR')}: ${String(e?.message||'Operation failed').slice(0,240)}`;}
class SessionNode extends vscode.TreeItem {
  constructor(folder,record){
    super(`${(record.session_id||'').slice(0,8)} · ${record.status||'UNKNOWN'}`,vscode.TreeItemCollapsibleState.None);
    this.description=folder.name;this.tooltip=`${folder.uri.fsPath}\nSession: ${record.session_id}\n${record.status}`;
    this.contextValue='voderSession';this.iconPath=new vscode.ThemeIcon(record.status==='VERIFIED'?'verified':record.status==='FAILED'?'error':'terminal');
    this.command={command:'voder.showStatus',title:'Inspect Voder session',arguments:[{folder,id:record.session_id}]};
  }
}
class VoderExtension {
  constructor(context){
    this.context=context;this.clients=new Map();this.connecting=new Map();this.terminalMap=new WeakMap();this.terminalBindNotices=new WeakSet();this.bindingPending=new WeakMap();this.terminalSet=new Set();this.activeRuns=new Map();this.eventCounts=new Map();this.eventMetadata=new Map();this.diffDocuments=new Map();this._secretQueue=Promise.resolve();this.repairReservations=new Set();
    this.chatRuns=new ChatRunService((folder,profile,{readOnly=false}={})=>this._open(folder,readOnly?{mode:'draft',gateway:''}:profile?{mode:'native',gateway:profile}:{}));this.chatSessionFolder=new Map();this.chatReviewProfiles=new Map();this.chatView=new VoderChatViewProvider(this);this.webProjects=new ProjectEntry(this);this.multiProjects=new MultiEntry(this);this.generalTasks=new TaskEntry(this);this.toolkit=new ToolkitController(this);
    this.emitter=new vscode.EventEmitter();this.output=vscode.window.createOutputChannel('Voder — evidence metadata');
    this.status=vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left,70);this.status.command='voder.showStatus';this.status.text='$(shield) Voder';this.status.tooltip='Voder: SDK+VGR session status';this.status.show();
    context.subscriptions.push(this.emitter,this.output,this.status);
  }
  get onDidChangeTreeData(){return this.emitter.event;}
  refresh(){this.emitter.fire();}
  _config(folder){return vscode.workspace.getConfiguration('voder',folder?.uri);}
  _journal(folder){const hash=crypto.createHash('sha256').update(folder.uri.fsPath).digest('hex').slice(0,20);
    return path.join(this.context.globalStorageUri.fsPath,'sessions-'+hash);
  }
  async _validateAgent(root){
    if(!root){
      // Discover only inside explicitly opened trusted project roots. Never scan arbitrary home folders.
      const candidates=[process.env.VODER_AGENT_ROOT,...(vscode.workspace.workspaceFolders||[]).flatMap(f=>[f.uri.fsPath,path.join(f.uri.fsPath,'voder-agent')]),path.resolve(__dirname,'../runtime/voder-agent'),path.resolve(__dirname,'../../..')];
      for(const candidate of candidates){
        if(typeof candidate!=='string'||!path.isAbsolute(candidate))continue;
        try{const s=await fs.lstat(path.join(candidate,'packages/voder-sdk/src/index.mjs'));if(s.isFile()&&!s.isSymbolicLink()){root=candidate;break;}}catch(e){if(e.code!=='ENOENT')throw e;}
      }
    }
    if(!root||!path.isAbsolute(root))throw Object.assign(new Error('Open the voder-agent project or set VODER_AGENT_ROOT in the remote extension host'),{code:'VODER_AGENT_ROOT_REQUIRED'});
    const trusted=await fs.realpath(root),st=await fs.lstat(root);
    if(!st.isDirectory()||st.isSymbolicLink())throw new Error('Agent root must be an existing real directory');
    const cli=path.join(trusted,'packages','voder-cli','bin','voder-cli.mjs');
    const sdk=path.join(trusted,'packages','voder-sdk','src','index.mjs');
    for(const f of [cli,sdk]){const s=await fs.lstat(f);if(!s.isFile()||s.isSymbolicLink())throw Error('Missing or unsafe Voder SDK/CLI file: '+f);}
    return cli;
  }
  async _open(folder,overrides={}){
    if(!vscode.workspace.isTrusted)throw Error('Workspace Trust is required before spawning Voder');
    if(!folder)throw Error('Select an explicit workspace folder before creating an Agent session');
    const overrideMode=overrides.mode??this._config(folder).get('mode','native');
    const overrideGateway=overrides.gateway??this._config(folder).get('nativeGatewayConfig','').trim()??'';
    const key=folder.uri.fsPath+'\0'+overrideMode+'\0'+overrideGateway+(overrides.toolkit===true?'\0toolkit':'')+(overrides.generalTask===true?'\0general-task':'');
    if(this.clients.get(key)?.alive)return this.clients.get(key);
    if(this.connecting.has(key))return this.connecting.get(key);
    const pending=(async()=>{
      const config=this._config(folder),root=config.get('agentRoot','').trim()||(this.context.globalState?.get('voder.chat.agentRoot')||''),mode=overrides.mode??config.get('mode','native');
      const gateway=overrides.gateway??(config.get('nativeGatewayConfig','').trim()||(process.env.VODER_NATIVE_GATEWAY_CONFIG||'').trim());
      if(!canSubmitInMode(vscode.workspace.isTrusted,mode,gateway))throw Error(mode==='native'?'Set voder.nativeGatewayConfig to a private real VGR gateway JSON or explicitly choose unverified draft mode':'Invalid Voder mode');
      const cli=await this._validateAgent(root);
      let exe=overrides.nodeExecutable??config.get('nodeExecutable','node');
      if(!exe||typeof exe!=='string'||/[\r\n\0]/.test(exe))throw Error('Invalid Node executable');
      const embeddedAgent=path.resolve(__dirname,'../runtime/voder-agent');
      if(exe==='node'&&path.resolve(path.dirname(cli),'../../..')===embeddedAgent){
        if(process.platform!=='linux'||process.arch!=='x64')throw Object.assign(Error('Bundled Voder native runtime supports Linux x64 only. Connect to a Linux x64 workspace through VS Code Remote SSH.'),{code:'VODER_NATIVE_PLATFORM_UNSUPPORTED'});
        const managed=path.resolve(__dirname,'../runtime/.voder-tools/node/bin/node');
        const managedStat=await fs.lstat(managed);
        if(!managedStat.isFile()||managedStat.isSymbolicLink()||(managedStat.mode&0o111)===0)throw Object.assign(Error('Bundled Node runtime missing or unsafe'),{code:'VODER_BUNDLED_NODE_INVALID'});
        exe=managed;
      }
      const journal=this._journal(folder);await fs.mkdir(journal,{recursive:true,mode:0o700});
      const args=[overrides.generalTask===true?path.join(__dirname,'general-control.mjs'):overrides.toolkit===true?path.join(__dirname,'tool-control.mjs'):cli,'rpc','--workspace',folder.uri.fsPath,'--session-dir',journal];
      if(overrides.toolkit===true||overrides.generalTask===true)args.push('--agent-root',path.resolve(path.dirname(cli),'../../..'));
      if(overrides.toolkit!==true&&overrides.generalTask!==true)args.push('--dialogue-storage-root',this.context.globalStorageUri.fsPath);
      if(mode==='draft')args.push('--draft');else args.push('--gateway-config',gateway);
      const rpc=new RpcClient({command:exe,args,cwd:folder.uri.fsPath,env:overrides.generalTask===true?require('./model-env.cjs').modelEnv(process.env):mode==='native'?nativeEnv(process.env):inspectionEnv(process.env),diagnosticFile:path.join(this.context.globalStorageUri.fsPath,'cli-stderr-diagnostics.jsonl')});
      rpc.on('event',params=>{
        const ev=params.event||{};this.eventCounts.set(params.sessionId,(this.eventCounts.get(params.sessionId)||0)+1);
        const safe={};for(const key of ['sequence','type','state','tool','code','status','candidate_digest','action_id','verdict'])if(typeof ev[key]==='number'||(typeof ev[key]==='string'&&ev[key].length<=160))safe[key]=ev[key];
        const history=this.eventMetadata.get(params.sessionId)||[];history.push(safe);if(history.length>256)history.shift();this.eventMetadata.set(params.sessionId,history);
        if(config.get('showEventMetadata',true))this.output.appendLine(`[${folder.name} ${params.sessionId.slice(0,8)}] ${String(ev.sequence||'')} ${String(ev.type||'event').slice(0,72)} ${String(ev.state||'').slice(0,50)}`);
        this.refresh();
      });
      rpc.on('disconnect',err=>{
        if(this.clients.get(key)===rpc)this.clients.delete(key);
        // Detached live bindings are historical only: a reopened control process cannot submit into them.
        for(const t of [...this.terminalSet]){const b=this.terminalMap.get(t);if(b?.folder.uri.fsPath===folder.uri.fsPath){this.terminalMap.delete(t);this.terminalSet.delete(t);}}
        this._statusForTerminal();
        this.output.appendLine(`[${folder.name}] Control plane DISCONNECTED (${err.code}); native reconciliation required for in-flight tools`);this.refresh();
      });
      let initialized;
      try{initialized=await rpc.request('initialize',{},15000);
        if(initialized?.protocol!=='voder.control.v1'||initialized?.capabilities?.vgr_commit!=='native_only'||initialized?.capabilities?.create!==true)
          throw Error('Unsupported Voder protocol/capabilities; refusing to execute');
      }catch(e){rpc.close('RPC_INIT_FAILED');throw e;}
      rpc.referenceContextCapable=initialized.capabilities.reference_context==='voder.references.v1';rpc.referenceFrameBytes=initialized.capabilities.reference_frame_bytes;rpc.recoveryCapable=initialized.capabilities.native_resume===true;rpc.diffCapable=initialized.capabilities.session_scoped_diff===true;
      this.clients.set(key,rpc);
      if(mode==='native'&&config.get('autoRecovery',false)===true&&rpc.recoveryCapable){
        void this._autoRecoverTrusted(folder,rpc).catch(e=>this.output.appendLine(`[${folder.name}] RECOVERY_BLOCKED ${textErr(e)}`));
      }
      return rpc;
    })();
    this.connecting.set(key,pending);
    try{return await pending;}finally{this.connecting.delete(key);}
  }
  async _agentRootForChat(folder){
    const configured=this._config(folder).get('agentRoot','').trim()||(this.context.globalState?.get('voder.chat.agentRoot')||'');
    try{return path.resolve(path.dirname(await this._validateAgent(configured)),'../../..');}
    catch(e){
      if(e.code!=='VODER_AGENT_ROOT_REQUIRED')throw e;
      const selected=await vscode.window.showOpenDialog({title:'Select your Voder agent source folder (contains packages/voder-cli and packages/voder-sdk)',canSelectFiles:false,canSelectFolders:true,canSelectMany:false,defaultUri:folder.uri});
      if(!selected?.length)throw Object.assign(Error('Voder agent directory selection cancelled'),{code:'CHAT_AGENT_ROOT_REQUIRED'});
      const cli=await this._validateAgent(selected[0].fsPath);
      const root=path.resolve(path.dirname(cli),'../../..');
      await this.context.globalState.update('voder.chat.agentRoot',root);
      return root;
    }
  }
  _chatProfileKey(folder){return 'voder.chat.native.profile.'+crypto.createHash('sha256').update(folder.uri.fsPath).digest('hex');}
  async ensureNativeChat(folder,{goal=null,forceSelection=false,requestApproval=null}={}){
    if(!this.trusted())throw Object.assign(Error('Trusted workspace required for native writes'),{code:'CHAT_WORKSPACE_UNTRUSTED'});
    const key=this._chatProfileKey(folder);
    // Resolve explicit user intent before considering a persisted profile, to prevent
    // accidentally authorizing a different target from the previous chat turn.
    if(!forceSelection&&placeholder(goal))throw inputError('CHAT_SCOPE_PLACEHOLDER','消息中仍是示例占位文件名，请选择下面列出的真实文件，或填写项目内的实际路径。');
    const scope=goal?resolveNativeIntent(folder.uri.fsPath,goal):null;
    if(!forceSelection&&goal!==null&&!scope)throw inputError('CHAT_SCOPE_INPUT_REQUIRED','还不能确定修改哪个文件及使用哪个独立测试，请补充当前项目的真实文件名；任务尚未开始。');
    const matches=(profile)=>(!goal||!!scope)&&(!scope||(profile.target===scope.target&&
      (profile.validatorTest===scope.test||profile.validator?.endsWith(' '+scope.test))));
    if(!forceSelection){
      const saved=this.context.globalState?.get(key);
      if(saved){try{const profile=await validateProfile(saved,folder);if(matches(profile)){
        await this._agentRootForChat(folder);return profile;
      }this.output.appendLine('[CHAT NATIVE] Prior approved profile has a different target/validator; new scoped approval required.');}
      catch(e){this.output.appendLine('[CHAT NATIVE] Saved profile blocked: '+textErr(e));}}
      const configured=this._config(folder).get('nativeGatewayConfig','').trim()||(process.env.VODER_NATIVE_GATEWAY_CONFIG||'').trim();
      if(configured){try{const profile=await validateProfile(configured,folder);if(matches(profile)){
        await this._agentRootForChat(folder);return profile;
      }this.output.appendLine('[CHAT NATIVE] Operator profile does not cover requested target/validator.');}
      catch(e){this.output.appendLine('[CHAT NATIVE] Existing profile blocked: '+textErr(e));}}
    }
    let profile;
    if(scope&&!forceSelection){
      // Exact Git-tracked source + independent test are already identifiable.
      // Prepare a private candidate profile; only an explicit modal approval persists it.
      const agentRoot=await this._agentRootForChat(folder);
      profile=await createProfile({folder,agentRoot,storageRoot:this.context.globalStorageUri.fsPath,
        targetFile:scope.targetFile,testFile:scope.testFile});
    }else{
      const chosen=await vscode.window.showQuickPick([
        {label:'Create a scoped native profile',value:'create',description:'Choose an existing Git-tracked file and an independent Python test'},
        {label:'Use an existing native VGR gateway JSON',value:'existing',description:'Must match this exact workspace'},
        {label:'Cancel without executing',value:'cancel'}
      ],{placeHolder:'Native verification requires a scoped authorization; no model call until approved'});
      if(!chosen||chosen.value==='cancel')throw Object.assign(Error('Native authorization was cancelled; nothing executed.'),{code:'CHAT_NATIVE_NOT_APPROVED'});
      if(chosen.value==='existing'){
        const selected=await vscode.window.showOpenDialog({title:'Choose private native VGR gateway.json (0600)',canSelectFiles:true,canSelectFolders:false,canSelectMany:false,filters:{JSON:['json']}});
        if(!selected?.length)throw Object.assign(Error('Native gateway selection cancelled.'),{code:'CHAT_NATIVE_NOT_APPROVED'});
        profile=await validateProfile(selected[0].fsPath,folder);
      }else{
        const selected=await vscode.window.showOpenDialog({title:'Select existing Git-tracked source file to modify',defaultUri:folder.uri,canSelectFiles:true,canSelectFolders:false,canSelectMany:false});
        if(!selected?.length)throw Object.assign(Error('No source file selected.'),{code:'CHAT_NATIVE_NOT_APPROVED'});
        const tests=await vscode.window.showOpenDialog({title:'Select separate Git-tracked Python test file',defaultUri:folder.uri,canSelectFiles:true,canSelectFolders:false,canSelectMany:false,filters:{Python:['py']}});
        if(!tests?.length)throw Object.assign(Error('No independent test file selected.'),{code:'CHAT_NATIVE_NOT_APPROVED'});
        const agentRoot=await this._agentRootForChat(folder);
        profile=await createProfile({folder,agentRoot,storageRoot:this.context.globalStorageUri.fsPath,targetFile:selected[0].fsPath,testFile:tests[0].fsPath});
      }
      if(!matches(profile))throw Object.assign(Error('Selected authorization does not match the files explicitly named in this task.'),{code:'CHAT_NATIVE_SCOPE_MISMATCH'});
    }
    const permission=typeof requestApproval==='function'?(await requestApproval({target:profile.target,validator:scope?.test||profile.validator})?'Approve scoped native':null):await vscode.window.showWarningMessage('Approve Voder native execution for THIS workspace, existing file '+profile.target+' and independent validator '+profile.validator+'? Model and Modal API usage may incur charges. Rust VGR alone decides commit. The approval will be reused ONLY while this exact file, validator and workspace remain valid.',{modal:true},'Approve scoped native');
    if(permission!=='Approve scoped native')throw Object.assign(Error('Native execution was not approved; no model or sandbox call.'),{code:typeof requestApproval==='function'?'CHAT_NATIVE_DECLINED':'CHAT_NATIVE_NOT_APPROVED'});
    await this._agentRootForChat(folder);
    // Persist only after validation AND consent; cancellation leaves prior approval intact.
    await this.context.globalState.update(key,profile.path);
    return profile;
  }
  async configureNativeChat(){const folder=await this._folder();if(!folder)return;return this.ensureNativeChat(folder,{forceSelection:true});}
  trusted(){return vscode.workspace.isTrusted;}
  chatWorkspace(){
    const folders=vscode.workspace.workspaceFolders||[];
    const uri=vscode.window.activeTextEditor?.document?.uri;
    return (uri&&vscode.workspace.getWorkspaceFolder(uri))||(folders.length===1?folders[0]:null);
  }
  chooseFolder(){return this._folder();}
  async _folder(preferEditor=true){
    const folders=vscode.workspace.workspaceFolders||[];
    if(preferEditor&&vscode.window.activeTextEditor){const f=vscode.workspace.getWorkspaceFolder(vscode.window.activeTextEditor.document.uri);if(f)return f;}
    if(folders.length===1)return folders[0];
    if(!folders.length)throw Error('Open a local, WSL, SSH, or Dev Container folder first');
    const selected=await vscode.window.showQuickPick(folders.map(folder=>({label:folder.name,description:folder.uri.fsPath,folder})),{placeHolder:'Choose an exact Voder workspace'});
    return selected?.folder||null;
  }
  async _bind(terminal,folder){
    if(this.terminalMap.has(terminal))return this.terminalMap.get(terminal);
    if(this.bindingPending.has(terminal))return this.bindingPending.get(terminal);
    const pending=(async()=>{
      const rpc=await this._open(folder);
      const autoContinue=this._config(folder).get('autoRecovery',false)===true&&this._config(folder).get('mode','native')==='native';
      if(autoContinue&&!rpc.recoveryCapable)throw Error('Rust VGR recovery capability unavailable');
      const created=await rpc.request('session.create',{autoContinue},12000);
      if(!/^[0-9a-f-]{36}$/i.test(created?.sessionId||''))throw Error('Invalid SDK session identity');
      await rpc.request('session.subscribe',{sessionId:created.sessionId,after:0},12000);
      const binding={id:created.sessionId,folder,terminal,autoContinue};this.terminalMap.set(terminal,binding);this.terminalSet.add(terminal);
      this.output.appendLine(`[${folder.name}] Bound terminal to Voder session ${created.sessionId}; no tools authorized`);
      this.refresh();this._statusForTerminal();return binding;
    })();
    this.bindingPending.set(terminal,pending);
    try{return await pending;}finally{this.bindingPending.delete(terminal);}
  }
  async onTerminalOpen(terminal){
    if(!vscode.workspace.isTrusted||!vscode.workspace.getConfiguration('voder').get('autoBindTerminals',true))return;
    const folders=vscode.workspace.workspaceFolders||[];
    const folder=resolveFolderForTerminal(folders,terminal);
    if(!folder){
      // A restored terminal outside this workspace is normal; Chat has its own folder identity.
      // Never guess a workspace or open an SDK connection just to suppress a notice.
      if(folders.length>1&&!this.terminalBindNotices.has(terminal)){this.terminalBindNotices.add(terminal);this.output.appendLine('[Voder] 终端暂未绑定：尚未确定所属工作区；不影响Voder聊天，使用终端任务时再选择。');}
      return;
    }
    try{await this._bind(terminal,folder);}catch(e){this.output.appendLine(`[Voder] Terminal not bound: ${textErr(e)}`);}
  }
  onTerminalClose(terminal){this.terminalMap.delete(terminal);this.terminalSet.delete(terminal);this._statusForTerminal();this.refresh();
    // Never cancel VGR work just because the IDE terminal closed.
  }
  _statusForTerminal(){const b=this.terminalMap.get(vscode.window.activeTerminal);
    this.status.text=b?`$(shield) Voder ${b.id.slice(0,8)}`:'$(shield) Voder';this.status.tooltip=b?`Session ${b.id} | ${b.folder.name}`:'No active terminal Voder binding';
  }
  async bindTerminal(){const terminal=vscode.window.activeTerminal;if(!terminal)throw Error('Open an IDE terminal first');
    const already=this.terminalMap.get(terminal);if(already){vscode.window.showInformationMessage('Terminal already bound: '+already.id);return already;}
    const folder=resolveFolderForTerminal(vscode.workspace.workspaceFolders||[],terminal)||await this._folder();
    if(folder)return this._bind(terminal,folder);
  }
  async openTerminal(folder){folder??=await this._folder();if(!folder)return;
    const terminal=vscode.window.createTerminal({name:'Voder Session',cwd:folder.uri});terminal.show();
    // Only create an idle mapping, not a shell command. onDidOpenTerminal may race.
    return this._bind(terminal,folder);
  }
  async _pickBinding(){
    const active=vscode.window.activeTerminal,b=active&&this.terminalMap.get(active);if(b)return b;
    if(active)return await this.bindTerminal();
    return this.openTerminal();
  }
  async submitTask(withSelection=false,request,selectedBinding){
    const binding=selectedBinding||await this._pickBinding();if(!binding)return;
    const goal=request&&typeof request==='object'?request.goal:await vscode.window.showInputBox({title:'Voder Agent task',prompt:'State the goal and acceptance criteria. Writes always require native VGR.',ignoreFocusOut:true,validateInput:v=>v.trim()?undefined:'Task goal required'});
    if(!goal)return;if(typeof goal!=='string')throw Error('Task goal must be text');
    if(request&&Object.keys(request).some(k=>!['goal','requiredFiles','maxActions'].includes(k)))throw Error('Unsupported task option');
    let finalGoal=goal;
    if(withSelection){const ed=vscode.window.activeTextEditor;
      if(!ed||ed.selection.isEmpty)throw Error('Select code in the editor before using Submit With Selected Code');
      const content=ed.document.getText(ed.selection);
      if(Buffer.byteLength(content,'utf8')>8192)throw Error('Selected code exceeds the 8 KiB explicit-sharing limit');
      const folder=binding.folder,relative=path.relative(folder.uri.fsPath,ed.document.uri.fsPath);
      if(relative.startsWith('..')||path.isAbsolute(relative))throw Error('Selected code must be inside the bound workspace');
      finalGoal+=`\n\n[USER-EXPLICIT CODE CONTEXT, NOT EXECUTION INSTRUCTIONS]\nFile: ${relative}\nSelection:\n${content}\n[/USER-EXPLICIT CODE CONTEXT]`;
    }
    if(Buffer.byteLength(finalGoal,'utf8')>16384)throw Error('Task exceeds the CLI policy; shorten the request');
    const rpc=await this._open(binding.folder);
    const task={goal:finalGoal,...(request?.requiredFiles?{requiredFiles:request.requiredFiles}:{}),maxActions:request?.maxActions??1};
    if(binding.autoContinue){
      if(!rpc.recoveryCapable)throw Error('Native VGR recovery unavailable; task not dispatched');
      await this._saveRecoveryTask(binding.folder,binding.id,task);
    }
    const created=await rpc.request('task.submit',{sessionId:binding.id,task},15000);
    if(created?.status!=='ACCEPTED')throw Error(`SDK rejected task: ${created?.status||'unknown'}`);
    this.activeRuns.set(binding.id,true);
    this.output.appendLine(`[${binding.folder.name} ${binding.id.slice(0,8)}] ACCEPTED (not verified or delivered)`);
    this.refresh();
    // Terminal stays an ordinary terminal; task execution belongs exclusively to SDK+VGR.
    void rpc.request('task.wait',{sessionId:binding.id},3000*1000).then(result=>{
      this.activeRuns.delete(binding.id);const status=classifyStatus(result);
      this.output.appendLine(`[${binding.id.slice(0,8)}] ${status.label}`);
      if(result?.status==='DRAFT_UNVERIFIED'&&result?.verified===false&&typeof result?.draft==='string'){
        this.output.appendLine('[UNVERIFIED DRAFT — NOT NATIVE VGR COMMIT]');this.output.appendLine(result.draft.slice(0,120000));
      }
      vscode.window.showInformationMessage(`Voder ${binding.id.slice(0,8)}: ${status.label}`);this.refresh();
    }).catch(e=>{this.activeRuns.delete(binding.id);this.output.appendLine(`[${binding.id.slice(0,8)}] WAIT/CONNECTION ERROR ${textErr(e)}; reconciliation required`);this.refresh();});
    vscode.window.showInformationMessage(`Voder task accepted in session ${binding.id.slice(0,8)} (NOT YET VERIFIED)`);
    return {sessionId:binding.id,status:'ACCEPTED'};
  }
  async repairFromValidation(arg){
    if(!vscode.workspace.isTrusted)throw Error('Workspace Trust required');
    const folder=await this._folder();if(!folder)return;
    let file=arg?.evidencePath;
    if(!file){const selected=await vscode.window.showOpenDialog({canSelectMany:false,filters:{'Validation failure':['json']},title:'Select failed validation evidence; this creates a new linked task'});file=selected?.[0]?.fsPath;}
    if(!file)return;
    const cli=await this._validateAgent(this._config(folder).get('agentRoot',''));
    const {loadValidationRepairPlan}=await import(require('node:url').pathToFileURL(path.resolve(path.dirname(cli),'../../voder-sdk/src/validation-repair.mjs')).href);
    const plan=await loadValidationRepairPlan(file,{cwd:folder.uri.fsPath});
    const key='voder.validationRepair.'+crypto.createHash('sha256').update(folder.uri.fsPath+'\0'+plan.parent_session_id+'\0'+plan.receipt_sha256).digest('hex');
    if(!this.context.globalState)throw Error('Durable repair reservation unavailable');
    if(this.repairReservations.has(key)||this.context.globalState.get(key))throw Object.assign(Error('Evidence already reserved; inspect prior repair before any retry'),{code:'VALIDATION_REPAIR_ALREADY_RESERVED'});
    const choice=arg?.userConfirmed===true?'Create repair task':await vscode.window.showWarningMessage('Create one linked repair task from this validation failure? Model and sandbox usage apply. Native VGR policy still controls writes.','Create repair task','Cancel');
    if(choice!=='Create repair task')return;
    // Reserve before dispatch. A crash or uncertain acknowledgement never silently replays a paid task.
    if(this.repairReservations.has(key)||this.context.globalState.get(key))throw Object.assign(Error('Evidence already reserved'),{code:'VALIDATION_REPAIR_ALREADY_RESERVED'});
    this.repairReservations.add(key);
    try{
      await this.context.globalState.update(key,{status:'RESERVED',parent_session_id:plan.parent_session_id,evidence_sha256:plan.evidence_sha256});
      const binding=await this.openTerminal(folder);if(!binding)throw Error('Repair session binding missing');
      const result=await this.submitTask(false,plan.task,binding);
      const linked={...result,parent_session_id:plan.parent_session_id,evidence_sha256:plan.evidence_sha256,receipt_sha256:plan.receipt_sha256,write_targets:plan.write_targets};
      await this.context.globalState.update(key,{status:'ACCEPTED',...linked});
      this.output.appendLine('[VALIDATION REPAIR] '+JSON.stringify(linked));return linked;
    }catch(e){await this.context.globalState.update(key,{status:'REQUIRES_REVIEW',parent_session_id:plan.parent_session_id,code:String(e.code||'ERROR')});throw e;}finally{this.repairReservations.delete(key);}
  }
  async _pickSession(arg){
    if(arg?.folder&&arg?.id)return arg;
    const active=vscode.window.activeTerminal;const b=active&&this.terminalMap.get(active);
    if(b)return {folder:b.folder,id:b.id};
    const all=[];
    for(const f of vscode.workspace.workspaceFolders||[]){
      try{const list=await (await this._reviewClient({folder:f,id:null})).request('session.list',{},10000);
        for(const rec of list||[])if(rec.status!=='JOURNAL_UNREADABLE')all.push({label:`${(rec.session_id||'').slice(0,8)} ${rec.status}`,description:f.name,folder:f,id:rec.session_id});
      }catch(e){this.output.appendLine(`[${f.name}] ${textErr(e)}`);}
    }
    const choice=await vscode.window.showQuickPick(all,{placeHolder:'Select exact Voder session (historical sessions are read-only)'});
    return choice?{folder:choice.folder,id:choice.id}:null;
  }
  async showStatus(arg){const s=await this._pickSession(arg);if(!s)return;
    const rpc=await this._reviewClient(s);
    // A visible terminal state may precede durable journal finalization.
    // task.wait is the control plane's completion barrier; never infer it from status.
    if(arg?.waitForSettlement===true)await rpc.request('task.wait',{sessionId:s.id},3000*1000);
    const payload=await rpc.request('session.status',{sessionId:s.id});
    const status=classifyStatus(payload);
    this.output.appendLine(`[SESSION ${s.id}] ${JSON.stringify({status:status.status,verified:status.verified,result:payload.result??null})}`);
    if(arg?.quiet!==true)this.output.show(true);return {...payload,editor_event_count:this.eventCounts.get(s.id)||0,editor_events:[...(this.eventMetadata.get(s.id)||[])]};
  }
  async showEvents(){const s=await this._pickSession();if(!s)return;
    const events=await (await this._open(s.folder)).request('session.events',{sessionId:s.id,after:0});
    // Events are the SDK's allowlisted metadata projection; not a native proof token.
    for(const e of events||[])this.output.appendLine(`[${s.id.slice(0,8)}] ${e.sequence} ${e.type} ${e.state}`);
    this.output.show(true);
  }
  async showSessions(){this.refresh();await vscode.commands.executeCommand('workbench.view.explorer');}
  async cancelTask(arg){const s=await this._pickSession(arg);if(!s)return;
    const confirmed=arg?.userConfirmed===true?'Cancel task':await vscode.window.showWarningMessage('Request Voder cancellation? In-flight VGR outcomes still require native reconciliation.','Cancel task','Keep running');
    if(confirmed!=='Cancel task')return;
    const ack=await (await this._reviewClient(s)).request('session.cancel',{sessionId:s.id});
    this.output.appendLine(`[${s.id}] ${ack.status}; not proof of rollback`);this.refresh();return ack;
  }
  async _reviewClient(s){
    const general=await this.generalTasks?.reviewClient?.(s);if(general)return general;
    const binding=this.chatReviewProfiles.get(s.id)||this.context.globalState?.get('voder.chat.reviewBindings',{})?.[s.id];
    if(binding&&binding.workspace!==s.folder.uri.fsPath)throw Error('Review workspace does not match the recorded session');
    // Legacy sessions may use the already approved workspace profile for read-only review.
    // The backend still verifies the exact journal, policy, action and Rust receipt.
    const profile=binding?.profile||this.context.globalState?.get(this._chatProfileKey(s.folder));
    return this._open(s.folder,profile?{mode:'native',gateway:profile}:{});
  }
  async _pickDiffSession(arg){
    if(arg?.folder&&arg?.id)return arg;
    const folder=arg?.folder||await this._folder();if(!folder)return null;
    const rpc=await this._reviewClient({folder,id:null});
    const rows=await rpc.request('session.list',{},10000);
    const delivered=(rows||[]).filter(r=>r.status==='DELIVERED'&&/^[0-9a-f-]{36}$/i.test(r.session_id||''));
    delivered.sort((a,b)=>String(b.started_at||'').localeCompare(String(a.started_at||''))||a.session_id.localeCompare(b.session_id));
    if(!delivered.length){vscode.window.showWarningMessage('This workspace has no delivered session to review. No task was rerun.');return null;}
    return {folder,id:delivered[0].session_id};
  }
  async inspectDiff(arg){if(!vscode.workspace.isTrusted)throw Error('Workspace Trust required for verified diff review');const s=await this._pickDiffSession(arg);if(!s)return;
    const rpc=await this._reviewClient(s);
    if(!rpc.diffCapable){const reason='This control connection does not support verified diff review. No code was changed by this request.';vscode.window.showWarningMessage(reason);return {status:'NOT_AVAILABLE',sessionId:s.id,reason};}
    const result=await rpc.request('session.diff',{sessionId:s.id},30000);
    if(result.status==='NOT_AVAILABLE'){const reason=String(result.reason||'No verified workspace delivery is available');this.output.appendLine('[DIFF '+s.id+'] '+reason);vscode.window.showWarningMessage('Session '+s.id.slice(0,8)+': '+reason);return result;}
    const files=validateVerifiedDiff(result,s.id),nonce=crypto.randomBytes(12).toString('hex'),documents=[];
    // TextDocumentContentProvider creates immutable review documents; no workspace write API.
    for(const file of files){
      const left=vscode.Uri.parse(`voder-evidence:/${s.id}/${nonce}/before/${file.relative_path}`),right=vscode.Uri.parse(`voder-evidence:/${s.id}/${nonce}/after/${file.relative_path}`);
      this.diffDocuments.set(left.toString(),file.before);this.diffDocuments.set(right.toString(),file.after);
      while(this.diffDocuments.size>64)this.diffDocuments.delete(this.diffDocuments.keys().next().value);
      await vscode.commands.executeCommand('vscode.diff',left,right,`Voder ${s.id.slice(0,8)} · ${file.relative_path}`,{preview:false});
      documents.push({relative_path:file.relative_path,before_uri:left.toString(),after_uri:right.toString(),before_sha256:file.before_sha256,after_sha256:file.after_sha256});
    }
    this.output.appendLine(`[${s.id}] Verified native diff: ${documents.length} file(s), commit ${result.commit_digest}; read-only snapshots.`);
    return {...result,editor_documents:documents};
  }
  async showDiagnostics(){const folder=await this._folder();if(!folder)return;
    const root=folder.uri.fsPath+path.sep;let count=0;const diagnostics=[];
    for(const [uri,list] of vscode.languages.getDiagnostics()){
      if(!uri.fsPath.startsWith(root))continue;
      for(const d of list.slice(0,100)){if(count++>=200)break;diagnostics.push({relative_path:path.relative(folder.uri.fsPath,uri.fsPath),line:d.range.start.line+1,severity:d.severity,message:d.message.slice(0,250)});this.output.appendLine(`[EDITOR DIAGNOSTIC / NOT VGR VERIFIED] ${path.relative(folder.uri.fsPath,uri.fsPath)}:${d.range.start.line+1} ${d.message.slice(0,250)}`);}
      if(count>=200)break;
    }
    this.output.appendLine(`Editor diagnostics reported: ${count} (not VGR evidence)`);this.output.show(true);return {status:'EDITOR_HINTS_UNVERIFIED',diagnostics};
  }
  async exportSession(){const s=await this._pickSession();if(!s)return;
    const to=await vscode.window.showSaveDialog({saveLabel:'Export UNVERIFIED session metadata',defaultUri:vscode.Uri.file(path.join(s.folder.uri.fsPath,`voder-session-${s.id}.json`)),filters:{JSON:['json']}});
    if(!to)return;
    const result=await (await this._open(s.folder)).request('session.export',{sessionId:s.id,outputPath:to.fsPath});
    this.output.appendLine(`[${s.id}] exported ${result.status} ${result.sha256}`);this.output.show(true);
  }
  _recoveryIndexKey(folder){return 'voder.vgr.recovery.index.'+crypto.createHash('sha256').update(folder.uri.fsPath).digest('hex');}
  _recoveryTaskKey(folder,id){return this._recoveryIndexKey(folder)+'.'+id;}
  async _withSecretsLock(fn){
    const p=this._secretQueue.catch(()=>{}).then(fn);this._secretQueue=p;return p;
  }
  async _saveRecoveryTask(folder,id,task){return this._withSecretsLock(async()=>{
    if(!this.context.secrets?.store||!this.context.secrets?.get)throw Error('VS Code SecretStorage unavailable; recovery-enabled task not submitted');
    const key=this._recoveryIndexKey(folder);
    // Persist the exact trusted task before allowing the CLI to fsync its native intent.
    // Preserve exact goal only in the editor's opt-in SecretStorage; never journal it.
    const previous=await this.context.secrets.get(key);
    let ids=[];try{const p=JSON.parse(previous||'[]');if(Array.isArray(p))ids=p.filter(x=>/^[0-9a-f-]{36}$/i.test(x));}catch{}
    if(!ids.includes(id))ids.push(id);
    if(ids.length>64)throw Error('Recovery task index limit exceeded; task not dispatched');
    await this.context.secrets.store(this._recoveryTaskKey(folder,id),JSON.stringify(task));
    try{await this.context.secrets.store(key,JSON.stringify(ids));}
    catch(e){await this.context.secrets.delete(this._recoveryTaskKey(folder,id)).catch(()=>{});throw e;}
  });}
  async _autoRecoverTrusted(folder,rpc){return this._withSecretsLock(async()=>{
    if(!rpc.recoveryCapable||this._config(folder).get('autoRecovery',false)!==true)return;
    const value=await this.context.secrets?.get(this._recoveryIndexKey(folder));
    let ids=[];try{const a=JSON.parse(value||'[]');if(Array.isArray(a)&&a.length<=64)ids=a;}catch{throw Error('Recovery secret index corrupted');}
    if(ids.some(x=>typeof x!=='string'||!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(x)))throw Error('Recovery secret index has invalid session');
    const retained=[];
    for(const id of ids){
      const saved=await this.context.secrets.get(this._recoveryTaskKey(folder,id));
      if(!saved){this.output.appendLine(`[${id.slice(0,8)}] RECOVERY_BLOCKED missing trusted task secret`);continue;}
      try{
        const status=await rpc.request('session.status',{sessionId:id},10000);
        if(status?.session_id!==id)throw Error('Session identity mismatch');
        if(status.status==='CREATED'){retained.push(id);continue;}
        if(status.status!=='INSPECTING'){
          if(['VERIFIED','DELIVERED','CANCELLED','FAILED','BLOCKED','DRAFT_UNVERIFIED','WAITING_FOR_INPUT','RECOVERY_REVIEW_REQUIRED'].includes(status.status)){
            // Only settled, recognized terminal states permit clearing the recovered-task secret.
            await this.context.secrets.delete(this._recoveryTaskKey(folder,id));
          }else retained.push(id);
          continue;
        }
        const result=await rpc.request('session.recover',{sessionId:id,task:JSON.parse(saved)},120000);
        this.output.appendLine(`[${id.slice(0,8)}] ${result.status}: ${String(result.reason||'native reconciliation passed; re-planning only').slice(0,160)}`);
        // The recovered run is still unverified until independent Rust VGR receipt settles.
        retained.push(id);
        if(result?.status==='AUTO_REPLANNING'){
          void rpc.request('task.wait',{sessionId:id},3000*1000).then(out=>{
            this.output.appendLine(`[RECOVERY ${id.slice(0,8)}] SETTLED ${String(out?.status||'UNKNOWN').slice(0,64)}; VGR final receipt remains authoritative`);this.refresh();
          }).catch(e=>this.output.appendLine(`[RECOVERY ${id.slice(0,8)}] WAIT BLOCKED ${textErr(e)}`));
        }
      }catch(e){retained.push(id);this.output.appendLine(`[${id.slice(0,8)}] RECOVERY_BLOCKED ${textErr(e)}`);}
    }
    await this.context.secrets.store(this._recoveryIndexKey(folder),JSON.stringify(retained));
    this.refresh();
  });}
  async inspectRecovery(arg){
    if(!vscode.workspace.isTrusted)throw Error('Workspace Trust required to inspect history');
    const s=arg?.folder&&arg?.id?arg:await this._selectHistoricalSession();
    if(!s)return;
    const rpc=await this._open(s.folder);
    const result=await rpc.request('session.resume',{sessionId:s.id},15000);
    if(!result||result.sessionId!==s.id||result.mode!=='READ_ONLY_REATTACH'||result.canContinue!==false)
      throw Error('Unsupported recovery contract: automatic execution blocked');
    const status=await rpc.request('session.status',{sessionId:s.id},15000);
    if(!status||status.session_id!==s.id)throw Error('History/status identity mismatch: blocked');
    this.output.appendLine(`[RECOVERY ${s.id}] READ_ONLY_REATTACH; status=${String(status.status).slice(0,64)}; canContinue=false; no tool replay`);
    this.output.appendLine(`[RECOVERY ${s.id}] ${String(result.reason||'Native VGR reconciliation required').slice(0,220)}`);
    this.output.show(true);this.refresh();
    vscode.window.showInformationMessage(`Voder ${s.id.slice(0,8)}: history attached read-only; no task replay.`);return result;
  }
  async _selectHistoricalSession(){
    const all=[];
    for(const folder of vscode.workspace.workspaceFolders||[]){
      const rpc=await this._open(folder),rows=await rpc.request('session.list',{},10000);
      for(const row of rows||[]){
        if(!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(row.session_id||''))continue;
        all.push({label:`${row.session_id.slice(0,8)} · ${String(row.status||'UNKNOWN').slice(0,64)}`,description:folder.name,folder,id:row.session_id});
      }
    }
    const choice=await vscode.window.showQuickPick(all,{placeHolder:'Inspect existing Voder session (read-only, never replays commands)'});
    return choice?{folder:choice.folder,id:choice.id}:null;
  }
  async retryGuardedRecovery(){
    if(!vscode.workspace.isTrusted)throw Error('Workspace Trust required');
    const s=await this._selectHistoricalSession();if(!s)return;
    if(this._config(s.folder).get('mode','native')!=='native')throw Error('Draft mode cannot authorize native recovery');
    const rpc=await this._open(s.folder);
    if(!rpc.recoveryCapable)throw Error('Rust VGR recovery adapter is not configured');
    const saved=await this.context.secrets?.get(this._recoveryTaskKey(s.folder,s.id));
    if(!saved)throw Error('Trusted original task secret missing; recovery blocked, no manual goal substitution');
    const result=await rpc.request('session.recover',{sessionId:s.id,task:JSON.parse(saved)},120000);
    this.output.appendLine(`[RECOVERY ${s.id}] ${result.status} ${String(result.reason||'').slice(0,160)}`);
    this.output.show(true);this.refresh();
    vscode.window.showInformationMessage(`Voder ${s.id.slice(0,8)}: ${result.status} (not proof of native commit)`);
  }
  async reconnect(){for(const client of this.clients.values())client.close('USER_RECONNECT');this.clients.clear();
    const f=await this._folder();if(f){await this._open(f);vscode.window.showInformationMessage('Reconnected; prior executions NOT automatically replayed.');}
  }
  getTreeItem(element){
    if(element instanceof vscode.TreeItem)return element;
    throw new TypeError('Invalid Voder session tree item');
  }
  async getChildren(){const all=[];
    for(const folder of vscode.workspace.workspaceFolders||[]){
      if(!vscode.workspace.isTrusted)continue;
      try{const rpc=await this._open(folder),rows=await rpc.request('session.list',{},10000);
        for(const row of rows||[])all.push(new SessionNode(folder,row));
      }catch(e){const item=new vscode.TreeItem(`${folder.name}: DISCONNECTED / CONFIG REQUIRED`);item.tooltip=textErr(e);all.push(item);}
    }
    return all;
  }
  async _runChatParticipant(request,context,stream,token){
    if(!this.trusted())throw Error('Workspace Trust is required');
    const folder=await this._folder();if(!folder)return {metadata:{status:'WORKSPACE_NOT_SELECTED'}};
    const goal=request?.prompt;
    if(typeof goal!=='string'||!goal.trim())throw Error('Enter a coding task for @voder');
    let run,stopListener;
    try{
      stream.progress('Creating a new isolated Voder task session');
      const history=(context?.history||[]).filter(h=>typeof h?.prompt==='string').slice(-3).map(h=>({goal:h.prompt,status:'PRIOR_TURN'}));
      const referenceContext=request?.references?.length||/(?:^|[\s(])@/.test(goal)?await this.chatView.references.requestContext(folder,goal,request?.references||[]):[];
      if(this.generalTasks?.participant)return this.generalTasks.participant(folder,goal,stream,token,referenceContext);
      // Built-in @voder Chat must support read-only reference analysis without
      // prompting for a native file-write profile. Never mint execution authority.
      if(!writeRequest(goal)){
        run=await this.chatRuns.inspect(folder,goal,event=>{
          if(event.kind==='accepted')this._trackChatSession(event.sessionId,folder);
          if(event.kind==='event'&&event.event?.type==='model.message_end')stream.progress('Voder read-only PDF analysis completed');
        },{history,referenceContext});
        this._trackChatSession(run.sessionId,folder);
        if(token?.onCancellationRequested)stopListener=token.onCancellationRequested(()=>{
          void this.chatRuns.cancel(folder,run.sessionId).catch(()=>{});
        });
        if(token?.isCancellationRequested)void this.chatRuns.cancel(folder,run.sessionId).catch(()=>{});
        const settled=await run.completion;
        const report=inspectReport(folder.uri.fsPath,settled);
        // The SDK only returns a non-committable draft for read-only tasks.
        stream.markdown(report.answer.slice(0,60000).replace(/[<>]/g,c=>c==='<'?'&lt;':'&gt;'));
        return {metadata:{status:report.status,verified:false,sessionId:settled.sessionId}};
      }
      const nativeProfile=await this.ensureNativeChat(folder,{goal});
      run=await this.chatRuns.start(folder,goal,event=>{
        if(event.kind==='accepted')this._trackChatSession(event.sessionId,folder,nativeProfile.path);
        if(event.kind==='event'&&['sandbox.candidate_settled','vgr.action_settled','task.settled'].includes(event.event?.type))
          stream.progress('Voder: '+event.event.type+' · '+String(event.event.status||'pending').slice(0,48));
      },{history,...(referenceContext.length?{referenceContext}:{}),nativeProfile:nativeProfile.path,requiredFiles:[nativeProfile.target],operatorTarget:nativeProfile.target});
      this._trackChatSession(run.sessionId,folder,nativeProfile.path);
      if(token?.onCancellationRequested)stopListener=token.onCancellationRequested(()=>{
        // Only explicit Chat Stop requests cancellation, never view hide or disconnect.
        void this.chatRuns.cancel(folder,run.sessionId).catch(()=>{});
      });
      if(token?.isCancellationRequested)void this.chatRuns.cancel(folder,run.sessionId).catch(()=>{});
      const settled=await run.completion;
      stream.markdown(`**Voder — ${settled.label}**\n\nSession: \`${settled.sessionId}\`\n\n`+
        (settled.verified?'Native VGR verification confirmed. Use **Voder: Inspect Verified Session Diff** to review changes.':'No verified native delivery is established by this outcome.'));
      if(settled.draft)stream.markdown('\n\n**UNVERIFIED model draft (not executed or committed)**\n\n'+settled.draft.slice(0,8000).replace(/[<>]/g,c=>c==='<'?'&lt;':'&gt;'));
      return {metadata:{status:settled.status,verified:settled.verified,sessionId:settled.sessionId}};
    }finally{stopListener?.dispose();}
  }
  _trackChatSession(id,folder,profile){
    if(typeof id!=='string')return;
    this.chatSessionFolder.set(id,folder);
    if(typeof profile==='string'&&path.isAbsolute(profile)){
      const binding={workspace:folder.uri.fsPath,profile};this.chatReviewProfiles.set(id,binding);
      while(this.chatReviewProfiles.size>128)this.chatReviewProfiles.delete(this.chatReviewProfiles.keys().next().value);
      const saved={...(this.context.globalState?.get('voder.chat.reviewBindings',{})||{}),[id]:binding};
      while(Object.keys(saved).length>128)delete saved[Object.keys(saved)[0]];
      Promise.resolve(this.context.globalState?.update('voder.chat.reviewBindings',saved)).catch(e=>this.output.appendLine('Review binding could not be saved: '+textErr(e)));
    }
    while(this.chatSessionFolder.size>128)this.chatSessionFolder.delete(this.chatSessionFolder.keys().next().value);
  }
  async addReference(type,uri){
    if(!this.trusted())throw Error('Workspace Trust is required');
    await this.openChat();
    const view=[...this.chatView.views][0];
    if(!view)throw Error('Open Voder Chat before adding a reference');
    return this.chatView.referenceCommand(view,uri?'reference_command_file':type,{uri});
  }
  async openChat(){await vscode.commands.executeCommand('workbench.view.extension.voder');await vscode.commands.executeCommand('voder.chat.focus');}
  activate(){
    if(!this.context.globalState?.get('voder.release.firstRunPrompted',false)){
      void this.context.globalState?.update('voder.release.firstRunPrompted',true).then(()=>
        vscode.window.showInformationMessage('Voder: Configure your own models, Modal and search APIs before use.','Configure now').then(choice=>{
          if(choice==='Configure now')void vscode.commands.executeCommand('voder.configureFirstRun');
        })
      ).catch(()=>{});
    }

    const commands={configureModel:()=>require('./release-model-config.cjs').configureModel(vscode),previewCurrentFile:()=>this.generalTasks.previewCurrentFile(),extractArchive:()=>require('./archive-input.cjs').extractArchiveCommand(vscode),fileInputDiagnostics:async()=>{const f=await this._folder();if(!f)return;return require('./vision-setup.cjs').inputDiagnostics(vscode,this.context.extensionPath,await this._agentRootForChat(f));},configureVisionModel:()=>require('./vision-setup.cjs').configureVision(vscode),configureFirstRun:()=>require('./first-run-setup.cjs').run(vscode),addFileContext:uri=>this.addReference('reference_editor',uri),addSelectionContext:()=>this.addReference('reference_selection'),runIsolatedPythonChecks:()=>this.toolkit.sandboxPython(),inspectProjectTools:()=>this.toolkit.inspectCurrent(),restoreProjectSession:()=>this.toolkit.restore(),inspectGeneratedProject:async()=>{const s=await this.toolkit.selectSession();if(s)return this.toolkit.inspect(s.id);},inspectProjectRuntime:async()=>{const s=await this.toolkit.selectSession();if(s)return this.toolkit.runtime(s.id);},configureNativeChat:()=>this.configureNativeChat(),openChat:()=>this.openChat(),openTerminal:()=>this.openTerminal(),bindTerminal:()=>this.bindTerminal(),submitTask:arg=>this.submitTask(false,arg),submitWithSelection:()=>this.submitTask(true),
      repairFromValidation:arg=>this.repairFromValidation(arg),showStatus:arg=>this.showStatus(arg),showEvents:()=>this.showEvents(),showSessions:()=>this.showSessions(),cancelTask:arg=>this.cancelTask(arg),
      inspectDiff:arg=>this.inspectDiff(arg),showDiagnostics:()=>this.showDiagnostics(),exportSession:()=>this.exportSession(),inspectRecovery:arg=>this.inspectRecovery(arg),retryGuardedRecovery:()=>this.retryGuardedRecovery(),reconnect:()=>this.reconnect()};
    for(const [k,cb] of Object.entries(commands))this.context.subscriptions.push(vscode.commands.registerCommand('voder.'+k,async(...args)=>{
      try{return await cb(...args);}catch(e){this.output.appendLine(`[ERROR] ${textErr(e)}`);vscode.window.showErrorMessage('Voder: '+textErr(e));return {status:'EDITOR_COMMAND_REJECTED',code:String(e.code||'VODER_ERROR')};}
    }));
    this.context.subscriptions.push(vscode.workspace.registerTextDocumentContentProvider('voder-evidence',{provideTextDocumentContent:uri=>{const value=this.diffDocuments.get(uri.toString());if(value===undefined)throw Error('Expired verified diff snapshot');return value;}}));
    this.context.subscriptions.push(vscode.window.registerWebviewViewProvider('voder.chat',this.chatView,{webviewOptions:{retainContextWhenHidden:true}}));
    // When VS Code exposes built-in Chat Participant API, @voder also works in the standard right-hand Chat view.
    // The standalone Voder view works even when Copilot Chat or the Chat API is unavailable.
    if(typeof vscode.chat?.createChatParticipant==='function'){
      const participant=vscode.chat.createChatParticipant('voder.agent',async(request,context,stream,token)=>{
        try{return await this._runChatParticipant(request,context,stream,token);}catch(e){
          stream.markdown('Voder could not confirm a delivery. '+String(e?.code||'CHAT_ERROR').replace(/[^A-Z0-9_]/g,'').slice(0,48)+'. Inspect SDK evidence before retrying.');
          this.output.appendLine('[CHAT] '+textErr(e));return {metadata:{status:'NOT_CONFIRMED'}};
        }
      });
      this.context.subscriptions.push(participant);
    }
    this.context.subscriptions.push(vscode.window.registerTreeDataProvider('voder.sessions',this),
      vscode.window.onDidOpenTerminal(t=>{void this.onTerminalOpen(t);}),
      vscode.window.onDidCloseTerminal(t=>this.onTerminalClose(t)),
      vscode.window.onDidChangeActiveTerminal(()=>this._statusForTerminal()),
      vscode.workspace.onDidChangeConfiguration(e=>{if(e.affectsConfiguration('voder')){for(const c of this.clients.values())c.close('CONFIG_CHANGED');this.clients.clear();this.refresh();}}));
    for(const t of vscode.window.terminals)void this.onTerminalOpen(t);
  }
  deactivate(){for(const view of this.chatView.views)this.chatView.references.dispose(view);this.toolkit.dispose();for(const c of this.clients.values())c.close('EXTENSION_DEACTIVATED');this.clients.clear();this.diffDocuments.clear();this.eventCounts.clear();this.eventMetadata.clear();this.chatSessionFolder.clear();this.chatReviewProfiles.clear();}
}
function activate(context){instance=new VoderExtension(context);instance.activate();}
function deactivate(){instance?.deactivate();instance=undefined;}
module.exports={activate,deactivate,VoderExtension};
