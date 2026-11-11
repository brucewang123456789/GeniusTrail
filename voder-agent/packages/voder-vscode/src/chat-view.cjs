'use strict';
/** VS Code sandboxed chat Webview, with all workspace operations in ChatRunService. */
const crypto=require('node:crypto');
// Bundled local parser; no CDN, network request, or workspace execution.
let mathScript='';
try{mathScript=require('node:fs').readFileSync(require('node:path').join(__dirname,'vendor/katex.min.js'),'utf8').replace(/<\/script/gi,'<\\/script');}catch{/* Preserve chat if a broken deployment omits the optional typesetter. */}
const {ReferenceController}=require('./reference-ui.cjs');
const referenceScript=require('./reference-webview.cjs');
const {validateGoal}=require('./chat-core.cjs');
const {startNotice}=require('./native-start.cjs');
const {scopeFeedback}=require('./scope-feedback.cjs');
const {isProjectGoal}=require('./project-workspace.cjs');
const {parseMultiGoal}=require('./multi-workspace.cjs');
const {runEntry,planEntry}=require('./entry-orchestrator.cjs');
const {createAssistantPresentation}=require('./output-presentation.cjs');
const {formatTaskDuration,createTaskClock}=require('./task-duration.cjs');
const {createProgressiveOutput}=require('./progressive-output.cjs');
function chatHtml(webview){
 // Raw host diagnostics require explicit operator opt-in; ordinary UI stays business-facing.
 const showDeveloperDiagnostics=process.env.VODER_DEVELOPER_DIAGNOSTICS==='YES';
 const nonce=crypto.randomBytes(18).toString('base64');
 const logoUri="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMjggMTI4Ij48ZyBmaWxsPSJub25lIiBzdHJva2U9IiNlZGVkZWQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTEwNiA4M0E0NiA0NiAwIDEgMSA5OSAzNCIgc3Ryb2tlLXdpZHRoPSI1Ii8+PHBhdGggZD0iTTg4IDUwQTI4IDI4IDAgMSAwIDQ1IDg1IiBzdHJva2Utd2lkdGg9IjMuNSIvPjxwYXRoIGQ9Im00OSA2MCAxMyAxNiAyMi0yOSIgc3Ryb2tlLXdpZHRoPSI1Ii8+PC9nPjxwYXRoIGZpbGw9IiNlZGVkZWQiIGQ9Im05MiAzOSAxNiAzLTUtMTZ6TTM5IDc2bDMgMTUgMTMtOXoiLz48L3N2Zz4=";
 const welcome=`<div class="empty" id="empty"><img class="welcome-mark" src="${logoUri}" width="128" height="128" alt="Voder verification loop emblem"><div class="welcome-name">VODER</div><h1>Intelligence proposes.<br>Evidence decides.</h1><p>Preserve what is proven.<br>Commit only verified progress.</p></div>`;
 const csp=`default-src 'none'; img-src ${webview.cspSource} data:; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}'; font-src ${webview.cspSource};`;
 return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><meta http-equiv="Content-Security-Policy" content="${csp}"><title>Voder Chat</title><style nonce="${nonce}">
*{box-sizing:border-box}
:root{color-scheme:dark}
body{--bg:#080808;--panel:#111111;--line:#262626;--muted:#969696;--accent:#ededed;margin:0;background:var(--bg);color:#e8e8e8;font:14px/1.65 "Times New Roman",Times,Georgia,serif;height:100vh;height:100dvh;display:flex;flex-direction:column;overflow:hidden;-webkit-font-smoothing:antialiased}
button,textarea{font:inherit}button{cursor:pointer;transition:background .16s,border-color .16s,color .16s}button:disabled{opacity:.35;cursor:not-allowed}button:focus-visible,textarea:focus-visible,summary:focus-visible{outline:2px solid #c7c7c7;outline-offset:3px}
.head{flex:none;display:flex;justify-content:space-between;align-items:center;gap:12px;padding:21px 22px 18px;border-bottom:1px solid #1b1b1b;background:#080808}.head strong{font-size:18px;font-weight:400;letter-spacing:.22em;color:#fafafa}.project{margin-top:8px;max-width:min(60vw,620px);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;color:#aaa}.sub{font-size:8px;letter-spacing:.17em;color:#777;margin-top:7px}.head button{border:1px solid #2a2a2a;background:transparent;border-radius:20px;color:#aaa;padding:7px 12px;font-size:11px}.head button:hover{color:white;border-color:#666;background:#171717}
#messages{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;padding:28px 22px 36px;display:flex;flex-direction:column;gap:28px;scrollbar-width:thin;scrollbar-color:#363636 transparent}
.msg{width:100%;max-width:760px;margin:0 auto;overflow-wrap:anywhere;line-height:1.75;font-size:15px}.user{width:auto;max-width:88%;margin:0 0 0 auto;align-self:flex-end;padding:12px 16px;background:#1b1b1b;border:1px solid #2b2b2b;border-radius:18px 18px 5px 18px;white-space:pre-wrap}.agent{align-self:stretch;padding:0 0 22px;border-bottom:1px solid #202020;background:transparent}.tag{font-size:9px;font-weight:600;letter-spacing:.14em;color:#929292;margin-bottom:11px}.agent>.tag{color:#ddd}.user>.tag{font-size:8px;color:#888;margin-bottom:5px}
.empty{margin:auto;text-align:center;max-width:480px;width:100%;padding:32px 8px 40px;line-height:1.8;color:#8d8d8d}.orbit{position:relative;width:51px;height:51px;border:1px solid #555;border-radius:50%;margin-bottom:32px;transform:rotate(-28deg);box-shadow:inset 0 0 17px #ffffff08}.orbit:before{content:'';position:absolute;left:-10px;right:-10px;top:24px;border-top:1px solid #d5d5d5}.orbit:after{content:'';position:absolute;width:5px;height:5px;background:#eee;border-radius:50%;right:1px;top:7px}.eyebrow{font-size:8px;letter-spacing:.22em;color:#858585}.empty h1{margin:22px 0 18px;font-size:clamp(26px,4vw,38px);line-height:1.18;letter-spacing:-.025em;font-weight:400;color:#f4f4f4}.welcome-mark{display:block;margin:0 auto 28px;width:88px;height:88px;opacity:.95}.welcome-name{font-size:11px;letter-spacing:.32em;color:#bcbcbc;padding-left:.35em}.empty p{margin:0 auto;max-width:300px;font-size:13px;line-height:1.65;color:#a4a4a4}.empty .caption{display:block;margin-top:28px;font-size:9px;color:#777;letter-spacing:.13em}
.foot{flex:none;padding:13px 18px 14px;background:#080808;border-top:1px solid #1b1b1b}.composer{display:grid;grid-template-columns:1fr auto;gap:0 8px;border:1px solid #383838;background:#131313;border-radius:14px;padding:12px;box-shadow:0 8px 24px #0003}.composer:focus-within{border-color:#909090;box-shadow:0 0 0 1px #909090}#goal{grid-column:1/-1;width:100%;resize:vertical;min-height:65px;max-height:160px;background:transparent;border:0;outline:none;color:#eee;padding:2px 2px 10px;line-height:1.7;border-radius:5px}#goal:focus-visible{outline:none}#goal::placeholder{color:#767676}
#state{display:flex;gap:6px;align-items:center;min-width:0;overflow-wrap:anywhere;font-size:10px;line-height:1.5;color:#989898;margin-bottom:11px;padding:0 2px}#state:before{content:'';height:4px;width:4px;flex:none;background:#bbb;border-radius:50%}.actions{grid-column:1/-1;display:flex;justify-content:flex-end;align-items:center;gap:8px;margin-top:5px}.actions button{padding:8px 13px;border:1px solid #333;border-radius:20px;font-size:11px;min-height:34px}.actions #cancel{margin-right:auto;background:transparent;color:#a8a8a8}.actions #send{background:#ededed;color:#111;border-color:#ededed;font-weight:600}.actions #send:hover:not(:disabled){background:#fff;border-color:#fff}.actions #cancel:hover:not(:disabled){background:#252525;color:white}
.hint{margin:10px 2px 0;color:#737373;font-size:10px;line-height:1.7}.hint summary{cursor:pointer;font-size:9px;list-style:none}.hint summary:before{content:'+';display:inline-block;margin-right:7px;color:#aaa}.hint[open] summary:before{content:'−'}#warn{grid-column:1/-1;white-space:pre-wrap;color:#f0b5b5;font-size:11px;line-height:1.6}#warn:empty{display:none}
.small{font-size:11px;line-height:1.7;color:#969696;margin-top:9px;overflow-wrap:anywhere}.draft{white-space:pre-wrap;margin-top:15px;color:#d0d0d0;line-height:1.85}.trace{border-top:1px solid #222;margin-top:14px;padding-top:10px}.trace summary{cursor:pointer;color:#ddd;font-size:13px;padding:8px 0}.trace{border-color:#444}.trace .small{font:11px/1.7 "Times New Roman",Times,Georgia,serif;color:#888}.phase{display:flex;gap:4px;margin:12px 0}.phase span{height:2px;flex:1;border-radius:1px;background:#242424}.phase span.on{background:#bcbcbc}
.line{display:flex;flex-wrap:wrap;gap:7px;margin-top:14px}.line button{padding:8px 12px;border:1px solid #383838;border-radius:18px;background:#181818;color:#d6d6d6;font-size:11px}.line button:hover:not(:disabled){border-color:#777;background:#242424}.approval-panel{margin-top:18px;padding:16px;border:1px solid #505050;border-radius:14px;background:#161616;color:#d6d6d6;line-height:1.8;font-size:12px}.approval-status{color:#d4d4d4;border-top:1px solid #333;margin-top:12px;padding-top:10px;font-size:11px}.approval-panel .line button:first-child{background:#e7e7e7;border-color:#e7e7e7;color:#111}.review-feedback{font-size:11px;color:#aaa;white-space:pre-wrap;margin-top:12px}.review-feedback:empty{display:none}.review-feedback[data-status=ERROR]{color:#f0b5b5}
@media(min-width:700px){#messages{padding-left:36px;padding-right:36px}.foot{padding-left:max(22px,calc((100% - 760px)/2));padding-right:max(22px,calc((100% - 760px)/2))}.head{padding-left:28px;padding-right:28px}}
@media(max-width:360px){.head{padding:17px 15px}.sub{font-size:7px;letter-spacing:.1em}#messages{padding:22px 15px 28px;gap:24px}.foot{padding:11px}.empty h1{font-size:22px}.composer{padding:10px}.user{max-width:94%}}
@media(max-height:480px){.head{padding-top:11px;padding-bottom:10px}.foot{padding-top:8px;padding-bottom:8px}#goal{min-height:42px}.empty{padding:12px 4px}.welcome-mark{width:52px;height:52px;margin-bottom:12px}.empty h1{font-size:20px;margin-top:10px}.hint{margin-top:5px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}

/* Presentation only: monochrome surfaces, quiet hierarchy, no execution policy changes. */
body{--bg:#070707;--panel:#111;--line:#282828;--muted:#aaa;background:#070707;font-size:14px}
.head{padding:18px 24px;background:#070707}.head strong{font-size:17px;letter-spacing:.24em;font-weight:600}.sub{font-size:8px;color:#999}.project{color:#999;font-size:11px;max-width:65vw}
#messages{gap:24px;padding-top:24px}.msg{font-size:15px;max-width:720px}.user{max-width:92%;font-size:14px;padding:12px 16px;background:#161616;border-color:#292929}.agent{padding-bottom:24px;display:flex;flex-direction:column}.agent>.trace{order:90}.agent>.secondary{order:80}.agent>.review-feedback{order:70}.tag{font-size:10px;letter-spacing:.16em}.agent>.tag{color:#c9c9c9}.agent>.response-status{font-size:18px;line-height:1.4;color:#f3f3f3}
.empty{max-width:560px;padding:20px 12px;display:flex;flex-direction:column;align-items:center}.welcome-mark{width:116px;height:116px;margin:0 auto 12px;opacity:1}.welcome-name{font-size:10px;letter-spacing:.38em;color:#ddd}.empty h1{font-size:clamp(29px,4.2vw,42px);font-weight:400;line-height:1.14;letter-spacing:-.025em;margin:12px 0 12px}.empty p{font-size:13px;line-height:1.5;color:#aaa}.welcome-motto{font-family:"STKaiti","KaiTi","Kaiti SC","Noto Serif CJK SC",serif;font-size:15px;font-weight:400;line-height:1.8;letter-spacing:.04em;color:#c2c2c2;margin-top:16px;padding-top:10px;border-top:1px solid #1c1c1c;white-space:nowrap}
.trace{margin-top:18px;border-color:#252525;padding-top:4px}.trace summary,.secondary summary{font-size:11px;line-height:1.6;color:#a3a3a3;padding:8px 0;cursor:pointer}.trace .small{font:11px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;color:#999;white-space:pre-wrap}.activity{margin:14px 0;color:#b8b8b8;font-size:12px;line-height:1.8}.activity>div{padding:3px 0 3px 13px;border-left:1px solid #333}.activity>div:last-child{color:#ededed;border-left-color:#ddd}
.secondary{margin-top:10px}.secondary>.line{margin-top:4px}.line{gap:8px;margin-top:16px}.line button{border-radius:8px;padding:8px 13px;min-height:35px;background:#111;color:#ddd;border-color:#373737}.line button:hover:not(:disabled){background:#1b1b1b;border-color:#888}.line button:first-child{color:#f5f5f5}
.approval-panel{border-radius:10px;border-color:#393939;background:#111;padding:16px;font-size:13px;line-height:1.7}.approval-scope{margin:10px 0;color:#bbb;font-size:12px}.approval-scope summary{cursor:pointer;color:#dedede}.approval-scope>div{white-space:pre-wrap;padding:10px 0;overflow-wrap:anywhere}.approval-cost{font-size:11px;color:#aaa;margin:10px 0}.approval-panel.resolved{padding:10px 14px;background:transparent;border-color:#292929}.approval-panel.resolved>div:first-child,.approval-panel.resolved>.approval-cost{display:none}.approval-panel.resolved>.approval-status{margin-top:5px;padding-top:5px;border:0}.approval-panel [hidden]{display:none!important}
.foot{background:#070707}.composer{background:#111;border-color:#3c3c3c;border-radius:12px;box-shadow:none}#state{font-size:11px;color:#bbb}#elapsed{font-size:10px;min-height:0;margin:0 2px 8px;color:#999}#elapsed:empty{display:none}#goal{min-height:60px}.hint{font-size:11px;color:#999}.hint summary{font-size:10px;color:#999}
@media(max-width:400px){.head{padding:16px}.welcome-mark{width:92px;height:92px}.empty h1{font-size:29px}.welcome-motto{font-size:13px;letter-spacing:.04em;margin-top:14px}.agent>.response-status{font-size:16px}.approval-panel{padding:12px}.line button{font-size:11px;padding:7px 10px}}
@media(max-height:650px){.welcome-mark{width:70px;height:70px;margin-bottom:10px}.empty h1{font-size:27px;margin:10px 0 10px}.welcome-motto{margin-top:12px;padding-top:8px}.empty{padding:12px 8px}.empty p{font-size:12px}}

.project{white-space:normal;overflow:visible}.project-label{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.project-details{font-size:10px;color:#777;margin-top:2px}.project-details summary{cursor:pointer}.project-details>div{white-space:pre-wrap;overflow-wrap:anywhere;padding-top:4px;color:#999}
.reference-tools{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}.reference-tools button,.reference-chip button,#reference-preview button{background:#171717;color:#ccc;border:1px solid #383838;border-radius:7px;padding:5px 8px;font-size:11px}.reference-chip{display:flex;max-width:100%;margin:4px 0}.reference-chip button:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.reference-chip button:last-child{flex:none}#references,#reference-preview,#reference-note{grid-column:1/-1}#reference-note{font-size:10px;color:#999;margin:5px 0;overflow-wrap:anywhere}#reference-preview{max-height:260px;overflow:auto;font-size:11px}#reference-preview:empty{display:none}#reference-preview pre{white-space:pre-wrap;overflow-wrap:anywhere}#reference-preview img{max-width:100%}.reference-receipt{font-size:11px;color:#aaa;margin-top:12px;overflow-wrap:anywhere}

/* Compact reference picker: one trigger, descriptive choices, keyboard-friendly controls. */
.reference-tools{position:relative;align-items:flex-start;flex-direction:column;gap:0;margin-bottom:7px}
.reference-tools #add-context{min-height:34px;min-width:86px;padding:6px 11px;border:1px solid #414141;border-radius:9px;background:#1b1b1b;color:#efefef;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:space-between;gap:12px}
.reference-tools #add-context:hover:not(:disabled),.reference-tools #add-context[aria-expanded="true"]{background:#252525;border-color:#747474}
.add-chevron{font-size:13px;color:#999}
.add-menu{width:min(100%,310px);max-width:100%;margin-top:6px;border:1px solid #393939;border-radius:12px;padding:5px;background:#171717;box-shadow:0 9px 22px #0006}
.add-menu[hidden]{display:none!important}
.reference-tools .add-menu button{width:100%;min-height:50px;display:flex;align-items:center;gap:11px;text-align:left;border:0;border-radius:8px;padding:7px 10px;background:transparent;color:#eee;font-size:12px}
.reference-tools .add-menu button:hover:not(:disabled),.reference-tools .add-menu button:focus-visible{background:#2a2a2a}
.reference-tools .add-menu button:disabled{opacity:.42}
.add-menu button span:last-child{display:flex;flex-direction:column;gap:1px;min-width:0}
.add-menu button strong{font-size:12px;font-weight:600;color:inherit}
.add-menu button small{font-size:10px;color:#9e9e9e}
.menu-item-icon{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border:1px solid #393939;border-radius:7px;background:#202020;color:#dcdcdc;font-family:ui-monospace,monospace;font-size:15px;flex:none}
#reference-hint{font-size:10px;line-height:1.5;color:#999;max-width:315px;margin-top:5px}
#reference-hint:empty{display:none}
@media(max-height:600px){.reference-tools .add-menu button{min-height:40px;padding:4px 8px}.add-menu{max-height:34dvh;overflow:auto}}

/* Bound attachment growth to preserve the conversation and composer viewport. */
html{height:100%;overflow:hidden}
.head{max-height:25vh;overflow:auto;min-height:0}
.head>div{min-width:0}.head>button{flex-shrink:0}
#messages{flex:1 1 0;overflow-y:scroll}
#messages>.msg{flex-shrink:0}
.foot{flex:0 1 auto;min-height:0;max-height:65vh;max-height:65dvh;overflow-y:auto;overflow-x:hidden;scrollbar-gutter:stable}
.composer{min-width:0;grid-template-columns:minmax(0,1fr) auto}
#references{min-width:0;max-height:120px;max-height:min(120px,18dvh);overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;scrollbar-gutter:stable}
#reference-preview{min-width:0;max-height:min(180px,22vh);max-height:min(180px,22dvh);overscroll-behavior:contain}
.reference-chip{min-width:0}.reference-chip button:first-child{min-width:0}
#goal{overflow-y:auto;max-height:min(160px,22vh);max-height:min(160px,22dvh)}
#messages,#references,#reference-preview,.foot{scrollbar-width:thin;scrollbar-color:#686868 #111}
#messages::-webkit-scrollbar,#references::-webkit-scrollbar,#reference-preview::-webkit-scrollbar,.foot::-webkit-scrollbar{width:8px}
#messages::-webkit-scrollbar-thumb,#references::-webkit-scrollbar-thumb,#reference-preview::-webkit-scrollbar-thumb,.foot::-webkit-scrollbar-thumb{background:#686868;border-radius:4px}
@media(max-height:600px){.head{padding-top:8px;padding-bottom:8px}.sub{display:none}.foot{padding-top:6px;padding-bottom:6px}#references{max-height:64px}#reference-preview{max-height:64px}.reference-tools{margin-bottom:4px}.reference-tools button{padding:3px 6px}#reference-note{max-height:30px;overflow-y:auto}#goal{min-height:42px;max-height:80px}.hint{max-height:34px;overflow-y:auto}#state{margin-bottom:5px}}

/* Professional assistant output: structured, readable, and always plain-DOM safe. */
.agent>.response-status,.agent>.task-duration{font:500 12px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:.01em}
.agent>.response-status{color:#a1a1a1;margin-bottom:10px}
.assistant-prose{font:400 14px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#dedede;overflow-wrap:anywhere;word-break:normal;min-width:0;white-space:normal;margin-top:11px}
.assistant-prose> :first-child{margin-top:0}.assistant-prose> :last-child{margin-bottom:0}
.assistant-prose p{margin:0 0 13px;line-height:1.8}.assistant-prose h3{font-size:15px;line-height:1.5;font-weight:650;color:#f3f3f3;margin:22px 0 9px;letter-spacing:0}.assistant-prose h4{font-size:14px;line-height:1.5;color:#ededed;margin:18px 0 8px}
.assistant-prose strong{color:#f0f0f0;font-weight:650}.assistant-prose ul,.assistant-prose ol{padding-left:22px;margin:0 0 15px}.assistant-prose li{padding-left:3px;margin:5px 0}.assistant-prose li::marker{color:#888}
.assistant-prose code{font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace;color:#e7e7e7;background:#222;border:1px solid #313131;border-radius:4px;padding:1px 4px;overflow-wrap:anywhere}
.assistant-prose .response-code{background:#121212;border:1px solid #343434;border-radius:9px;overflow:hidden;margin:13px 0 18px}
.assistant-prose .response-code-lang{padding:5px 13px;background:#1b1b1b;border-bottom:1px solid #2d2d2d;color:#aaa;font:11px/1.5 ui-monospace,monospace}
.assistant-prose pre{margin:0;padding:13px 15px;overflow-x:auto;white-space:pre;tab-size:4;scrollbar-width:thin}
.assistant-prose pre code{background:transparent;border:0;padding:0;color:#ddd;white-space:pre;overflow-wrap:normal;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}
.assistant-prose blockquote{margin:12px 0;padding:5px 0 5px 15px;border-left:2px solid #666;color:#b9b9b9}.assistant-prose hr{border:0;border-top:1px solid #333;margin:18px 0}
.assistant-prose a{color:#dedede;text-decoration:underline;text-decoration-color:#777;text-underline-offset:3px}
.assistant-prose .response-table-wrap{overflow-x:auto;margin:12px 0 17px;border:1px solid #333;border-radius:7px}.assistant-prose table{border-collapse:collapse;width:100%;font-size:12px}.assistant-prose th,.assistant-prose td{padding:9px 11px;border-bottom:1px solid #303030;text-align:left;vertical-align:top;white-space:normal}.assistant-prose th{font-weight:600;color:#eee;background:#1d1d1d}.assistant-prose tr:last-child td{border-bottom:0}
@media(max-width:400px){.assistant-prose{font-size:13px}.assistant-prose pre{padding:10px}.assistant-prose h3{font-size:14px}}

/* Compact composer: text above a single attachment/action toolbar. */
.composer{grid-template-columns:minmax(0,1fr) auto;column-gap:8px;row-gap:0}
#references{grid-row:1;grid-column:1/-1}
#references:empty{display:none}
#reference-preview{grid-row:2;grid-column:1/-1}
#goal{grid-row:3;grid-column:1/-1;height:56px;min-height:56px;padding:2px 2px 6px;line-height:1.5}
#reference-note{grid-row:4;grid-column:1/-1;margin:0 2px 6px;line-height:1.5}
#warn{grid-row:5;grid-column:1/-1}
.reference-tools{grid-row:6;grid-column:1;min-width:0;margin:0;align-self:start}
.actions{grid-row:6;grid-column:2;align-self:start;margin:0;gap:8px}
.actions #cancel{margin-right:0}
.reference-tools #add-context,.actions button{min-height:34px}
.add-menu{width:310px}
@media(max-width:400px){.composer{column-gap:6px}.actions{gap:6px}.actions button{padding:6px 10px}.reference-tools #add-context{min-width:74px;padding:6px 9px;gap:8px}.add-menu{width:240px}}
@media(max-height:600px){#goal{height:42px;min-height:42px}.reference-tools{margin-bottom:0}#reference-note{margin-bottom:4px}}
.assistant-prose .response-math{font-size:1.08em}.assistant-prose .response-math-display{display:block;max-width:100%;overflow-x:auto;padding:10px 0}.assistant-prose .response-math-block{margin:16px 0;max-width:100%;overflow-x:auto}.assistant-prose math{color:inherit}.assistant-prose .response-math-fallback{white-space:pre-wrap;font-family:monospace}

/* VODER_PUBLIC_PROGRESS_UI40: presentation changes only. */
.agent .activity{height:clamp(156px,31vh,340px);min-height:156px;max-height:340px;overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;scrollbar-width:thin;scrollbar-color:#777 #151515;scrollbar-gutter:stable;padding:11px 13px;margin:12px 0 14px;border:1px solid #313131;border-radius:9px;background:#0e0e0e;font:12px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.agent .activity>div{padding:5px 0 5px 13px;border-left:1px solid #383838;overflow-wrap:anywhere;color:#c1c1c1}
.agent .activity>div:last-child{color:#f0f0f0;border-left-color:#dedede}
.agent .activity[hidden],.agent .phase[hidden]{display:none!important}
.agent .activity::-webkit-scrollbar{width:7px}
.agent .activity::-webkit-scrollbar-thumb{background:#666;border-radius:4px}
.approval-panel{padding:11px 13px}
.approval-scope{margin:5px 0;font-size:11px;max-height:92px;overflow-y:auto;scrollbar-width:thin}
.approval-scope>div{padding:5px 0;line-height:1.5}
@media(max-height:600px){.agent .activity{height:clamp(120px,25vh,220px);min-height:120px}.approval-scope{max-height:66px}}

/* VODER_PUBLIC_PROGRESS_UI41: compact professional activity surface, no other view redesign. */
.agent .activity{height:clamp(205px,35vh,420px);max-height:420px;margin:12px 0 12px;padding:10px 13px;border-color:#303030;border-radius:10px;background:#101010}
.agent .activity:before{content:"LIVE EXECUTION";display:block;position:sticky;top:-10px;z-index:1;padding:10px 0 8px;background:#101010;border-bottom:1px solid #272727;margin:0 0 5px;color:#969696;font:600 10px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:.14em}
.agent .activity>div{line-height:1.55;padding:7px 3px 7px 13px;border-left:2px solid #343434;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:12px}
.agent .activity>div:last-child{border-left-color:#e8e8e8;color:#f1f1f1}
.agent>.response-status{margin-bottom:2px}
.approval-scope{max-height:86px;overflow:auto}
@media(max-height:600px){.agent .activity{height:clamp(135px,27vh,245px);min-height:135px}}
.agent>.tag{order:-30}.agent>.response-status{order:-10}.agent>.task-duration{order:-20;display:flex;align-items:center;gap:8px;align-self:flex-start;width:100%;margin:0 0 14px;padding:0 0 10px;border-bottom:1px solid #282828;text-align:left;color:#b8b8b8;font-variant-numeric:tabular-nums}
.agent>.task-duration::before{content:"";display:block;flex:none;width:1em;height:1em;background:center/contain no-repeat url("data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27 fill=%27none%27 stroke=%27%23dedede%27 stroke-width=%271.25%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27%3E%3Ccircle cx=%278%27 cy=%278%27 r=%276%27/%3E%3Cpath d=%27M8 4.5V8l2.5 1.5%27/%3E%3C/svg%3E")}
@media(forced-colors:active){.agent>.task-duration{color:CanvasText}.agent>.task-duration::before{background:none;border:1px solid currentColor;border-radius:50%}}\n.assistant-prose [hidden]{display:none!important}.assistant-prose[aria-busy="true"]{overflow-anchor:none}\n</style></head><body><div class="head"><div><strong>VODER</strong><div class="sub">VERIFIABLE INTELLIGENCE</div><div class="project" id="workspace" role="status">Project selected when a task starts</div></div><button id="new" title="Clear this view only; never cancels a running task">New chat</button></div><template id="welcome-template">${welcome}</template><main id="messages" aria-live="polite">${welcome}</main><div class="foot"><div id="state" role="status">Ready · one task at a time</div><div id="elapsed" class="small" role="status"></div><div class="composer" id="reference-dropzone"><div class="reference-tools"><button id="add-context" type="button" aria-haspopup="true" aria-expanded="false" aria-controls="add-menu" title="Add reference context">＋ Add <span class="add-chevron" aria-hidden="true">⌄</span></button><div id="add-menu" class="add-menu" hidden><button id="ref-files" type="button"><span class="menu-item-icon" aria-hidden="true">＋</span><span><strong>Files</strong><small>Choose documents or code</small></span></button><button id="ref-selection" type="button" disabled><span class="menu-item-icon" aria-hidden="true">⌗</span><span><strong>Selection</strong><small>Selected text in the editor</small></span></button><button id="ref-editor" type="button" disabled><span class="menu-item-icon" aria-hidden="true">▤</span><span><strong>Open file</strong><small>Entire active editor buffer</small></span></button></div><div id="reference-hint" role="status" aria-live="polite"></div></div><div id="references" aria-live="polite"></div><div id="reference-preview"></div><textarea id="goal" maxlength="16384" placeholder="Describe what you want to build, understand, or improve…" aria-label="Voder coding task"></textarea><div id="reference-note">Add files or use @path to bring your code into context.</div><div id="warn" role="alert"></div><div class="actions"><button id="cancel" disabled>Stop</button><button id="send" title="Send (Enter) &#10;New line (Shift+Enter)" aria-keyshortcuts="Enter">Send ↗</button></div></div></div><script nonce="${nonce}">
 ${mathScript}
 (function(){'use strict';const developerDiagnostics=${showDeveloperDiagnostics};const outputPresenter=(${createAssistantPresentation.toString()})(document,typeof katex==='undefined'?null:katex);const api=acquireVsCodeApi(),el=id=>document.getElementById(id),list=el('messages'),input=el('goal');let busy=false,session=null,runBubble=null,revision=0,phaseIndex=-1,outcomeUncertain=false,entryPhase='native';const reviewButtons=new Map(),approvalPanels=new Map();let startedAt=0,lastEventAt=0;
 const durationText=(${formatTaskDuration.toString()}),taskClock=(${createTaskClock.toString()})(()=>typeof performance!=='undefined'?performance.now():Date.now());
 function durationRow(){if(!runBubble)return null;if(runBubble.duration)return runBubble.duration;const n=document.createElement('div');n.className='task-duration small';n.setAttribute('role','note');if(typeof runBubble.node.insertBefore==='function')runBubble.node.insertBefore(n,runBubble.content);else runBubble.node.append(n);runBubble.duration=n;return n;}
 function refreshDuration(){const footer=el('elapsed');if(footer){footer.textContent='';footer.hidden=true;}if(!taskClock.started)return;const n=durationRow();if(n)n.textContent=(taskClock.frozen?(taskClock.label||'Completed in '):'Elapsed ')+durationText(taskClock.elapsed());}
 function freezeDuration(label='Completed in '){if(!runBubble)return;taskClock.freeze();taskClock.label=taskClock.label||label;refreshDuration();}
 let followOutput=true;
 if(typeof list.addEventListener==='function')list.addEventListener('scroll',()=>{followOutput=list.scrollHeight-list.scrollTop-list.clientHeight<64;},{passive:true});
 const progressiveOutput=(${createProgressiveOutput.toString()})(document,{render:outputPresenter.render,
  requestFrame:typeof requestAnimationFrame==='function'?requestAnimationFrame:undefined,cancelFrame:typeof cancelAnimationFrame==='function'?cancelAnimationFrame:undefined,
  now:()=>typeof performance!=='undefined'?performance.now():Date.now(),reducedMotion:()=>typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches,
  onUpdate:()=>{if(followOutput&&!(document.getSelection?.()?.isCollapsed===false))list.scrollTop=list.scrollHeight;}});
 if(typeof setInterval==='function')setInterval(refreshDuration,1000);
 const noticeText={CHAT_NATIVE_DECLINED:'Approval declined. Write execution did not start.',CHAT_NATIVE_NOT_APPROVED:'Approval is required before write execution.',CHAT_INVALID_GOAL:'Enter a task of at most 16 KiB.',CHAT_SCOPE_PLACEHOLDER:'Replace example filenames with actual project paths.',CHAT_SCOPE_REQUIRED:'Specify an existing source file and an independent test.',CHAT_SCOPE_GIT_UNAVAILABLE:'Open a project with a readable Git file index.',CHAT_SCOPE_FILE_NOT_TRACKED:'The selected file is not tracked in this repository.',CHAT_SCOPE_AMBIGUOUS_PATH:'Use the full repository-relative path to disambiguate the file.',CHAT_SCOPE_AMBIGUOUS_TARGET:'Select one existing source file for this task.',CHAT_SCOPE_AMBIGUOUS_TEST:'Select the independent test to use for validation.',CHAT_SCOPE_FILE_MISSING:'The selected tracked file is missing from the workspace.',CHAT_SCOPE_UNSAFE_FILE:'Select a regular source file, not a symbolic link.',CHAT_FILE_NOT_TRACKED:'Source and validator must be committed and unchanged before execution.',CHAT_TARGET_NOT_TRACKED:'The source must be committed and unchanged before execution.'};
 const projectLabels=new Map();let pendingProjectLabel='';
 function labelFromGoal(goal){const match=String(goal||'').match(/^\\s*Create a new (?:multi-file )?web project:\\s*([^\\r\\n]+)/i);return match?match[1].replace(/[\\u0000-\\u001f\\u007f]/g,'').trim().slice(0,96).replace(/[.]$/,''):'';}
 function showProject(path){path=String(path||'');const base=path.split(/[\\\\/]/).filter(Boolean).pop()||'';const generated=/^(?:multi|web|project)-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(base);if(generated&&pendingProjectLabel){projectLabels.set(path,pendingProjectLabel);while(projectLabels.size>20)projectLabels.delete(projectLabels.keys().next().value);}pendingProjectLabel='';const name=generated?(projectLabels.get(path)||'Generated project'):(base||'No project selected');const project=el('workspace');if(!project)return;const label=document.createElement('span');label.className='project-label';label.textContent='Project: '+name;const details=document.createElement('details');details.className='project-details';const summary=document.createElement('summary');summary.textContent='Project details';const location=document.createElement('div');const segments=path.replaceAll(String.fromCharCode(92),'/').split('/');const privateCopy=segments.some((x,i)=>x==='working-copies'&&/^[a-f0-9-]{36}$/i.test(segments[i+1]||''));location.textContent=privateCopy?'Private working copy (internal path hidden)':path;details.append(summary,location);project.replaceChildren(label,details);}
 function preparationText(m){return noticeText[m.code]||String(m.message||'The task has not started');}
 function setBusy(v){busy=v;el('send').disabled=v;el('cancel').disabled=!v||!session;el('state').textContent=v?'Voder is running · do not close the SSH session':'Ready · one task at a time';if(v&&!startedAt){startedAt=Date.now();taskClock.start();}if(!v)startedAt=0;refreshDuration();}
 function clearEmpty(){el('empty')?.remove();}
 function bubble(role,title,body){clearEmpty();const node=document.createElement('section');node.className='msg '+role;const tag=document.createElement('div');tag.className='tag';tag.textContent=title;const content=document.createElement('div');content.textContent=body;content.className="response-status";node.append(tag,content);let trace,phase;if(role==='agent'){phase=document.createElement('div');phase.className='phase';phase.setAttribute('aria-hidden','true');for(let i=0;i<4;i++)phase.append(document.createElement('span'));node.append(phase);if(developerDiagnostics){trace=document.createElement('details');trace.className='trace';trace.open=false;const label=document.createElement('summary');label.textContent='Developer diagnostics';trace.append(label);node.append(trace);}}list.append(node);list.scrollTop=list.scrollHeight;return{node,content,trace,phase};}
 function detail(text,trace=false){if(!developerDiagnostics||!runBubble?.trace)return;const small=document.createElement('div');small.className='small';small.textContent=outputPresenter.redact(text);runBubble.trace.append(small);}

 // VODER_PUBLIC_PROGRESS_UI40: public evidence-based UI updates; never private chain-of-thought.
 const progressCounts=new Map();let progressClosed=false,priorCheckFailed=false,modelPasses=0,visualPasses=0,publicSummarySeen=false;
 function closeProgress(){progressClosed=true;const box=runBubble?.node?.querySelector('.activity');if(box)box.hidden=true;if(runBubble?.phase)runBubble.phase.hidden=true;}
 function activity(text,key){if(!runBubble||!text||progressClosed)return;const safe=outputPresenter.redact(String(text)).slice(0,210);let box=runBubble.node.querySelector('.activity');if(!box){box=document.createElement('div');box.className='activity';box.setAttribute('role','log');box.setAttribute('aria-label','Task activity');box.setAttribute('aria-live','off');runBubble.node.append(box);}const nearBottom=box.scrollTop+box.clientHeight>=box.scrollHeight-28;const id=String(key||safe).slice(0,160);let record=progressCounts.get(id);if(record){record.count++;record.item.textContent=safe+(record.count>1?' · '+record.count+' occurrences':'');box.append(record.item);}else{const item=document.createElement('div');item.textContent=safe;box.append(item);progressCounts.set(id,{item,count:1});}while(box.children.length>72){const first=box.children[0];for(const [k,v] of progressCounts){if(v.item===first){progressCounts.delete(k);break;}}first.remove();}if(nearBottom)box.scrollTop=box.scrollHeight;}
 function presentEvent(e){if(progressClosed||!runBubble)return;const type=String(e.type||''),tool=String(e.tool||''),status=String(e.status||'');let line='',key='';
  const tools={voder_list_files:'Inspecting the repository file inventory',voder_read_file:'Reading implementation files',voder_verify_candidate:'Checking the current candidate',voder_commit_verified_candidate:'Requesting a checked commit',voder_ext_project_capabilities:'Inspecting project capabilities',voder_ext_project_syntax:'Checking project syntax',voder_ext_project_dependencies:'Reviewing dependency compatibility',voder_task_review_render:'Reviewing rendered images against the task',voder_task_render_svg:'Rendering a visual artifact',voder_task_browser_inspect:'Inspecting the project in a browser',voder_task_browser:'Inspecting browser output',voder_task_open_browser:'Opening browser inspection'};
  if(type==='model.progress_summary'){if(typeof e.summary!=='string'||e.summary.length<12||e.summary.length>160)return;line=e.summary;key='model-summary:'+line;publicSummarySeen=true;}
  else if(type==='model.request_start'){modelPasses++;line=modelPasses===1?'Reading the task and determining the next action':priorCheckFailed?'Analyzing the failed checks before the next revision':'Analyzing recent tool results and selecting the next action';key='model:'+((priorCheckFailed)?'recovery':'normal');}
  else if(type==='model.tool_execution_start'){const short=tool.startsWith('voder_')?tool.replace(/^voder_(?:ext_|task_)?/,'').replace(/_/g,' '):'';line=tools[tool]||(short?'Using '+short+' to inspect or modify the task':'Executing a project tool');key='tool:'+tool;}
  else if(type==='model.tool_execution_end'&&['FAIL','ERROR','DENIED'].includes(status)){line=(tools[tool]||'A project tool')+' · reported '+status.toLowerCase()+(e.code?' ('+String(e.code).slice(0,48)+')':'');key='tool-error:'+tool+':'+status;}
  else if(type==='model.image_request'){line='Preparing visual evidence for model inspection';key='vision:request';}
  else if(type==='model.vision_selection'){line=e.imageCount>0?'Visual evidence selected: '+e.imageCount+' frame(s)':'Visual evidence selection recorded';key='vision:selection';}
  else if(type==='task.browser_stage'){if(status==='BROWSER_SANDBOX_CREATE'){visualPasses++;line='Preparing isolated visual inspection'+(visualPasses>1?' · pass '+visualPasses:'');key='browser:prepare';}else if(status==='BROWSER_EXECUTION'){line='Running browser checks and capturing rendered frames';key='browser:execute';}else if(status==='BROWSER_SANDBOX_TERMINATE'){line='Releasing the isolated browser after inspection';key='browser:close';}}
  else if(type==='sandbox.candidate_requested'){line='Running configured checks on the candidate in isolation';key='check:start';}
  else if(type==='sandbox.candidate_settled'){priorCheckFailed=status==='FAIL';line=status==='PASS'?'Candidate passed its configured checks':status==='FAIL'?'Candidate failed configured checks; revision is needed':'Candidate checks reported '+(status||'an outcome');key='check:'+status;}
  else if(type==='vgr.action_requested'){line='Submitting the checked candidate to native VGR';key='vgr:requested';}
  else if(type==='vgr.action_settled'){line=status==='COMMITTED'?'Native VGR committed the checked candidate':'Native VGR returned '+(status||'a decision');key='vgr:'+status;}
  else if(type==='task.cancel_requested'){line='Cancellation requested; awaiting confirmation';key='task:cancel';}
  else if(type==='task.failed'){line='Task execution reported a failure';key='task:failure';}
  if(!line)return;activity(line,key);if(type==='model.progress_summary'||!publicSummarySeen)runBubble.content.textContent=line;
 }
 function friendlyError(message){const text=String(message||'');if(/timeout|timed out/i.test(text))return 'The request timed out. Check task details before trying again.';if(/connection|transport|RPC/i.test(text))return 'The connection was interrupted. Check task details before trying again.';return 'The task could not complete. Check task details for more information.';}
 function action(title,command,target){if(!runBubble)return;const owner=runBubble;const b=document.createElement('button');b.textContent=title;const feedback=document.createElement('div');feedback.className='review-feedback';feedback.setAttribute('role','status');owner.node.append(feedback);const key=target+':'+command;reviewButtons.set(key,{b,feedback,title});b.addEventListener('click',()=>{b.disabled=true;feedback.textContent=command==='diff'?'Opening checked changes…':command==='preview'?'Opening preview…':command==='open-project'?'Opening project…':'Opening task details…';api.postMessage({type:'action',command,sessionId:target});});let container=owner.node;if(!['diff','preview','open-project'].includes(command)){container=owner.node.querySelector('.secondary');if(!container){container=document.createElement('details');container.className='secondary';const title=document.createElement('summary');title.textContent='More options';container.append(title);owner.node.append(container);}}let row=Array.from(container.children).find(n=>n.className==='line');if(!row){row=document.createElement('div');row.className='line';container.append(row);}row.append(b);}
 function stage(index,label){if(!runBubble||index<phaseIndex)return;phaseIndex=index;if(!publicSummarySeen)runBubble.content.textContent=label;Array.from(runBubble.phase.children).forEach((n,i)=>n.className=i<=index?'on':'');}

 function submit(){if(busy||referencesLoading)return;const goal=input.value.trim();if(!goal){el('warn').textContent='Enter a coding task first.';return;}if(new TextEncoder().encode(goal).length>16384){el('warn').textContent='Maximum size is 16 KiB.';return;}followOutput=true;taskClock.reset();taskClock.label='';pendingProjectLabel=labelFromGoal(goal);el('warn').textContent='';bubble('user','YOU',goal);runBubble=bubble('agent','VODER','Understanding your request. Investigate first; approve changes when needed.');progressCounts.clear();progressClosed=false;priorCheckFailed=false;modelPasses=0;visualPasses=0;publicSummarySeen=false;session=null;phaseIndex=-1;outcomeUncertain=false;lastEventAt=0;setBusy(true);input.value='';api.postMessage({type:'submit',goal});}
 el('send').addEventListener('click',submit);input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();submit();}});
 el('cancel').addEventListener('click',()=>{if(busy&&session){activity('Stopping the task · awaiting confirmation');detail('Cancellation requested; confirmation is still pending.');api.postMessage({type:'cancel',sessionId:session});}});
 el('new').addEventListener('click',()=>{if(busy||referencesLoading){el('warn').textContent='Wait for the task to settle before clearing this view.';return;}list.replaceChildren(el('welcome-template').content.cloneNode(true));reviewButtons.clear();approvalPanels.clear();progressiveOutput.cancelAll();taskClock.reset();taskClock.label='';refreshDuration();runBubble=null;session=null;phaseIndex=-1;outcomeUncertain=false;el('warn').textContent='';el('state').textContent='Ready · one task at a time';revision++;api.postMessage({type:'clear'});});
 window.addEventListener('message',evt=>{const m=evt.data;if(!m||typeof m!=='object')return;handleReferenceMessage(m);
 if(m.type==='phase_started'){entryPhase=m.phase;phaseIndex=-1;session=null;setBusy(true);if(runBubble)runBubble.content.textContent=m.phase==='inspect'?'Investigating the project · no edits or tests':m.phase==='respond'?'Working on your request':'Checking the proposed scope and approval';}
 if(m.type==='task_planned'&&runBubble){setBusy(true);el('cancel').disabled=true;runBubble.content.textContent='Task planned · review execution scope';const d=document.createElement('div');d.className='assistant-prose';progressiveOutput.render(String(m.answer||''),d);runBubble.node.append(d);activity('Preparation complete · execution continues after scope approval');}
 if(m.type==='task_delivered'){freezeDuration();closeProgress();const names=(m.files||[]).filter(x=>typeof x==='string').map(x=>x.split('/').pop()).slice(0,16);if(runBubble){runBubble.content.textContent='Files saved to your project';const summary=document.createElement('div');summary.className='assistant-prose';summary.textContent=names.join(' · ');runBubble.node.append(summary);}activity('Verified files saved to your project');action('Open artifact','preview',m.sessionId);action('Open project folder','open-project',m.sessionId);}
 if(m.type==='inspection_complete'&&runBubble){if(!m.continuing){freezeDuration();closeProgress();}setBusy(!!m.continuing);runBubble.content.textContent=m.status==='INSPECTED'?'Investigation complete · no files changed':'Investigation is incomplete';const answer=document.createElement('div');answer.className='assistant-prose';if(typeof document.createTextNode==='function')progressiveOutput.render(m.answer,answer);else answer.textContent=String(m.answer||'');runBubble.node.append(answer);if(m.scope)detail('Proposed scope: '+m.scope.target+' · '+m.scope.test+' (write approval pending)');if(m.continuing)detail('Next: review this scope and approve the change.');else el('state').textContent='Ready · investigation complete';}
 if(m.type==='approval_requested'){if(approvalPanels.has(m.requestId))return;if(!runBubble){runBubble=bubble('agent','VODER','Awaiting approval');session=null;}setBusy(true);runBubble.content.textContent='Awaiting write approval · task paused';const panel=document.createElement('div');panel.className='approval-panel';const text=document.createElement('div');text.textContent=m.publication===true?'Publish verified files to your original project?':m.project===true?'Create a new isolated project using only the files listed below.':'Allow changes only to the selected files.';const scope=document.createElement('details');scope.className='approval-scope';scope.open=true;const scopeTitle=document.createElement('summary');scopeTitle.textContent='Files and validation';const scopeText=document.createElement('div');const files=Array.isArray(m.files)?m.files:[];const safeFiles=files.filter(f=>typeof f==='string'&&f.length<=240&&f.split('').every(c=>{const x=c.charCodeAt(0);return x>=32&&x!==127&&![42,58,60,62,63,92,124].includes(x)&&!(x>=0x202a&&x<=0x202e)&&!(x>=0x2066&&x<=0x2069)})&&!f.startsWith('/')&&!f.split('/').some(x=>!x||x==='.'||x==='..')).slice(0,16);const scopeValid=safeFiles.length>0&&safeFiles.length===files.length;scopeText.textContent=scopeValid?'Files ('+safeFiles.length+'):\\n'+safeFiles.map(f=>'• '+f).join('\\n')+'\\n'+String(m.operation||'Only these files may be changed.')+'\\n'+String(m.checksSummary||'Validation details are not displayed here.'):'Exact file scope unavailable. Decline this approval and inspect the task.';scope.append(scopeTitle,scopeText);const costs=document.createElement('div');costs.className='approval-cost';costs.textContent=m.publication===true?'This publishes already verified files only; no additional model or sandbox run.':'Model and sandbox usage may incur costs. Passing configured checks does not certify appearance or interaction.';panel.append(text,scope,costs);const row=document.createElement('div');row.className='line';for(const choice of [{label:m.publication===true?'Approve delivery':'Approve writes',decision:'approve'},{label:'Not now',decision:'decline'}]){const button=document.createElement('button');button.textContent=choice.label;if(choice.decision==='approve'&&!scopeValid)button.disabled=true;button.addEventListener('click',()=>{Array.from(row.children).forEach(b=>b.disabled=true);api.postMessage({type:'authorization_decision',requestId:m.requestId,decision:choice.decision});});row.append(button);}panel.append(row);approvalPanels.set(m.requestId,{panel,row,scope});runBubble.node.append(panel);el('state').textContent='Awaiting your approval for these files';}
 if(m.type==='authorization_resolved'){const card=approvalPanels.get(m.requestId);if(card){card.scope.open=false;card.row.hidden=true;card.panel.className='approval-panel resolved';Array.from(card.row.children).forEach(b=>b.disabled=true);let note=card.panel.querySelector('.approval-status');if(!note){note=document.createElement('div');note.className='approval-status';note.setAttribute('role','status');card.panel.append(note);}note.textContent=m.approved?'Approved · this task only':'Write scope declined · no new write authority';}el('state').textContent=m.approved?'Write scope approved · preparing changes':'Write scope declined · no new write authority';}
 if(m.type==='input_required'||m.type==='not_started'){freezeDuration('Stopped after ');closeProgress();setBusy(false);const context=m.scopeContext;el('warn').textContent=context?String(context.reason||preparationText(m))+String.fromCharCode(10)+'Project: '+String(context.workspace||'').split(/[\\/]/).filter(Boolean).pop()+String.fromCharCode(10)+'Requested: '+(Array.isArray(context.requested)&&context.requested.length?context.requested.join(', '):'No explicit file path')+String.fromCharCode(10)+'Your original request is unchanged.':preparationText(m);if(runBubble)runBubble.content.textContent=m.type==='input_required'?'Scope needs attention · execution has not started':'Task not started';if(!input.value&&typeof m.goal==='string')input.value=m.goal;}
 if(m.type==='workspace_selected')showProject(m.path);
 if(m.type==='workspace_isolated')activity('Verifying changes in an isolated workspace');

 if(m.type==='publication_pending'){activity('Verification complete · preparing project delivery');if(runBubble)runBubble.content.textContent='Saving verified files to your project…';}
 if(m.type==='publication_incomplete'){el('warn').textContent='Verified changes remain in the private workspace; original project was not updated. Review delivery status before retrying.';}
 if(m.type==='created'){session=m.sessionId;}
 if(m.type==='accepted'){session=m.sessionId;el('cancel').disabled=false;el('state').textContent=entryPhase==='plan'?'Planning task · execution has not started':entryPhase==='inspect'?'Investigation in progress':entryPhase==='respond'?'Working on your request':'Approved · execution in progress';if(runBubble)runBubble.content.textContent=entryPhase==='plan'?'Planning files and checks · no writes yet':entryPhase==='inspect'?'Investigating the project · no edits or tests':'Preparing the task · no result saved yet';}
 if(m.type==='event'&&runBubble&&m.sessionId===session){lastEventAt=Date.now();const e=m.event||{};let state=[e.type,e.tool,e.status,e.code].filter(Boolean).join(' · ');detail(state.slice(0,180),true);if(e.type==='model.request_start')stage(0,entryPhase==='inspect'?'Reading the project and analyzing your request…':entryPhase==='respond'?'Analyzing your request…':'Inspecting the task and preparing a candidate…');if(e.type==='sandbox.candidate_requested')stage(1,'Running independent sandbox checks…');if(e.type==='sandbox.candidate_settled')stage(1,e.status==='PASS'?'Configured checks passed':e.status==='FAIL'?'Checks found issues':'Check results received');if(e.type==='vgr.action_requested')stage(2,'Saving checked changes…');if(e.type==='vgr.action_settled')stage(2,'Confirming the saved result…');presentEvent(e);}
 if(m.type==='settled'&&m.errorCode?.startsWith('REFERENCE_'))el('warn').textContent=m.errorCode==='REFERENCE_IMAGE_UNSUPPORTED'?'The selected model/runtime cannot accept these images; choose an image-capable model or remove the images.':m.errorCode==='REFERENCE_IMAGE_CHANGED'?'The image adapter omitted or transformed an attachment. No request containing that incomplete image context was sent.':'Reference processing failed: '+m.errorCode;
 if(m.type==='settled'){freezeDuration(m.status==='CANCELLED'?'Stopped after ':'Completed in ');closeProgress();if(m.taskAcceptance?.blocker)el('warn').textContent='Task preparation blocked. Review task details or contact the administrator.';session=m.sessionId;setBusy(false);el('state').textContent=m.presentation?.footer||(m.verified?'Ready · checked changes saved':'Task ended · no verified delivery');if(runBubble){if(m.verified)stage(3,'');runBubble.content.textContent=m.presentation?.title||(m.verified?'Changes saved · configured checks passed':'Task ended · result not verified');if(!['ANSWER','RENDERED_RESPONSE'].includes(m.presentation?.kind))activity(m.status==='DELIVERED'?'Verified changes saved':m.status==='VERIFIED_NOT_DELIVERED'?'Verification passed · project delivery incomplete':'Task complete');if(m.verified)activity('Configured checks passed · changes saved'); /* Commit identifiers remain available through explicit task review, not the chat transcript. */if(m.draft){activity(m.presentation?.responseLabel||'Draft response · no verified changes saved');const d=document.createElement('div');d.className='assistant-prose';if(typeof document.createTextNode==='function')progressiveOutput.render(m.draft,d);else d.textContent=m.draft;runBubble.node.append(d);}if(m.missing?.length)detail('Required input: '+m.missing.join(', '));if(m.verified)action('View changes','diff',m.sessionId);action('Task details','status',m.sessionId);}}
 if(m.type==='memory_exported'){closeProgress();if(runBubble)runBubble.content.textContent='Memory exported · '+m.entries+' entries · '+(m.paths||[m.path]).map(x=>String(x).split('/').pop()).join(' · ');setBusy(false);el('state').textContent='Memory file ready';}
 if(m.type==='memory_imported'){closeProgress();if(runBubble)runBubble.content.textContent='Portable project memory imported · '+m.entries+' entries';setBusy(false);el('state').textContent='Memory imported';}
 if(m.type==='memory_policy_result'||m.type==='memory_search_results'){closeProgress();if(runBubble)runBubble.content.textContent=String(m.summary||'Memory command completed');setBusy(false);el('state').textContent='Memory ready';}
 if(m.type==='project_delivered'){freezeDuration();detail('Generated project: '+String(m.path||''));activity('Project ready · appearance and interaction still need review');action('Open preview','preview',m.sessionId);action('Open project','open-project',m.sessionId);action('File checks','tools',m.sessionId);action('Runtime checks','runtime',m.sessionId);}
 if(m.type==='uncertain'||m.type==='error'){freezeDuration('Stopped after ');closeProgress();detail(String(m.message||m.code||'Unknown error'));activity(m.type==='uncertain'?'Connection outcome unknown · task was not replayed':'The task reported an error');if(m.type==='uncertain'){outcomeUncertain=true;if(m.sessionId)session=m.sessionId;}setBusy(false);el('warn').textContent=m.type==='uncertain'?'Outcome unknown. Check task details before any retry.':(String(m.code||'').startsWith('TASK_')?String(m.message||m.code):friendlyError(m.message||m.code));if(runBubble)runBubble.content.textContent=outcomeUncertain?(entryPhase==='inspect'?'Investigation outcome unknown · no automatic retry':'Execution outcome unknown · inspect session evidence before retrying'):'Task incomplete · review the error details';}
 if(m.type==='action_result'){const entry=reviewButtons.get(m.sessionId+':'+m.command);if(entry){entry.b.disabled=m.status==='OPENING';entry.feedback.dataset.status=m.status;entry.feedback.textContent=m.status==='ERROR'?'Unable to open this view. Check task details.':m.status==='NOT_AVAILABLE'?'This view is not available for this task.':m.status==='OPENING'?'Opening…':'Opened';detail(String(m.message||m.status||'Review finished'));}}
 if(m.type==='cancel_ack'){activity('Cancellation response received · saved changes are not automatically undone');detail('Cancellation acknowledgement: '+String(m.status||'UNKNOWN')+'; not proof of rollback.');}
 if(m.type==='info'){detail(String(m.text||''));if(String(m.text||'').startsWith('Native VGR authorized for EXISTING file:'))el('state').textContent='Write scope approved · preparing changes';}
 });
 ${referenceScript}
 api.postMessage({type:'ready'});
 })();</script></body></html>`;
}
const {ProjectConversationMemory}=require('./project-conversation-memory.cjs');
const {command:parseMemoryPolicyCommand}=require('./project-memory-policy.cjs');
// Explicit user commands; ordinary requests remain normal Voder tasks.
function memoryCommand(goal){
 if(typeof goal!=='string')return null;
 const g=goal.trim();
 // Keep implementation requests in the ordinary coding route; only explicit
 // memory management requests are intercepted.  Natural-language export may
 // request organization/summary, without pretending to invoke a model.
 if(/^\/memory\s+export\b/i.test(g))return 'export';
 if(/^\/memory\s+import\b/i.test(g))return 'import';
 if(/^(?:请|帮我|现在)?\s*(?:实现|开发|修复|编写|改造|测试|检查)(?:一个|一下|这个)?[^\n]{0,90}(?:记忆|memory)/iu.test(g))return null;
 if(/(?:记忆|memory)/iu.test(g)&&/(?:导出|导成|保存成|整理成|汇总成|备份到)/u.test(g)&&/^(?:请|帮我|现在|把|将|导出|整理|汇总|备份|保存|给我)[^\n]{0,600}$/u.test(g))return 'export';
 if(/(?:记忆|memory)/iu.test(g)&&/(?:导入|载入|插入|恢复)/u.test(g)&&/^(?:请|帮我|现在|把|将|从|导入|载入|插入|恢复)[^\n]{0,600}$/u.test(g))return 'import';
 return null;
}
class VoderChatViewProvider {
 constructor(app,{entryPlan=planEntry,referenceController=new ReferenceController(),projectMemory}={}){this.references=referenceController;this.entryPlan=entryPlan;this.app=app;this.views=new Set();this.pending=new Map();this.history=new Map();this.approvals=new Map();this.projectMemory=projectMemory??new ProjectConversationMemory(app.context.globalStorageUri.fsPath);}
 editorState(view){
  let editor;
  try{editor=this.references.api().window?.activeTextEditor;}catch{return this.post(view,{type:'reference_editor_state',hasEditor:false,hasSelection:false});}
  const uri=editor?.document?.uri;
  const hasEditor=!!(uri&&uri.scheme==='file'&&typeof uri.fsPath==='string'&&uri.fsPath);
  this.post(view,{type:'reference_editor_state',hasEditor,hasSelection:hasEditor&&!!editor.selection&&!editor.selection.isEmpty});
 }

 resolveWebviewView(view){
  this.views.add(view);
  view.webview.options={enableScripts:true,localResourceRoots:[]};
  view.webview.html=chatHtml(view.webview);
  const listeners=[];
  try{
   const api=this.references.api();
   for(const subscribe of [api.window?.onDidChangeActiveTextEditor,api.window?.onDidChangeTextEditorSelection]){
    if(typeof subscribe==='function'){
     const disposable=subscribe.call(api.window,()=>this.editorState(view));
     if(disposable?.dispose)listeners.push(disposable);
    }
   }
  }catch{} // Unit-test hosts may omit VS Code's event API.
  view.onDidDispose(()=>{for(const listener of listeners)listener.dispose();const approval=this.approvals.get(view);if(approval){this.approvals.delete(view);approval.resolve(false);}this.references.dispose(view);this.views.delete(view);this.pending.delete(view);this.history.delete(view);void this.projectMemory.end(view).catch(()=>{});});
  view.webview.onDidReceiveMessage(msg=>{void this.handle(view,msg).catch(e=>{this.post(view,{type:'error',code:String(e?.code||'VODER_CHAT_ERROR'),message:String(e?.message||'Chat error').slice(0,240)});});});
 }
 // A draft attachment belongs to one accepted task, not to every future turn.
 // Keep its immutable prepared referenceContext in the dispatched task; clear only
 // the host-owned draft after the SDK confirms task.submit was ACCEPTED.
 consumeAcceptedReferences(view,message){
  const pending=this.pending.get(view);
  if(!pending||pending.referencesConsumed||!pending.referenceIds?.length)return;
  if(typeof message.sessionId!=='string'||!message.sessionId||pending.sessionId&&pending.sessionId!==message.sessionId)return;
  const state=this.references.state(view);
  if(state.loading||state.store.version!==pending.referenceVersion)return;
  if(state.store.items.length!==pending.referenceIds.length||state.store.items.some((item,i)=>item.id!==pending.referenceIds[i]))return;
  state.store.clear();pending.referencesConsumed=true;
  this.post(view,{type:'references',items:[],loading:false});
 }
 post(view,message){
  if(!this.views.has(view))return;
  if(message?.type==='accepted'&&message.phase!=='plan')this.consumeAcceptedReferences(view,message);
  // Adapter validation is not provider acknowledgement. Strip transport metadata at the public message boundary.
  if(message?.type==='event'&&message.event?.type==='model.image_request'){
   message={...message,event:{type:'model.image_request'}};
  }
  void view.webview.postMessage(message);
 }
 requestNativeApproval(view,scope,{signal}={}){
  if(!this.views.has(view))return Promise.resolve(false);
  if(this.approvals.has(view))throw Error('Another approval is already pending');
  const requestId=crypto.randomUUID();
  if(signal?.aborted)return Promise.resolve(false);
  return new Promise(resolve=>{
   const finish=approved=>{signal?.removeEventListener('abort',abort);resolve(approved);};
   const abort=()=>{const p=this.approvals.get(view);if(p?.requestId===requestId){this.approvals.delete(view);this.post(view,{type:'authorization_resolved',requestId,approved:false});}finish(false);};
   // Authorization metadata stays host-side. Webview receives only exact relative labels and fixed business language.
   const inputFiles=Array.isArray(scope.files)?scope.files:String(scope.target||'').split('\n').slice(1);
   const fallback=String(scope.target||'');
   const candidates=inputFiles.length?inputFiles:(fallback?[fallback]:[]);
   const isSafeFile=f=>typeof f==='string'&&f.length>0&&f.length<=240&&!f.startsWith('/')&&
      !/[\\:\u0000-\u001f\u007f<>|?*\u202a-\u202e\u2066-\u2069]/u.test(f)&&
      f.split('/').every(x=>x&&x!=='.'&&x!=='..');
   const files=candidates.filter(isSafeFile);
   const visibleScope=files.length>0&&files.length===candidates.length&&files.length<=64?files:[];
   const project=scope.project===true,publication=scope.publication===true;
   const operation=publication?'Deliver the listed verified files to the selected project.':project?'Create a separate project containing the listed editable files.':'Allow changes to the listed project files. Preparation may also create private verification artifacts.';
   const checksSummary=publication?'Previously verified changes only; project files must still match their approved versions.':project?'Structural checks do not by themselves verify appearance or interaction.':'Configured checks are enforced independently; they do not by themselves certify appearance or interaction.';
   const publicScope={requestId,files:visibleScope,project,publication,operation,checksSummary};
   this.approvals.set(view,{requestId,resolve:finish,publicScope});signal?.addEventListener('abort',abort,{once:true});
   this.post(view,{type:'approval_requested',...publicScope});
  });
 }
 async memoryContext(view,root,goal){return require('./dialogue-model-tool.cjs').boundContext(await this.projectMemory.begin(view,root,goal));}
 async memoryFinish(view,root,goal,result){return this.projectMemory.finish(view,root,goal,result);}
 async memoryFail(view,root,code){return this.projectMemory.fail(view,root,code);}
 async openDeliveredDiff(view,folder,settled){
  if(settled?.status!=='DELIVERED'||settled.verified!==true||!this.views.has(view))return;
  const reply=(status,message)=>this.post(view,{type:'action_result',command:'diff',sessionId:settled.sessionId,status,message});
  reply('OPENING','Delivery complete · opening verified diff…');
  try{
   const result=await this.app.inspectDiff({folder,id:settled.sessionId});
   reply(result?.editor_documents?.length?'OPENED':'NOT_AVAILABLE',result?.editor_documents?.length?'Verified diff opened automatically.':String(result?.reason||'Verified diff is not available.'));
  }catch(e){reply('ERROR',String(e?.message||'Unable to open review').slice(0,280));}
 }
 async startNative(view,folder,msg,history){
  this.post(view,{type:'phase_started',phase:'native'});
    let profile;
    try{profile=await this.app.ensureNativeChat(folder,{goal:msg.goal,requestApproval:scope=>this.requestNativeApproval(view,scope)});}
    catch(e){this.post(view,scopeFeedback(folder.uri.fsPath,msg.goal,e));return;}
    this.post(view,{type:'info',text:'Native VGR authorized for EXISTING file: '+profile.target});
    const nativeContext=require('./dialogue-model-tool.cjs').boundContext(await this.projectMemory.begin(view,folder.uri.fsPath,msg.goal));
    try{
    const running=await this.app.chatRuns.start(folder,msg.goal,update=>{if(update.kind==='accepted'||update.kind==='created'){this.app._trackChatSession(update.sessionId,folder,profile.path);const current=this.pending.get(view);if(current)current.sessionId=update.sessionId;}if(update.kind==='settled'){const rows=this.history.get(view)||[];rows.push({goal:msg.goal,status:update.status});this.history.set(view,rows.slice(-3));}this.post(view,{...update,type:update.kind});},{history,...(msg.referenceContext?.length?{referenceContext:msg.referenceContext}:{}),conversationContext:nativeContext,nativeProfile:profile.path,requiredFiles:[profile.target],operatorTarget:profile.target});
    const current=this.pending.get(view);if(current)current.sessionId=running.sessionId;
    const settled=await running.completion;
    await this.projectMemory.finish(view,folder.uri.fsPath,msg.goal,settled);
    if(settled?.status==='DELIVERED'&&settled.verified===true)void this.openDeliveredDiff(view,folder,settled);
    }catch(e){await this.projectMemory.fail(view,folder.uri.fsPath,e?.code||'OUTCOME_UNKNOWN').catch(()=>{});throw e;}
 }
 async referenceCommand(view,type,payload={}){
  if(!this.views.has(view))return;
  if(this.pending.has(view))throw Object.assign(Error('Wait for the current task before changing references'),{code:'CHAT_BUSY'});
  if(!this.app.trusted())throw Object.assign(Error('Workspace Trust is required before reading references'),{code:'REFERENCE_TRUST'});
  const state=this.references.state(view);
  if(state.loading)throw Object.assign(Error('Reference preparation is already in progress'),{code:'REFERENCE_BUSY'});
  if(type==='reference_preview'){const item=state.store.snapshot().find(x=>x.id===payload.id);if(item)this.post(view,{type:'reference_preview',item});return;}
  state.loading=true;this.post(view,{type:'references_loading',loading:true});
  try{
   if(type==='reference_files')await this.references.files(view);
   else if(type==='reference_command_file')await this.references.files(view,[payload.uri]);
   else if(type==='reference_selection')await this.references.editor(view,{selection:true});
   else if(type==='reference_editor')await this.references.editor(view);
   else if(type==='reference_drop')await this.references.dropped(view,payload);
   else if(type==='reference_remove')state.store.remove(payload.id);

  }catch(e){const missingEditor=['reference_selection','reference_editor'].includes(type)&&['REFERENCE_SELECTION','REFERENCE_URI'].includes(e.code);this.post(view,{type:missingEditor?'reference_hint':'reference_error',code:String(e.code||'REFERENCE_ERROR'),message:String(e.message||'Unable to add reference').slice(0,300)});if(missingEditor)this.editorState(view);}
  finally{state.loading=false;if(this.views.has(view))this.post(view,{type:'references',items:state.store.list(),loading:false});}
 }
 async prepareReferences(view,folder,goal){
  const state=this.references.state(view);
  if(state.loading)throw Object.assign(Error('Wait for reference loading to finish'),{code:'REFERENCE_BUSY'});
  if(!state.store.items.length&&!/(?:^|[\s(])@/.test(goal))return [];
  const items=await this.references.prepare(view,folder.uri.fsPath,goal);
  if(!this.views.has(view))throw Object.assign(Error('Chat view closed before dispatch'),{code:'REFERENCE_CANCELLED'});
  this.post(view,{type:'references_included',items:items.map(({content,data,...x})=>({...x,bytes:content!==undefined?Buffer.byteLength(content):Buffer.from(data,'base64').length}))});
  return items;
 }
 async handle(view,msg){
  if(!msg||typeof msg!=='object'||Array.isArray(msg)||typeof msg.type!=='string')return;
  if(['reference_files','reference_selection','reference_editor','reference_drop','reference_remove','reference_preview'].includes(msg.type))return this.referenceCommand(view,msg.type,msg);
  if(msg.type==='reference_editor_state_request'){this.editorState(view);return;}
  if(msg.type==='ready'){this.editorState(view);this.post(view,{type:'references',items:this.references.state(view).store.list(),loading:this.references.state(view).loading});const folder=this.app.chatWorkspace?.();if(folder)this.post(view,{type:'workspace_selected',path:folder.uri.fsPath});const approval=this.approvals.get(view);if(approval)this.post(view,{type:'approval_requested',...approval.publicScope});return;}
  if(msg.type==='authorization_decision'){
   const pending=this.approvals.get(view);
   if(!pending||msg.requestId!==pending.requestId||!['approve','decline'].includes(msg.decision))return;
   this.approvals.delete(view);this.post(view,{type:'authorization_resolved',requestId:msg.requestId,approved:msg.decision==='approve'});pending.resolve(msg.decision==='approve');return;
  }
  if(msg.type==='clear'){if(this.pending.has(view))throw Object.assign(Error('Task active; cannot clear'),{code:'CHAT_BUSY'});const state=this.references.state(view);if(state.loading)throw Object.assign(Error('Wait for reference loading before clearing'),{code:'REFERENCE_BUSY'});state.store.clear();this.post(view,{type:'references',items:[],loading:false});this.history.delete(view);await this.projectMemory.end(view);return;}
  if(msg.type==='submit'){
   const dialogueIntent=/^\/memory\s+dialogue\s+(?:search\s+)?([^\n]{1,2000})$/iu.exec(String(msg.goal||'').trim());
   const policyIntent=dialogueIntent?{kind:'dialogue_search',query:dialogueIntent[1].trim()}:parseMemoryPolicyCommand(msg.goal);
   const memoryIntent=policyIntent||memoryCommand(msg.goal);
   if(memoryIntent){
    if(this.pending.has(view))throw Object.assign(Error('Task active'),{code:'CHAT_BUSY'});
    const folder=this.app.chatWorkspace?.()||await this.app.chooseFolder();
    if(!folder)throw Object.assign(Error('Open a project folder first'),{code:'MEMORY_PROJECT_REQUIRED'});
    if(policyIntent){
     if(policyIntent.kind==='policy_set'||policyIntent.kind==='policy_reset'){
      const p=policyIntent.kind==='policy_reset'?await this.projectMemory.resetPolicy(folder.uri.fsPath):await this.projectMemory.setPolicy(folder.uri.fsPath,policyIntent.policy);
      this.post(view,{type:'memory_policy_result',summary:'Project memory policy saved · mode='+p.mode+' · include='+p.include.join(' | ')+' · exclude='+p.exclude.join(' | ')+' · prefer='+p.prefer.join(' | ')});
     }else if(policyIntent.kind==='policy_show'){
      const p=await this.projectMemory.getPolicy(folder.uri.fsPath);
      this.post(view,{type:'memory_policy_result',summary:'Project memory policy: '+JSON.stringify(p)});
     }else if(policyIntent.kind==='dialogue_search'){
      const query=policyIntent.query;
      if(!query||Buffer.byteLength(query,'utf8')>2000)throw Object.assign(Error('Invalid archive search query'),{code:'MEMORY_SEARCH_QUERY'});
      const recent=/^(?:recent|最近|最近三轮|近期)(?:\s+page\s+\d{1,5})?$/iu.test(query);
      const pageMatch=/(?:^|\s)page\s+(\d{1,5})$/iu.exec(query);
      const page=pageMatch?Number(pageMatch[1]):1;
      const topic=pageMatch?query.slice(0,pageMatch.index).trim():query;
      const result=await this.projectMemory.search(folder.uri.fsPath,recent?'仅依据工作区对话档案回顾最近三轮':topic,{limit:5,archiveRecent:recent,archiveOnly:true});
      const rows=result.dialogue||[];
      const summary=require('./workspace-dialogue-memory.cjs').formatDialoguePage(rows,page,16000,recent?'/memory dialogue recent':'/memory dialogue search '+topic);
      this.post(view,{type:'memory_search_results',summary});
     }else if(policyIntent.kind==='search'){
      const result=await this.projectMemory.search(folder.uri.fsPath,policyIntent.query,{limit:5});
      const rows=[...(result.dialogue||[]).map(x=>({title:'[WORKSPACE DIALOGUE · '+(x.sourceFiles?.join(', ')||'来源文件未知')+'] '+x.goal,content:x.answer})),...result.turns.map(x=>({title:x.goal,content:x.answer})),...result.imported.map(x=>({title:x.title,content:x.content}))].slice(0,5);
      const summary=rows.length?rows.map((x,i)=>(i+1)+'. '+String(x.title||'').slice(0,160)+' — '+String(x.content||'').slice(0,800)).join('\n'):'No relevant project memory found.';
      this.post(view,{type:'memory_search_results',summary});
     }
    }else if(memoryIntent==='export'){
     const output=await this.projectMemory.exportProject(folder.uri.fsPath,msg.goal);
     const api=this.references.api();let displayPaths=output.paths||[output.path];
     // Save As makes the MD genuinely portable. Cancellation or an unavailable
     // VS Code FS API keeps the protected original export, never loses it.
     if(output.format==='md'&&api.workspace?.fs?.copy){
      try{
       if(displayPaths.length===1&&api.window?.showSaveDialog){
        const selected=await api.window.showSaveDialog({defaultUri:api.Uri.file(output.path),saveLabel:'Save portable project memory',filters:{Markdown:['md']}});
        if(selected){await api.workspace.fs.copy(api.Uri.file(output.path),selected,{overwrite:false});displayPaths=[selected.fsPath];}
       }else if(displayPaths.length>1&&api.window?.showOpenDialog){
        const chosen=await api.window.showOpenDialog({canSelectFiles:false,canSelectFolders:true,canSelectMany:false,openLabel:'Choose folder for portable memory parts'});
        if(chosen?.length){
         const path=require('node:path');const dest=chosen[0];const copied=[];
         for(const source of displayPaths){const to=api.Uri.joinPath(dest,path.basename(source));await api.workspace.fs.copy(api.Uri.file(source),to,{overwrite:false});copied.push(to.fsPath);}
         displayPaths=copied;
        }
       }
      }catch(e){void api.window?.showWarningMessage?.('Memory was exported to protected storage, but Save As did not complete: '+String(e?.message||e).slice(0,200));}
     }
     this.post(view,{type:'memory_exported',paths:displayPaths,entries:output.entries});
     try{await api.commands?.executeCommand?.('vscode.open',api.Uri.file(displayPaths[0]));}catch{}
     void api.window?.showInformationMessage?.('Voder memory exported · '+output.entries+' entries in '+displayPaths.length+' file(s).');
    }else{
     const api=this.references.api();const files=await api.window?.showOpenDialog?.({canSelectFiles:true,canSelectFolders:false,canSelectMany:false,openLabel:'Import Voder memory MD or JSON',filters:{'Voder memory':['md','json']}});
     if(!files?.length){this.post(view,{type:'memory_imported',entries:0});return;}
     const result=await this.projectMemory.importProject(folder.uri.fsPath,files[0].fsPath);
     this.post(view,{type:'memory_imported',entries:result.entries});
     void api.window?.showInformationMessage?.('Imported '+result.entries+' portable memory entries into this project.');
    }
    return;
   }
   msg={type:'submit',goal:msg.goal}; // Reference bytes may come only from the host-owned per-view store.
   if(this.references.state(view).loading){this.post(view,{type:'not_started',goal:msg.goal,message:'Wait for reference loading to finish'});return;}
   if(this.pending.has(view))throw Object.assign(Error('This chat view already has a task in flight'),{code:'CHAT_BUSY'});
   try{validateGoal(msg.goal);}catch(e){this.post(view,{...startNotice(e),goal:msg.goal});return;}
   if(!this.app.trusted()){this.post(view,{type:'not_started',status:'START_BLOCKED',message:'Workspace Trust is required. The task has not started.',goal:msg.goal});return;}
   // Reserve before folder-picker awaits to prevent double-click races.
   const draftState=this.references.state(view);
   this.pending.set(view,{folder:null,sessionId:null,referenceVersion:draftState.store.version,referenceIds:draftState.store.items.map(item=>item.id),referencesConsumed:false});
   try{
    if(!this.app.generalTasks&&(parseMultiGoal(msg.goal)||isProjectGoal(msg.goal))){const state=this.references.state(view);if(state.store.items.length||/(?:^|[\s(])@/.test(msg.goal)){const anchor=this.app.chatWorkspace?.()||await this.app.chooseFolder();if(!anchor)throw Object.assign(Error('Choose a workspace for references'),{code:'REFERENCE_SCOPE'});msg.referenceContext=await this.prepareReferences(view,anchor,msg.goal);}if(!this.views.has(view))return;if(parseMultiGoal(msg.goal))await this.app.multiProjects.start(this,view,msg);else await this.app.webProjects.start(this,view,msg);return;}
    let folder;try{folder=await this.app.chooseFolder();}catch(e){this.post(view,{type:'not_started',status:'START_BLOCKED',message:String(e?.message||'Unable to select a workspace'),goal:msg.goal});return;}if(!folder){this.post(view,{type:'not_started',status:'NOT_STARTED',message:'No workspace selected. The task has not started.',goal:msg.goal});return;}
    const pending=this.pending.get(view);if(pending)pending.folder=folder;
    this.post(view,{type:'workspace_selected',path:folder.uri.fsPath});
    msg.referenceContext=await this.prepareReferences(view,folder,msg.goal);
    if(!this.views.has(view))return;
    if(this.app.generalTasks&&this.entryPlan===planEntry){await this.app.generalTasks.start(this,view,msg,folder);return;}
    const history=[...(this.history.get(view)||[])];
    try{
     await runEntry({root:folder.uri.fsPath,goal:msg.goal,plan:this.entryPlan,
      inspect:async goal=>{
       this.post(view,{type:'phase_started',phase:'inspect'});
       const running=await this.app.chatRuns.inspect(folder,goal,update=>{
        if(['created','accepted'].includes(update.kind)){this.app._trackChatSession(update.sessionId,folder);const current=this.pending.get(view);if(current)current.sessionId=update.sessionId;}
        if(update.kind!=='settled')this.post(view,{...update,type:update.kind,phase:'inspect'});
       },{history,...(msg.referenceContext?.length?{referenceContext:msg.referenceContext}:{})});
       const current=this.pending.get(view);if(current)current.sessionId=running.sessionId;
       return running.completion;
      },
      showInspection:async(report,{continuing})=>{
       this.post(view,{type:'inspection_complete',answer:report.answer,scope:report.scope,continuing,status:report.status});
       if(!continuing){const rows=this.history.get(view)||[];rows.push({goal:msg.goal,status:report.status});this.history.set(view,rows.slice(-3));}
      },
      native:goal=>this.startNative(view,folder,{...msg,goal},history)
     });
    }catch(e){if(String(e?.code||'').startsWith('CHAT_SCOPE_')){this.post(view,scopeFeedback(folder.uri.fsPath,msg.goal,e));return;}throw e;}
   }catch(e){if(String(e?.code||'').startsWith('REFERENCE_')||['ENOENT','EACCES','ENOTDIR'].includes(e?.code)){this.post(view,{type:'reference_error',blocking:true,goal:msg.goal,code:String(e.code),message:String(e.message).slice(0,300)});return;}throw e;}finally{this.pending.delete(view);}
  }else if(msg.type==='cancel'){
   const p=this.pending.get(view);
   if(!p||typeof msg.sessionId!=='string'||p.sessionId!==msg.sessionId)throw Object.assign(Error('Unknown active session'),{code:'CHAT_SESSION_MISMATCH'});
   const service=this.app.generalTasks?.runs.activeSession(p.folder)===p.sessionId?this.app.generalTasks.runs:this.app.multiProjects?.runs.activeSession(p.folder)===p.sessionId?this.app.multiProjects.runs:this.app.webProjects?.runs.activeSession(p.folder)===p.sessionId?this.app.webProjects.runs:this.app.chatRuns;
   const ack=await service.cancel(p.folder,p.sessionId);
   this.post(view,{type:'cancel_ack',status:ack?.status});
  }else if(msg.type==='action'){
   if(!['diff','status','preview','open-project','tools','runtime'].includes(msg.command)||typeof msg.sessionId!=='string')return;
   if(['tools','runtime'].includes(msg.command)){try{if(msg.command==='tools')await this.app.toolkit.inspect(msg.sessionId);else await this.app.toolkit.runtime(msg.sessionId);this.post(view,{type:'action_result',command:msg.command,sessionId:msg.sessionId,status:'OPENED',message:'Inspection opened; observations do not establish visual acceptance.'});}catch(e){this.post(view,{type:'action_result',command:msg.command,sessionId:msg.sessionId,status:'ERROR',message:String(e.message).slice(0,280)});}return;}
   if(['preview','open-project'].includes(msg.command)){const reply=(status,message)=>this.post(view,{type:'action_result',command:msg.command,sessionId:msg.sessionId,status,message});try{const general=await this.app.generalTasks?.action(msg.sessionId,msg.command);const owner=this.app.toolkit?.source(msg.sessionId)||this.app.webProjects;const opened=general||await owner?.action(msg.sessionId,msg.command);reply(opened?'OPENED':'NOT_AVAILABLE',opened?'Opened.':'No generated project is attached to this session.');}catch(e){reply('ERROR',String(e.message).slice(0,280));}return;}
   const p=this.pending.get(view);
   // After settlement, caller must resolve the exact identity through the view's per-view history.
   const known=this.app.chatSessionFolder.get(msg.sessionId);
   const folder=p?.sessionId===msg.sessionId?p.folder:known;
   const reply=(status,message)=>this.post(view,{type:'action_result',command:msg.command,sessionId:msg.sessionId,status,message});
   if(!folder){reply('NOT_AVAILABLE','Session is no longer attached to this view. Open its saved session evidence.');return;}
   try{
    const args={folder,id:msg.sessionId};
    if(msg.command==='diff'){
     const result=await this.app.inspectDiff(args);
     if(result?.status==='VERIFIED_DELIVERED'&&result.editor_documents?.length)reply('OPENED','Verified diff opened in the editor.');
     else reply('NOT_AVAILABLE',String(result?.reason||'No verified diff is available for this session.'));
    }else{await this.app.showStatus(args);reply('OPENED','Session evidence opened in the Voder output panel.');}
   }catch(e){reply('ERROR',String(e?.message||'Unable to open review').slice(0,280));}
  }
 }
}
module.exports={VoderChatViewProvider,chatHtml};
