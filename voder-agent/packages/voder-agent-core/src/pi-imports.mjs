/** Resolve public ESM exports using this Node executable, never require conditions.
 * The resolver child evaluates only the fixed string below; it does not import packages,
 * use a shell, load NODE_OPTIONS, or receive service credentials.
 */
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
const run=promisify(execFile);
const resolver=`const parent=process.argv[1];
const entry=import.meta.resolve('@earendil-works/pi-coding-agent',parent);
const aiEntry=import.meta.resolve('@earendil-works/pi-ai',entry);
console.log(JSON.stringify({entry,aiEntry}));`;
export async function resolvePiImports(parent=import.meta.url){
 const origin=new URL(parent);if(origin.protocol!=='file:')throw new Error('PI_IMPORT_ORIGIN_MUST_BE_FILE');
 const {stdout}=await run(process.execPath,['--experimental-import-meta-resolve','--input-type=module','--eval',resolver,origin.href],{env:{},timeout:10000,maxBuffer:16384,windowsHide:true});
 const result=JSON.parse(stdout);
 for(const name of ['entry','aiEntry'])if(typeof result[name]!=='string'||new URL(result[name]).protocol!=='file:')throw new Error('PI_IMPORT_ENTRY_MUST_BE_FILE');
 return Object.freeze(result);
}
