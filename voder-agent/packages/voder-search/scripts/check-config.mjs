import {normalizeSearchConfig,buildSearchJob,checkAdapterPath,SEARCH_PRESETS} from '../src/search-config.mjs';
import {readFileSync} from 'node:fs';
if(process.argv.includes('--presets')){console.log(SEARCH_PRESETS.join(', '));process.exit(0);}
try{
 const bytes=readFileSync(0);if(bytes.length>128*1024)throw Object.assign(new Error(),{code:'WEB_CONFIG_INVALID'});
 const cfg=normalizeSearchConfig(JSON.parse(bytes.toString('utf8')));
 if(!cfg.providers.length){if(process.argv.includes('--allow-unconfigured')){console.log('CONFIG=UNCONFIGURED; PROVIDERS=0; NETWORK_REQUESTS=0');process.exit(0);}throw Object.assign(new Error(),{code:'WEB_CONFIG_INVALID'});}
 cfg.providers.forEach(checkAdapterPath);
 const ready=cfg.providers.filter(p=>p.ready);
 if(!ready.length){if(process.argv.includes('--allow-unconfigured')){console.log('CONFIG=CREDENTIALS_PENDING; NETWORK_REQUESTS=0');process.exit(0);}throw Object.assign(new Error(),{code:'WEB_CREDENTIAL_MISSING'});}
 // Compile templates without issuing any network request. Plugins are checked during real search.
 for(const p of ready.filter(p=>!p.adapter))await buildSearchJob(p,'configuration check',1000);
 console.log('CONFIG=PASS; PROVIDERS='+cfg.providers.length+'; READY='+ready.length+'; NETWORK_REQUESTS=0');
}catch(e){console.log('CONFIG_ERROR='+(['WEB_CREDENTIAL_MISSING','WEB_CONFIG_INVALID'].includes(e?.code)?e.code:'WEB_CONFIG_INVALID'));process.exitCode=1;}
