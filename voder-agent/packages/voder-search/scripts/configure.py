#!/usr/bin/env python3
"""Configure only this retrieval module. Keys are hidden and never printed."""
import argparse,datetime,getpass,json,os,pathlib,shutil,subprocess,tempfile,urllib.parse
PRESETS=('brave','tavily','serper','serpapi','bigmodel','deepseek','searxng')
def ask(prompt,default=''):
 value=input(prompt+(' ['+default+']'if default else '')+': ').strip();return value or default

def collect(current):
 providers=[]
 print('选择搜索服务或 custom；不再绑定任何指定供应商。Key 输入隐藏。',flush=True)
 while len(providers)<8:
  name=ask('Provider ('+'/'.join(PRESETS)+'/custom'+('；回车保留已有配置'if not providers and current.get('providers')else '')+')').lower()
  if not name and not providers and current.get('providers'):return current
  if name not in PRESETS+('custom',):raise RuntimeError('UNKNOWN_PRESET; choose custom for any other API')
  provider={'id':ask('配置名称','search-'+str(len(providers)+1))}
  if name!='custom':provider['preset']=name
  if name in ('custom','searxng'):
   endpoint=ask('搜索 API 地址');provider['endpoint']=endpoint
   parsed=urllib.parse.urlparse(endpoint)
   if parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','::1'):provider['allow_loopback']=True
  if name=='custom':
   method=ask('请求方法 GET/POST','GET').upper()
   if method not in ('GET','POST'):raise RuntimeError('METHOD_MUST_BE_GET_OR_POST')
   provider['method']=method
   field=ask('查询参数名称（嵌套模板请使用 --config JSON 文件）','q'if method=='GET'else'query')
   target='params'if method=='GET'else'body';provider[target]={field:'{{query}}'}
   auth=ask('认证 none/bearer/header/query/body','bearer').lower()
   if auth not in ('none','bearer','header','query','body'):raise RuntimeError('UNKNOWN_AUTH_MODE')
   if auth!='none':
    key=getpass.getpass('搜索 API Key（隐藏）: ').strip()
    provider['credentials']={'api_key':{'value':key}}
    ref='{{secret:api_key}}'
    if auth=='bearer':provider['headers']={'Authorization':'Bearer '+ref}
    elif auth=='header':provider['headers']={ask('认证头名称','X-API-Key'):ref}
    elif auth=='query':provider.setdefault('params',{})[ask('认证参数名称','api_key')]=ref
    elif method=='POST':provider.setdefault('body',{})[ask('认证字段名称','api_key')]=ref
    else:raise RuntimeError('BODY_AUTH_REQUIRES_POST')
   provider['response']={'items':ask('结果数组 JSON Pointer','/results'),'url':ask('每条结果 URL Pointer','/url'),'title':ask('标题 Pointer','/title'),'excerpt':ask('摘要 Pointer','/snippet')}
  elif name!='searxng':
   key=getpass.getpass(name+' search API Key（隐藏；回车使用对应环境变量）: ').strip()
   if key:provider['credentials']={'api_key':{'value':key}}
  providers.append(provider)
  if ask('再添加一个备用搜索服务？y/N','N').lower()!='y':break
 return {'version':1,'providers':providers,**({k:current[k] for k in ('network','budget') if k in current})}

def node_binary(module):
 root=module.parents[2]
 for candidate in [os.environ.get('VODER_NODE_BINARY'),str(root/'.voder-tools/node/bin/node'),shutil.which('node')]:
  if candidate and pathlib.Path(candidate).is_file():
   try:
    version=subprocess.check_output([candidate,'--version'],text=True).strip()
    if tuple(map(int,version.lstrip('v').split('.')))>=(22,19,0):return candidate
   except (ValueError,subprocess.SubprocessError):pass
 raise RuntimeError('NODE_22_19_REQUIRED')

def main():
 os.umask(0o077);module=pathlib.Path(__file__).resolve().parent.parent
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=pathlib.Path);parser.add_argument('--check',action='store_true');args=parser.parse_args()
 path=module/'search-config.json'
 if path.is_symlink():raise RuntimeError('SYMLINK_CONFIG_DENIED')
 current=json.loads(path.read_text())if path.exists()else {'version':1,'providers':[]}
 value=json.loads(args.config.read_text())if args.config else current if args.check else collect(current)
 data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
 result=subprocess.run([node_binary(module),str(module/'scripts/check-config.mjs')],input=data,capture_output=True)
 print(result.stdout.decode(),end='',flush=True)
 if result.returncode:raise RuntimeError('CONFIG_VALIDATION_FAILED; old config unchanged')
 if not args.check:
  if path.exists():
   backup=module/('search-config.backup-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json');backup.write_bytes(path.read_bytes());backup.chmod(0o600)
  fd,tmp=tempfile.mkstemp(prefix='.search-config-',dir=module)
  try:
   with os.fdopen(fd,'wb')as f:f.write(data);f.flush();os.fsync(f.fileno())
   os.chmod(tmp,0o600);os.replace(tmp,path)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
  print('SEARCH_CONFIG_SAVED; live API verification not run',flush=True)
if __name__=='__main__':
 try:main()
 except (RuntimeError,ValueError,OSError,subprocess.SubprocessError):print('SEARCH_CONFIG_FAILED; check settings; previous configuration preserved',flush=True);raise SystemExit(1)
 finally:print('CURRENT_TERMINAL_REMAINS_OPEN=True',flush=True)
