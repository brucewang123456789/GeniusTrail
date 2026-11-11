# 用户自行配置搜索服务

本模块不绑定任何搜索厂商。通用 HTTP 搜索 API 可以配置请求模板与返回字段映射；不同签名、XML/文本格式等可安装本地适配器。不是只换一个 Key 就自动识别所有厂商协议，也不承诺账号权限、索引覆盖或网络可达性。

## 常用方式

安装完成后，在云端 Linux 终端执行：

```bash
python3 -B -u /root/autodl-tmp/voder-agent/packages/voder-search/scripts/configure.py
```

选择预设或 `custom`，按提示配置。API Key 输入隐藏；配置校验通过后，仅原子替换本模块 `search-config.json`，权限 0600，旧配置备份仍在同一模块。这个命令不重启应用、不调用模型，也不发网络请求。API 实网验证请另行执行下面的命令。已经启用的默认 Rust 搜索实例每次读取配置，切换配置会改变缓存身份；已经运行的旧版代码仍需重新加载。

已有 JSON 文件也可以导入或只检查：

```bash
python3 -B -u /root/autodl-tmp/voder-agent/packages/voder-search/scripts/configure.py --config /absolute/path/my-search.json
python3 -B -u /root/autodl-tmp/voder-agent/packages/voder-search/scripts/configure.py --check
```

## 预设不是厂商白名单

预设只是替用户填写协议默认值，可覆盖 endpoint、headers、params、body、response 等字段。

| preset | 默认方法 | 结果数组 | 凭据默认环境变量 |
|---|---|---|---|
| brave | GET | `/web/results` | `VODER_SEARCH_BRAVE_API_KEY` |
| tavily | POST | `/results` | `VODER_SEARCH_TAVILY_API_KEY` |
| serper | POST | `/organic` | `VODER_SEARCH_SERPER_API_KEY` |
| serpapi | GET | `/organic_results` | `VODER_SEARCH_SERPAPI_API_KEY` |
| bigmodel | POST | `/search_result` | `VODER_SEARCH_BIGMODEL_API_KEY` |
| deepseek | POST | `/results` | `VODER_SEARCH_DEEPSEEK_API_KEY` |
| searxng | GET | `/results` | 无，必须提供自己的 endpoint |

数组顺序就是优先顺序，最多 8 个服务。它们仍共享现有任务的请求预算：默认一次任务最多 3 个外部操作，并为正文抓取保留一个名额，不会因为配置了 8 家就无计量调用 8 家。冷却按服务配置和凭据隔离，不把一个供应商的故障扩散为全局禁用。

## 自定义 API 示例

下面演示一个不存在于预设列表中的 API，无需改 Voder 核心代码：

```json
{
  "version": 1,
  "providers": [{
    "id": "my-search",
    "endpoint": "https://my-search.example.com/v2/search",
    "method": "POST",
    "headers": {"Authorization": "Bearer {{secret:token}}"},
    "credentials": {"token": {"env": "MY_SEARCH_TOKEN"}},
    "body": {"input": {"query": "{{query}}"}, "count": 8},
    "response": {
      "items": "/data/results",
      "url": "/source/url",
      "title": "/title",
      "excerpt": ["/summary", "/snippet"]
    }
  }]
}
```

将示例域名改为自己真实的搜索 API。`credentials` 可用 `env` 从 Voder 运行进程环境读取，也可以用 `value` 保存本地值；两者二选一。终端临时 export 不保证被已经运行的 VS Code 扩展继承。配置、同目录备份均可能含 Key，请勿公开。

`{{query}}` 仅填到参数值/请求体中，不能控制 API 地址或请求头；替换只做一遍，查询文本内的模板符号不会再展开为凭据。`{{secret:name}}` 引用配置中的指定凭据，不在错误、网络回执中打印。

- GET 使用 `params`；POST 支持 JSON `body`、`body_format: "form"` 的扁平表单，以及 `body_format: "text"` 的原始字符串。
- `headers` 可配置 Bearer、Basic（将已编码值作为 secret）、任意 API Key 头等；查询参数、JSON 和表单内认证也可用 secret 模板。不允许伪造 Host、Content-Length 等传输头。
- 返回映射使用 JSON Pointer：`/data/results`、`/hits/hits`、数组索引如 `/pages/0/results`，转义 `/` 用 `~1`，转义 `~` 用 `~0`。不是任意 JS/JSONPath 表达式，不执行表达式。
- `title`、`excerpt` 可省略，或给最多 4 个备用 Pointer；URL 必须能映射到字符串。数组缺失、URL 字段错型会报告错误，不伪装为成功空结果。
- API endpoint 固定由本机配置控制，支持公网 HTTPS 自定义端口。DNS 校验、地址固定、证书验证和字节/耗时限制照常执行。
- 若 API 不需要认证，可以完全省略 credentials，不存在强制智谱/DeepSeek Key 的前置条件。

## 本机搜索服务

可参考 `search-config.searxng.example.json`。仅对显式配置的本机搜索 API，允许 `http://127.0.0.1:PORT/...` 或 `http://[::1]:PORT/...` 并设置 `allow_loopback: true`。服务需要自己实际运行，并启用 JSON 搜索返回；本包不安装或启动搜索引擎。

此权限只授予配置中的搜索 API，绝不授予搜索结果或网页正文抓取。公网普通 HTTP、私网 IP、云元数据地址不能借此放行；搜索 API 也不跟随重定向。HTTPS 自签名证书不会自动信任。

## 特殊协议或签名

在本模块 `adapters/` 放入自己的受信任 `.mjs` 文件，配置：

```json
"adapter": "adapters/my-signer.mjs"
```

接口：

```javascript
export function buildRequest({query, request, credentials}) {
  // Add a signature to request.headers/params/body; preserve request.url.
  return request;
}
export function parseResponse(payload) {
  // Convert the provider's response to the common candidate array.
  return [{url: "https://source.example.org/article", title: "Title", excerpt: "Snippet"}];
}
```

适配器不得改变固定的 API endpoint，也不得扩大本机 API 权限。它负责签名和格式转换，网络仍应由 Rust 请求内核执行，不应自行发请求、读写文件或执行命令。适配器是用户安装的本地可信代码，不加载远程代码或模型给出的路径；不允许父目录遍历和符号链接跳转。异步转换有截止时间；它不是任意恶意插件的安全沙箱。

JSON 返回直接传对象。其他文本格式可以设 `response_format: "text"`，支持 XML、纯文本、CSV 等响应并交给适配器解析；原始请求体使用 `body_format: "text"`，需要时自行设置 Content-Type。参考 `adapters/example-text.mjs` 的 TSV 转换实现。OAuth 获取/刷新、厂商 SDK 专有二进制协议等不属于仅填 Key 的自动兼容范围，需另行实现受信任适配器或 HTTP 网关。

## 实网验证与边界

```bash
node /root/autodl-tmp/voder-agent/packages/voder-search/scripts/verify-live.mjs /root/autodl-tmp/repair/search-live.json --query 'site:doc.rust-lang.org/book ownership'
```

最多两次搜索和一次本地正文抓取，模型调用为零。成功表示当前配置完成候选发现和一次正文获取，不等于问题答案已经核实，更不等于长期 SLA。

不传 `--query` 时保留此前诺贝尔奖问题的官网路径测试；`--source URL` 可指定本次发现结果中应抓取的那一条来源。报告会含候选与网络回执，文件权限 0600。

正文继续由 Voder 本机获取，不调用模型厂商 Fetch API。搜索 API 成功不保证来源站点可达。当前远程通用传输支持直连；适用的显式 HTTPS/ALL 代理仍明确报 `WEB_PROXY_UNSUPPORTED`，没有偷偷改成绕过代理或交给代理解析目标地址。本次未扩展代理能力，也未修改其他 Voder 模块。

旧 12 版配置仍可读取；SDK 显式注入的 provider 或旧 keylessEngine 接口仍按原有优先级工作，默认 Rust 通用搜索才使用本配置。

## 14 版：独立、可配置的可信 DNS

13 版云端证据显示，`api.search.brave.com` 的系统解析返回了与境外可信解析不一致的地址，其中 IPv6 属于 Meta 地址空间。14 版默认不使用这条系统解析路径，而使用 RFC 8484 标准 DNS-over-HTTPS，并将得到的公网地址固定到实际连接；原始域名继续用于 SNI 和证书校验。不改 `/etc/hosts`、`/etc/resolv.conf`、系统网络、其他模块或任何模型配置。

默认解析器为 Google、Cloudflare、Quad9、AliDNS，采用各自公开服务的 bootstrap 地址建立并验证 HTTPS。这里固定的是 DNS 服务的公开引导地址，绝不固定 Brave、其他搜索 API 或来源站点今天的 IP。查询只包含域名和 A/AAAA 类型，不把搜索词、API Key、请求头或正文发送给 DNS 服务。

可在顶层配置默认网络策略，也可在某一 provider 中用同样的 `network` 字段覆盖：

```json
"network": {
  "dns": {
    "mode": "secure",
    "address_family": "auto",
    "timeout_ms": 2500,
    "cache_ttl_secs": 300,
    "require_ad": false,
    "resolvers": [
      {
        "id": "my-doh",
        "url": "https://my-dns.example.com/dns-query",
        "bootstrap_ips": ["填写自己解析器的真实公网 IP"],
        "method": "POST"
      }
    ]
  },
  "connect_timeout_ms": 3000
}
```

示例 bootstrap 字符串是说明占位符，必须替换为真实公网 IP；不要原样导入。也可以省略 `resolvers` 使用默认列表，或省略 bootstrap_ips 用系统解析解析器自身的域名（仍校验证书和固定已检查的地址）。最多 4 个解析器，每个最多 4 个引导地址。支持标准 RFC 8484 GET / POST，不依赖某家的私有 JSON DNS 协议。

`address_family` 可选 auto / ipv4 / ipv6，只作用于本模块的当前请求，不全局禁用 IPv6。`mode: "system"` 是用户明确选择的兼容模式，仍校验整组公网地址并固定连接；安全解析失败不会自动降级到系统解析。若使用本地可信 DoH 服务，可以配置 `http://127.0.0.1:PORT/dns-query` 或 IPv6 回环并显式设置 `allow_loopback: true`，不填 bootstrap_ips；这个特许不授予来源网页。

DNS 应答校验问题名、类型、CNAME 链、截断、状态码和地址范围。一个解析器的相关应答中出现私网/保留地址，会使该解析器整组结果失效，其他健康解析器可独立继续。默认优先采用解析器报告 AD 已验证的应答；存在这类应答时，不混入其他未验证应答。AD 是来自已配置递归解析器的声明，不是本模块自行完成的全链 DNSSEC 验证；`require_ad: true` 会拒绝不带 AD 的结果，不适用于所有未签名域名。

缓存按域名、端口和网络策略隔离，受 DNS/CNAME 最短 TTL 与用户上限共同约束；最大 256 项，纯内存，不写系统或项目文件。网络策略切换立即改变缓存/冷却身份。连接缓存失效时，仅当能确认请求尚未发出，才在原截止时间内清缓存、重新解析后重连；对可能已经发出的搜索 POST 或读响应超时，不自动重放。

网络回执记录所用解析器、引导地址、A/AAAA 应答、TTL、AD 标志、选中地址、连接错误阶段和可取得的系统 errno。这里记录的是客户端可观察事件，不等于宿主机抓包。`tls_validation: required` 不代表握手已完成，只有取得 HTTPS 响应才标记 `tls_validated: true`。DNS 请求不消耗搜索 API 计费请求额度，但会产生独立的 DNS 网络请求；并行数、字节数与耗时均受限制。

这修复了依赖异常系统解析的设计缺口，但不会让防火墙阻断、真实路由故障或上游停机凭空消失。若所有已配置安全解析器不可达，会返回 `WEB_SECURE_DNS_UNAVAILABLE` 并保留回执，而不是伪装成搜索空结果或强行关闭证书验证。显式代理的既有限制仍保留，需要代理的环境应部署已验证的安全连接器；不做自动绕过。

完整网络记录保留在 provider 的有界诊断状态和实网验证报告中；模型工具只返回网络摘要，避免 DNS 回执挤占既有工具输出预算。
