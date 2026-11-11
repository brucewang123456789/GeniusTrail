# Voder 模型接入：两个独立可安装插件

依据原始源码 `packages/voder-provider-cloud` 与 `packages/voder-provider-open` 的插件划分，扩展到 Pi v1.1.0 的 `ModelRuntime.registerProvider` 机制。不是重新实现模型推理循环，也不改动上游引擎与 Rust VGR。使用两个不同模式：API=`voder-provider-cloud`，自部署权重服务=`voder-provider-open`，只能安装其中一个；管理员和普通开发者走相同入口。

## 单次安装（在 AutoDL 运行）

原始 ZIP 留在 `/root/autodl-tmp/repair/`，解压后：

`python3 .../scripts/install.py /root/autodl-tmp/voder-agent --mode api`

或：

`python3 .../scripts/install.py /root/autodl-tmp/voder-agent --mode weights`

**两种安装命令只选择一种**；切换模式必须追加 `--switch`，系统备份旧模块和配置，安装失败自动恢复；不得在同一个运行会话中切换。源码安装在主干 `plugins/voder-provider-*`，并非 `repair`。

## 填写模型编号/API Key并运行

由入口脚本交互询问模型ID及API Key，不回显Key、不把Key写入磁盘、不需要Pi登录：

`bash /root/autodl-tmp/voder-agent/scripts/run_with_model.sh -- node /root/autodl-tmp/repair/<验收目录>/run_real_model_vgr.mjs`

若运行器为 `run.sh`，则在 `--` 后使用 `bash <run.sh>`，并保留验收要求的 `VODER_MODAL_CLI`。可以通过 VODER_MODEL_ID 和 VODER_MODEL_API_KEY 环境变量为系统服务注入密钥，禁止把密钥写进源码、日志、命令行参数或Git仓库。

**云API模式**：根据明确的模型ID前缀自动识别 DeepSeek、OpenAI、Anthropic、Google Gemini、Kimi、GLM、Grok、Qwen；OpenRouter和其他自定义网关需额外 `VODER_MODEL_VENDOR=openrouter/custom`，自定义网关还需 `VODER_MODEL_ENDPOINT=https://.../v1`。模型是否真实存在、协议是否支持工具调用必须由上游API实测，插件不能凭模型名保证任何模型都可用；例如必须输入DeepSeek控制台实际提供的模型ID，不能猜测发行名就是接口ID。

**开源权重模式**：接入已启动且可访问的vLLM/SGLang/Ollama/llama.cpp OpenAI兼容服务；默认vLLM在127.0.0.1:8000/v1；可通过 VODER_OPEN_RUNTIME、VODER_MODEL_ENDPOINT 定制。该插件不会下载或启动权重，也不能保证8GB显存运行任意模型；服务器需自行启动并提供兼容Chat Completions及tool calling的推理接口。该模式也要求提供模型ID与接入密钥（没有认证的本地推理服务可采用专门的本地服务token，不可把它误认为云API Key）。

## 质量与边界

- `ModelRuntime` 每次Session独立创建，模型身份/凭据不写入Pi配置文件；只允许一个选定provider和模型进入SDK Session；现有Voder工具白名单/VGR可信执行不变。
- 默认上下文32768、最大输出4096只是安全的本地占位配置，未知模型真实限制必须在部署时配置 VODER_MODEL_CONTEXT_WINDOW / VODER_MODEL_MAX_TOKENS；计费元数据为占位零，不等于实际零费用。
- 除非实际完成模型响应、工具选择、VGR独立授权、Modal验证、提交/回滚和恢复测试，不标称生产级。
- MIT上游版权和许可证继续保留。第三方API兼容性取决于各厂商版本，不存在只用模型编号和Key就能自动识别全球全部私有协议的可靠方法。
