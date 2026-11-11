# 模块08实际P0阻断点与最小修复路线

以下是**源码证实的事实或实测受限**，不是推断已完成的功能。

| 门禁 | 源码/能力证据 | 现状 | 必须在正式主干修复与实测的工作 |
| --- | --- | --- | --- |
| A09 | `packages/voder-sdk/src/control-plane.mjs::getDiff()` | 始终`NOT_AVAILABLE` | 以真实session/run/action身份读取已验证前后像；校验沙箱、路径、受保护回执和原生Rust commit；不得直接输出全仓库Git diff或LLM文本；CLI和SDK共享该只读契约 |
| A09 | `packages/voder-cli/bin/voder-cli.mjs`之`diff` | 显式占位 | 以同一个SDK验证接口实现CLI差异展示/审批/历史重连；必须对未验证、失败、取消和无归属操作返回明确不可用原因 |
| A10 | `packages/voder-vscode/src/extension.cjs`＋扩展README | 本地RPC/会话入口有离线回归，但真实会话级diff不可用 | 以SDK返回的授权diff显示真实VS Code审查、审批与诊断；验证CLI与扩展同一session ID、真实Git工作树和VGR裁决，覆盖重启及无效权威拒绝 |
| A10 | `packages/voder-vscode/package.json` | 配置说明要求手工设置`voder.agentRoot`与网关路径 | 非开发者用户安装体验另做原生运行时发现、可信提示及权限隔离；发现失败停问，不伪造native配置或放宽安全策略 |
| A12 | SDK`safeSummary`及事件流 | 目前没有验证过完整任务级用量回执与费用统计 | 用Provider原始用量/模型价格记录、真实计费来源、跨会话账本核对；未知费用记N/A，禁止计算零元或虚构Token |
| A05/A06 | 原生授权工具在`TrustedVgrGatewayTransport`中限定`kind:'patch'`且命中唯一`targetPath` | 可以验收单文件代码修复，但无法证明任意复杂多文件Repo任务 | 先做单文件完整闭合，然后在不破坏Rust VGR边界的前提下扩展版本化多文件原子提交，重做沙箱+回滚/并发/验收；避免第二套提交权威 |

本版本不是上述功能修复包，**不会**主动覆盖已经冻结的SDK、CLI、Rust或扩展。原因是这些能力直接触及原生授权、提交身份与可恢复写入，不能通过未验证的临时补丁一笔带过；补丁应作为模块08内部增量变更，用独立证据评审后再导入主干。本次包提供其必须满足的不可篡改验收外壳和测试任务基础，消除“局部PASS即生产级”这一错误路径。
