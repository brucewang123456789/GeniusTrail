# Voder SDK＋CLI 模块——接口、部署与验收范围（2026-10-13）

**适用范围：** 当前 Voder Agent 蓝图的「第六项：voder-sdk＋voder-cli」，基于 2026-10-13 用户提供源码及已安装的个性化记忆更新版。保持原生 Pi 的真实会话/模型机制，**任何工具写入仍须 VGR 验证授权**。

## 实现（有离线实测）

- 沿用原 SDK `VoderSDK`：`createSession`、`submitTask`、`getEvents`、`cancel`、个性化记忆显式导入/导出和候选审批。新增 `subscribeEvents(sessionId, callback, {after})`，原有 Pi 工具白名单不变，`AgentKernel` 为事件观察提供只读回调。
- 新增 `VoderControlPlane` 与 `VoderJournal`：统一 SDK 层与 CLI 的会话、状态、审计事件、取消、只读历史和受限导出。日志写入私有 `0700` 目录，文件 `0600`，SHA256用于损坏检测，**不是签名，也不能替代 Rust VGR 交付证据**。
- CLI `voder run`：创建实际 SDK 会话并调用 Pi 模型，用户必须选择 `--gateway-config` 真实 VGR 路径或显式 `--draft`。仅原生 commit 被确认时，退出码才为 0；草稿退出码 3，失败退出码 2（启动配置错误退出码 1）。
- CLI `voder rpc`：JSON-RPC 2.0（每行一个 JSON），供 VS Code、JetBrains、Neovim 或其他进程通过 stdin/stdout 使用，**不依赖 IDE 特定插件**。`session.create`、`task.submit`、`task.wait`、`session.status`、`session.events`、`session.subscribe`/`session.unsubscribe`、`session.cancel`、`session.resume`（只读）、`session.export`、`session.diff`（未接通则明确拒绝）、`memory.proposals`、`memory.approve`。
- CLI `voder status|events|resume|diff|export`：跨进程读取持久化会话元数据，不会偷偷恢复未确认的侧效应。原本 `voder model|plugin|doctor|daemon|memory` 依旧交给既有命令处理。
- 缺文件阻断、取消、未授权动作的真实与模拟边界遵循原 kernel / broker，不通过 CLI 另做工具执行通路。

## 使用（云端根目录更新后）

**环境**：内置 Pi 1.1.0 声明 Node >=22.19.0；旧镜像如 Node 22.16.0，离线测试或能运行，但真实 Pi API 是否兼容须在该机器实测。不要把警告当通过。

```bash
cd /root/autodl-tmp
bash voder-agent/scripts/test_sdk_cli.sh
bash voder-agent/scripts/run_with_model.sh -- ./bin/voder run --goal '请检查当前项目的目录结构' --draft
```

`--draft` 为 **未验证草稿**，从不授予写入权限、不会冒充 VGR commit；启动需要原项目已选定的模型插件和有效 API key。若要真实 VGR 提交，管理员应先建立 `0600` 的私有 gateway 配置文件，包含现有 `TrustedVgrGatewayTransport` 的必要原生二进制/政策/目录路径，然后省略 `--draft`，改用 `--gateway-config /绝对路径/config.json`；最终需要完整的 Rust 证据回放与目标任务验收。

适用于其他编辑器的协议入口：

```bash
bash voder-agent/scripts/run_with_model.sh -- ./bin/voder rpc --draft
```

向 stdin 逐行发送 JSON-RPC；stdout 只返回 JSON（事件订阅通过 `session.event` 通知）。例如：

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"session.create","params":{"memoryFiles":[],"enableMemoryProposals":false}}
```

由 session.create 返回 ID 后，才用 `task.submit` 发送 `{ "sessionId": "...", "task": {"goal":"检查项目"} }`，用 `task.wait` 等待，或通过订阅/状态查询观察。同一进程可取消；服务退出后的会话仅只读，绝不重放可能有副作用的工具调用。

注意日志仅保存 **脱敏事件元数据和最小结果**，不会保存 API key、模型生成正文或源代码；导出文件具有用户私有信息属性，必须自行控制分享范围。

## 尚未完成，禁止宣称为 PASS

- **真实模型 + Pi SDK E2E（本模块）**：本地离线测试使用 mock host，无真实用户 API 和 VS Code 集成运行环境。先前第六模块记忆链路的真实模型验收，不等于这里的 CLI 调用和VGR交付完成。
- **VGR 正式任务级审批、diff 及原生断点恢复**：本版 `session.resume` 是只读恢复历史，不能接管或重新执行旧任务；`session.diff` 明确返回 `NOT_AVAILABLE`，直到 Rust 提供任务级、带前后快照的证据。现有 gateway 可以传递动作，但不能把 SDK 自身的摘要当作 native-verified commit。
- **跨进程即时取消**：一个 RPC 进程内的活动会话支持取消；已退出进程不支持通过状态读取进程发送取消信号。进程意外终止时必须核对原生VGR日志再决定下一次动作。
- **生产级可靠性和多编辑器插件**：VS Code/JetBrains/Neovim插件尚未实现；压测、网络异常、长任务、并发任务、真实仓库修复及蓝图 A01—A12 仍须全链路验收。

**结论**：新增SDK/CLI协议与审计骨架及离线合约回归，可作为后续IDE插件和原生VGR进一步集成的基础，**不是宣布第六项或整个Voder达到生产级冻结**。
