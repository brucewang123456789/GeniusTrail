# Voder 第六模块 · 可移植工程记忆 · 2026-10-13 验收说明

**模块级判定：IMPLEMENTED_TESTED_OFFLINE。整机/真实模型/真实Pi/VS Code/VGR端到端：UNVERIFIED。不能宣布生产级冻结。**

## 改动范围（最小集成、保留上游）

- 新增 `voder-agent/packages/voder-memory/`：记忆提案、人工筛选、文件导出/验证、会话导入与注入、无模型命令行。
- 修改 `voder-agent/packages/voder-agent-core/src/kernel.mjs`：在执行模型调用前校验用户选定的记忆文件，注入到**用户级上下文**；启用时暴露只生成内存提案的 `voder_memory_propose`；每任务清除过期提案。
- 修改 `voder-agent/packages/voder-agent-core/src/pi-host.mjs`：仅扩展明确工具白名单，仍严格限制上游实际可调用工具；**Pi原生会话、compaction、SDK生命周期与系统指令隔离不改**。
- 修改 `voder-agent/packages/voder-sdk/src/index.mjs`：`createSession({memoryFiles,enableMemoryProposals})`、`getMemoryProposals`、`approveMemoryProposal`、`exportMemory`、`inspectMemoryFile`、`previewMemoryContext`。
- 修改 `bin/voder`：新增 `voder memory ...` 独立分支；原有 `plugin/model/doctor/daemon` 路径、现有包、Rust VGR、上游Pi源文件均不改。
- 新增 `voder-agent/scripts/test_memory.sh` 与 README、离线测试集。根目录更新部署由 `tools/install_into_root.py` 实施，文件允许列表严格锁定，保存根目录独立备份。

## 行为验收

| 条件 | 本地结果 | 说明 |
|---|---|---|
| 明确选择，按条目导出 | PASS | 不选条目不导出 |
| 个性化用户/项目/会话元数据 | PASS | 数据标签，并非特权 |
| 人工明确确认后才可写出 | PASS | 不允许模型通过记忆提案工具写磁盘 |
| 自动 no-clobber、0600、本地软链接防护 | PASS | Linux/本测试环境；未对敌对并发多用户完成证明 |
| 导入严格验证 SHA/UTF-8/schema/数量 | PASS | SHA 不提供密码学签名或用户身份认证 |
| 上下文有限且保全选中记录 | PASS | 超限拒绝而非截断 |
| 新会话显式加载记忆 | PASS（模拟宿主） | 正式 Pi + VS Code UI 未实测 |
| Pi 工具白名单、系统指令隔离 | PASS（模拟 Pi SDK） | Pi 1.1.0 真 SDK 仍待联调 |
| 真实模型调用工具生成记忆 | UNVERIFIED | 无有效模型凭证/正式运行时调用证据 |
| VS Code 文件上传按钮/GUI | NOT IMPLEMENTED | 已提供 CLI 和 SDK 入口，VS Code 扩展本身尚未实现 |
| Rust VGR 真提交与沙箱工具回归 | UNVERIFIED | 未触碰旧 VGR 代码，不代表真实路径验收 |
| 程序重启后保留待审批草稿 | NOT IMPLEMENTED | 模型提案只在内存中；**经确认的导出文件**可持久携带 |

## 参考方案与边界

- Pi 1.1.0：沿用原生 SessionManager、上下文裁剪/压缩与直接工具 allowlist 接口，不拿 Pi 的会话完整 JSONL 当个性化便携记忆。
- Claude Code 2.1.89：借鉴项目规则、auto-memory 和手工管理记忆的产品分层；不复制用户提供的压缩商业代码。
- Voder 差异化目标：记忆条目由用户按需求选择、以可移植数据文件携带、主动跨新会话注入、且与模型执行权限分离。
- “比Pi好”只有这一具体功能维度新增能力，尚无性能、可靠性或成熟度对比实测，不能宣称全面超过Pi/Claude Code/Codex。

## 运行证据

2026-10-13 本地 Node v22.16.0：16/16 离线测试 PASS；原始包独立副本上初次安装 PASS、第二次安装幂等 PASS、根目录漂移拒绝覆盖 PASS；另将运行 SHA 校验以证明旧文件未发生意外变化。正式上架前仍需真实模型、真实Pi、VS Code、断电/并发、安全审计等完整验收。
