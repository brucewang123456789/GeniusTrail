# Voder SDK／CLI模型回答交付修复（2026-10-13）

- 修复真实Pi模型返回的`draft`被`VoderControlPlane.safeSummary`丢弃的问题：同一进程中的`task.wait`现在返回完整、受限长度的`draft`及`output_authority`，`voder run`默认JSON输出完整模型回答，`--format text`提供清楚标注的终端正文展示。
- 保留现有退出码：`0`仅供VGR验证成功，`3`表示`DRAFT_UNVERIFIED`而不是命令故障。不能把模型回复冒充已验证交付。
- 会话Journal、`status`、历史`resume`、`export`与事件流继续只携带安全元数据，模型正文仅返回当前存活的CLI／JSON-RPC会话，关闭进程即消失；不能通过历史事件读取全文。
- 入口`bin/voder`固定为可执行，并选择Node≥22.19；优先选择已验证的托管`v22.23.3`，可用`VODER_NODE_BINARY`明确覆盖。旧Node v22.16不再静默进入Pi执行路径。
- `prepare_node_runtime.sh`在缺兼容Node时仅从Node.js官方下载v22.23.3 Linux x86-64归档，核对官方SHA256后在Voder自有`.voder-tools`新目录中安装唯一`bin/node`文件。无法联网则停止并显示原因，不替换系统Node。
- 真实模型验收脚本`test_cli_real_model_e2e.mjs`经CLI的用户可见结果核对随机文件标记，校验无源文件写入、无VGR伪承诺、日志脱敏、跨进程元数据读取。

## 边界
本补丁不改Pi原生引擎、不改Rust VGR、不改Agent内核或记忆处理；真实模型E2E仍必须在AutoDL执行。VS Code及原生VGR最终提交不是本次修复范围。
