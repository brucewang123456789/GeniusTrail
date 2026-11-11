# Voder Agent Kernel / SDK — Stage 2: Native Commit Cross-Check

## 已实现（在主干中，不放在repair）

- `voder-agent-core/src/pi-host.mjs`：官方运行时只注册、只激活、只允许嵌套调用 `voder_authorized_action`；创建时及调用前后，检查Pi真实 `getActiveToolNames()`、`getCallableToolNames()`、`getAllTools()`；拒绝其他工具和SDK接口缺失。
- `voder-agent-core/src/broker.mjs`：不再仅凭JSON字段格式授予 `COMMITTED`；只有独立Rust daemon复核成功，才接受最终提交；失败保持该幂等键不可自动重试。
- `voder-native-bridge/src/commit-verifier.mjs`：使用独立、只读的`voderd`状态机读取 `session.resume → state.get → run.replay`，查验会话、run、action、唯一的 `vgr.final_committed` 事件及最终提交哈希；拒绝无法重放、进程占锁、数据歧义和错误标识。
- `voder-sdk/src/index.mjs`：可传入由宿主方固定的 `nativeBinary`、`nativeStateDir` 构造权威校验器；模型和工具请求不能设置这些路径。

## 范围与严格未完成项

**这不是完整自主编程Agent的生产级验收。** 官方Pi Session和本地Rust daemon是真实软件实例，但目前没有通用的、经Rust授权的 `executeAuthorized` Native Agent Gateway；现有`NativeVgrObserver`仍刻意无执行和提交接口。

缺口包括：可信Rust Gateway中的策略授权、模型提案与native run的不可混淆绑定、沙箱任务实际执行、独立validator证据、预算核算、最终commit/rollback、多轮纠错以及真正的模型提供商请求。现有Rust Phase6固定fixture可另行运行，以检验已存在的原生VGR沙箱/凭据/证据路径，但该fixture不等于模型驱动编程任务。

## 验证

1. 安装器在备份已有文件后仅更新指定Agent JS文件，自动运行Node离线回归；失败自动恢复原文件。
2. `scripts/probe_cloud.mjs`调用云端Pi 1.1.0真实SDK和Rust daemon，对真实注册工具进行审计，并实测伪造提交无法通过独立Rust状态检查。它**不调用任何大模型或付费资源**。
3. `scripts/run_real_vgr_fixture.sh`仅在管理员显式设 `VODER_RUN_NATIVE_MODAL_FIXTURE=YES`且配置合法`VODER_MODAL_CLI`时运行原有Rust写入fixture；可产生真实Modal费用，测试对象是隔离的工作区，证据另存于`voder-agent/evidence/native-fixture/`。

所有未运行或不满足项输出`NOT_TESTED/NOT_PASSED`，不得替换为`PASS`。

## 第三方来源及许可证

官方上游Agent源码固定为`earendil-works/pi` tag `v1.1.0`/commit `abe508e1b89912adde45528136c3221eb69acdd7`；第三方许可证继续保留于引擎的原始`LICENSE`。此补丁不复制上游源码，不修改既有VGR Rust crates，不修改云端的官方引擎源码副本。
