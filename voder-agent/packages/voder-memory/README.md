# Voder Portable Memory · 第六模块（可移植工程记忆）

**状态：已实现并通过离线单元/模拟 SDK 回归；尚未经过真实 Pi 模型、VS Code 扩展、长会话和 AutoDL 全链路验收。不是整个 Voder 的生产级验收。**

## 目标与边界

- **记忆是可插拔的数据，不是必须安装/启用的功能**：新终端默认无持久记忆，用户明确选择零到八个便携记忆文件；不因工程目录出现文件就自动加载。
- 用户可要求模型调用 `voder_memory_propose` 提出至多 12 条候选；此工具只在 `enableMemoryProposals:true` 启用后出现，只产生内存中的待确认草稿，不能写文件或执行代码。
- 用户审查候选后，使用 `approveMemoryProposal(...,{selectedIds,outputPath,userConfirmed:true})` **显式批准**所选条目；不勾选的记忆不进入文件。也可通过无模型 CLI 从自定义 JSON 输入中选择导出。
- 文件格式 `voder.memory.bundle.v1`，UTF-8 JSON，内含 schema、profile、entries、SHA-256；可移动到另一台机器，使用 SDK `createSession({memoryFiles:[path]})` 在新的独立对话启动时注入为**不可信用户数据**，而不是 system 指令、模型权威历史或 VGR 执行权限。
- 与 Pi 的原生 session/compaction、上下文管理、工具路由机制**并行互补**，不修改 `engine/voder-agent-engine` 上游源码，不劫持系统 Prompt。与 Claude Code 的项目规则/记忆概念相互借鉴，但文件格式和权限设计独立实现。
- 没有导出的历史无法被“记起”；选出的记忆是摘要与事实主张，不是完整历史 transcript。摘要可能不准确，必须由用户审核，关键事实还需原始证据核对。

## 无模型命令行演示

创建 `entries.json`：

```json
{
  "entries": [
    {"id": "editor-style", "kind": "preference", "scope": "user", "title": "编辑习惯", "content": "保留仓库原有文件名，不自动覆盖冻结文件。", "source_ref": "用户明确要求"},
    {"id": "project-build", "kind": "project", "scope": "project", "title": "项目入口", "content": "先检查构建脚本和上游接口，再改动模块。"},
    {"id": "ignore-me", "kind": "reference", "scope": "session", "title": "临时话题", "content": "不希望导出这一条。"}
  ]
}
```

在 AutoDL 安装后：

```bash
cd /root/autodl-tmp
./bin/voder memory export --input entries.json --select editor-style,project-build --output /path/for/user/carry.vmem.json --profile my-work --confirm
./bin/voder memory inspect --file /path/for/user/carry.vmem.json
./bin/voder memory verify --file /path/for/user/carry.vmem.json
./bin/voder memory render --file /path/for/user/carry.vmem.json
```

`render` 向标准输出打印可粘贴到新会话的上下文片段（注意其内容是用户私有数据）；亦可将文件移动/上传到新终端，传入 SDK 显式选择：

```js
const sdk = new VoderSDK({cwd: '/work/project',native: authorizedVgrTransport,modelHost: model});
const {sessionId} = sdk.createSession({memoryFiles:['/work/uploads/carry.vmem.json'],enableMemoryProposals:true});
await sdk.submitTask(sessionId,{goal:'继续当前代码任务，参考导入的个性化记忆'});
const pending=sdk.getMemoryProposals(sessionId); // 让用户审阅具体文字
if(pending.length){
  await sdk.approveMemoryProposal(sessionId,pending[0].proposalId,{
    selectedIds:['editor-style'],outputPath:'/work/exports/curated.vmem.json',profile:'my-work',userConfirmed:true
  });
}
```

> `authorizedVgrTransport` 和 `model` 是现有 Voder SDK 所需的宿主对象，示例没有伪造实现；正式 UI 必须让用户明确勾选并授权。已有会话历史不会自动改变，导入仅在新任务入口发生；若要添加更多记忆，请新建 Session。

## 设计不变量

1. 零自动加载、零自动记忆持久化；模型写提案不等于用户认可。
2. 只导出 `selectedIds`，每条采用固定结构、长度限额和去重；导出文件 128KiB 上限，文件模式 0600，原子 no-clobber，禁止软链接路径。
3. 导入校验严格 schema、UTF-8、长度和 SHA-256；跨文件同 ID 冲突直接报错。SHA-256 只做**完整性检查**，不能证明来源可信或内容正确。
4. 记忆注入属于 user prompt 的明确标注数据层，永远不能覆盖系统 Prompt、当前用户需求、Tools Broker 或 VGR 原生 commit 条件；超额报错，不截断选中记忆。
5. 规则拒绝明显的密钥模式，但**不能保证检测全部隐私和凭证**；用户必须在导出/转发前检查内容。受限文件输出不等同于加密或秘密安全存储。
6. `voder_memory_propose` 草稿只在 SDK 进程内存在，实例释放后消失，故用户必须在会话关闭前完成批准导出；这与便携已导出记忆的长期存储是两回事。

## 预期下一验收

在具备真实 Pi 1.1.0 构建与模型凭证、Voder 已完整部署且 VS Code 客户端可用的环境中，完成真实工具调用 / 用户交互式选择 / 新终端跨进程上传回放 / 错误注入 / VGR 真正提交安全边界测试；当前版本仅完成离线验证，没有触达这些门禁。
