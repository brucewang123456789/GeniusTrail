# Voder 模块08：真实任务与可靠性验收（增量部署 / 不重写前七模块）

**版本**：M8-0.1.0 / 2026-10-14。**状态**：验收工具已编码，可独立运行本地检查；不是Voder产品生产级认证，不代表通过A01—A12。

## 工程决策

用户提供的两个冻结包有2273个重合文件，文件内容CRC一致，大包另外包含283个文件；以较大包作为来源锁定基线。实际源码已具备Rust VGR、Agent Core、SDK、CLI、VS Code等组件，但`VoderControlPlane.getDiff`仍明确返回`NOT_AVAILABLE`，CLI的diff也是占位，VS Code仍要求手动指定`voder.agentRoot`且缺真实任务级完整审批/恢复证据。因此只能进入**补缺+严格验收**，不能直接宣布现有主干具备所有产品能力。

本包**不修改或覆盖前七模块的任何现存文件**；其唯一新增包是`packages/voder-acceptance`，唯一新增主干入口是`scripts/module8_accept.sh`。一切补功能必须在真实差异证据充分后制作独立补丁，再做受保护回归；不得用验收脚本替代SDK/VGR缺失功能。

## 部署（云端终端，统一repair目录）

1. 将集成包放入`/root/autodl-tmp/repair`，从该目录解压（文件名见交付压缩包）。
2. `bash /root/autodl-tmp/repair/Voder_Module8_Integration/install.sh /root/autodl-tmp`。安装程序拒绝覆盖现存模块及入口；若目的文件已存在，必须独立比较后人工处理。
3. `bash /root/autodl-tmp/voder-agent/scripts/module8_accept.sh audit`：只读比对来源哈希、运行本地离线测试、生成私有`AUDIT.json`，不调用模型、不运行Modal。
4. 仅在自愿接受真实模型/API费用和Modal费用、具备实际Rust可执行文件与模型配置后，设置明确的双重允许环境变量，运行`bash .../module8_accept.sh live`；所有任务仅修改独立临时fixture仓库，绝不将测试补丁写入Voder正式代码。

**运行时**：推荐Node ≥22.19（上游Pi包要求）、Python3、Git、完整已构建原生Rust gateway/daemon，以及有效Modal沙箱。没有这些条件只能执行可用层级，并准确输出阻断状态。

**真实任务**：冻结3个具有客观测试的Python代码修复场景：边界条件、解析与回归、不可同时满足的矛盾测试。RT01、RT02要求**模型真实出补丁→Rust授权→Modal跑Python unittest→Rust提交→宿主工作区受控交付**，RT03要求**无法交付且没有篡改初始文件**。所有任务由正式SDK/CLI控制层进入，绝不注入模拟`COMMITTED`。

## 强制门禁与成熟度表述

- A01—A12**每项**必须以具名、可追溯证据独立PASS；任一`NOT_VERIFIED`、`BLOCKED`或`FAIL`均不准发布生产候选。
- 真实性：本地Node测试、mock SDK回执、纯Native Gateway冒烟、真实模型草稿、VS Code单元测试、历史Modal日志**均不得**冒充真实整链PASS。
- A09/A10：当前能力缺口是主干代码真实行为，必须由正式增量工程修复和GUI/原生会话回归；验收包绝不修改权威状态来制造通过。
- A12：不仅要提供任务集真实通过率、回归率、错误种类和p50/p95时延，还须真实可核验的Token/费用数据。现有SDK没有统一任务级费用回执则记为**未知/未达标**，不能填零。
- 竞品对标：Codex Harness、Claude Code、Pi、DeepSeek Harness、Zcode仅作为目标；未运行可比较的公开数据集、同一环境及相同预算，不得宣称对标性能、领先、超越或等价。
- 所有产物位于`/root/autodl-tmp/repair/voder-module8-evidence/<run-id>/`；日志和源码哈希不含API Key；结果声明真实来源、真实环境与局限。**严禁**自动删除基线、自动部署正式包或自动推送代码。

## 发布所需后续独立证据（当前没有伪造）

A01缺文件阻断/恢复；A02跨阶段黄金法则与压缩；A03模型动态调用四类工具；A04Skill沙箱越权拒绝；A05真实多轮诊断修复；A06原生Rust权威提交；A07失败/超时/杀进程/幂等恢复；A08离线搜索和可撤销记忆；A09CLI完整提交/审批/diff/恢复/导出；A10真实VS Code扩展GUI及相同会话；A11≥2隔离会话资源/工作树一致性；A12冻结任务集统计与费用。**少一个门禁就不是生产候选**。

## 独立发布门禁

`bash .../module8_accept.sh release --audit /path/AUDIT.json --live /path/REAL_SUITE.json`：重新读取固定发布契约，不接受历史测试PASS或人为填写PASS替代真正的A01—A12证据；当前真实能力缺口未解决时，明确输出`BLOCKED`并保存机器可读证据。该入口刻意不能把3个单文件任务通过宣布为生产合格；后续必须在模块08内部继续补齐真实验收覆盖。
