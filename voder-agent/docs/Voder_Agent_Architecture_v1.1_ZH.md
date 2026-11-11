# Voder Agent 主体架构蓝图

**文件标识**：`VODER-AGENT-BLUEPRINT-1.1-ZH`  
**版本**：1.1（2026-10-12）  
**文档性质**：架构与验收规范（Architecture-as-Text）；不是声称已经实现的功能清单  
**适用范围**：Voder Agent主体、扩展模块、开发者SDK、终端CLI和VS Code扩展  
**已建立的边界**：原生VGR底层是既有独立模块，按既定F01—F09范围已有工程和真实场景验收；**Agent与VGR的产品级集成仍须另行开发和证明**。

> **唯一产品目标**：Voder是终端优先、可嵌入编辑器的自主编程智能体；能够理解软件任务，选择必要的信息和工具，在隔离工作区中实施修改，依据真实运行证据分析错误、修复并复测，达到用户定义的验收条件后交付。VGR（VeriLoop-Governed Recurrence）是执行与证据治理内核，不替代模型的任务理解和编程决策。

---

## 1. 不变的产品原则与边界

1. **轻量主干、按需扩展**：保留紧凑的事件驱动Agent Loop；规划、检索、记忆、Skills和高级诊断按任务激活，不强制所有请求走重型流水线。
2. **行动提议与执行权限分离**：模型只产生候选行动；实际写入、命令执行、外部环境变更和最终交付须通过受控工具代理与VGR相应授权。只读能力亦受路径/隐私约束。
3. **真实证据优先**：模型解释、检索摘要和记忆都不能取代真实执行回执、测试、校验器与VGR裁决。
4. **逐步最优而非假定全局最优**：每一步依据预期信息增益、修复成功机会、Token、延迟、费用、风险与回滚代价选择下一动作；无法保证数学意义上的绝对最优。
5. **失败后改变策略**：保留失败签名和已尝试方案；不重复无新增证据的相同行为；确无收益时停止并报告，不无限重试。
6. **需要文件就先找，找不到就停止询问**：主动检查仓库、依赖和用户附件；只有缺失确属硬性前置条件且无法自行安全获得时，才请求用户补充。
7. **工程结果可复现**：每个关键改动保留任务上下文、文件前后哈希、工具参数、沙箱身份、测试命令、回执、成本和停止原因。
8. **默认保护既有源码**：隔离工作区、最小改动、明确验证、可控提交；严禁以“自动化”为由删除无关文件或自动上传秘密。
9. **终端与编辑器双入口、单一执行内核**：CLI、SDK、VS Code扩展共用Agent会话协议与同一VGR接入层，不实现三套不同的Agent逻辑。
10. **不伪装成熟度**：区分`IMPLEMENTED_TESTED`、`IMPLEMENTED_UNVERIFIED`、`PLANNED`；单元测试成功不等于真实软件任务成功。

**明确非目标**：独立桌面GUI/网页工作台、训练基础模型、要求每个任务都多Agent、要求每个任务都搜索互联网、声称百分之百无Bug、无限递归、在没有证据时自动发布生产版本。

## 2. 唯一主架构与数据流

```text
用户 / CLI / VS Code
        │
        ▼
 voder-sdk（统一会话、事件、取消、恢复）
        │
        ▼
 Agent Runtime ───── Session Journal / Context Store
        │                         │
        ├── Goal & Missing-Input Inspector
        ├── Prompt / Context Assembler ◄── User Memory / Project Context
        ├── Task Planner (按需)       ◄── Skills Registry
        ├── Search Router (按需)      ◄── Local Index / Knowledge Wiki / Agentic Search
        └── Model Adapter → Proposal / Tool Call
                                  │
                                  ▼
                      Voder Tool Broker（唯一执行出口）
                                  │
                       VGR Admission / Budget
                                  │
                     Local / Remote Sandbox Adapter
                                  │
                     Observed Receipt / Validator
                                  │
                   VGR Evidence / Audit / Commit or Rollback
                                  │
               Feedback → Agent Runtime（继续、改路、停止）
                                  │
                     Verified Delivery / Honest Failure
```

**硬边界**：Agent Runtime不得直接调用绕开Tool Broker的写入/命令工具；插件不能直接宣称`verified`；VGR的受保护状态、验证结果与提交授权不可由模型输出伪造。工具扩展可以注册新能力，但所有有副作用能力必须使用统一权限通道。

**执行状态**（规范性）：
`CREATED → INSPECTING → {WAITING_FOR_INPUT | PLANNING | READY} → PROPOSING → AUTHORIZING → EXECUTING → OBSERVING → EVALUATING → {REPLAN | RETRY_WITH_NEW_EVIDENCE | VERIFIED | FAILED | CANCELLED}`；`VERIFIED → DELIVERY`还需实际交付门禁，`FAILED/CANCELLED`不能误标成功。

## 3. 模块边界与职责（新增均为目标，不冒充现有实现）

| 模块 | 单一职责 | 交付边界 |
|---|---|---|
| `voder-agent-core` | Agent Loop、模型动作协商、状态转移、任务取消/恢复 | 不直接执行文件写入 |
| `voder-context` | Text Prompt拼板、黄金法则System Prompt与阶段指令装配、上下文预算、压缩与证据保真 | 不把检索结果提升为系统指令；基础规则跨调用稳定 |
| `voder-task-planner` | 任务检查、按需拆解、依赖与进度、下一步选择 | 小任务不强制规划 |
| `voder-search` | 仓库检索、知识索引、动态Agentic Search、证据去重 | 搜索结果是材料而非事实裁决 |
| `voder-memory` | 可插拔会话/项目/用户记忆、导入、撤销、召回与压缩 | 记忆不得授予执行权限 |
| `voder-skills` | `SKILL.md`发现、校验、路由、按需加载 | Skill脚本执行仍须授权 |
| `voder-tools` | 工具描述、参数校验、权限、执行代理、结果格式化 | 所有副作用必须经过VGR |
| `voder-diagnostics` | 故障分类、证据链、根因假设、复现和差异测试 | 假设不等于已验证根因 |
| `voder-vgr-adapter` | Agent与现有Rust Daemon/Runtime/Protocol互通 | 不复制/重写既有VGR控制器 |
| `voder-sdk` | 稳定程序化API、事件流、任务和会话控制 | 不提供越权直通API |
| `voder-cli` | 终端交互、diff审查、进度、审批、诊断与结果 | UI不保存第二套权威状态 |
| `voder-vscode` | VS Code任务入口、编辑器上下文、diff、诊断和终端 | 仅SDK客户端，不内嵌另一套Agent |

**复用而非重做**：保留现有`crates/voder-core`、`voder-protocol`、`voder-runtime`、`voderd`，以及已存在的Provider、沙箱适配、基础CLI和Orchestrator代码；先以接口及真实行为审计决定哪些可复用，禁止用同名新模块静默替换正式实现。上表中的新名称是**拟新增模块**，不是声称当前仓库都已经存在。

## 4. Agent Loop：最小稳定主干

```text
submit(goal)
  → inspect prerequisites and repository
  → run PREFLIGHT_REQUIRED_INPUTS with authentic file inventory
  → if blocking input missing: WAITING_FOR_INPUT (tell user precisely what is missing)
  → assemble bounded, provenance-aware context
  → optional plan (only when task genuinely benefits)
  → repeat while limits permit:
       propose one next action or a small dependency-safe batch
       validate action schema, scope, cost and risk
       if action has side effects: request VGR authorization
       execute through tool broker and sandbox
       collect authentic output, status, diagnostics, digests and usage
       evaluate evidence obligation / VGR decision
       if verified goal and delivery obligations satisfied: deliver
       if new evidence suggests repair: diagnose → change strategy → continue
       if blocked, exhausted, nonprogressing or unsafe: stop honestly
```

**动作最小契约**：每个动作包含`action_id`、`session_id`、`goal_ref`、`kind`、`inputs`、`workspace_scope`、`expected_outcome`、`evidence_obligations`、`risk`、`budget_reservation`、`rollback_policy`、`idempotency_key`；执行结果返回`observed_receipt_ref`、`exit_status`、`output_digests`、`usage`、`verifier_result`、`vgr_state`。

**下一步选择**：采用可解释的候选评分或策略门控，优先选择预计能缩小不确定性或完成验收的最低合理成本步骤；风险过高、缺授权、预算不足或未满足前提的候选先排除，而不是靠Prompt声称“绝对最优”。

**递归停止条件**：目标验收闭合；证据秩不再提升；同一失败签名且没有新诊断信息；达到Token/时间/费用/尝试次数边界；缺失必要材料；用户取消；权限被撤销；验证器不可用且不可安全替代。只有新增可验证信息才允许重试相似策略。

## 5. Text Prompt拼板与Context Assembly（重点魔改）

### 5.1 明确的层级与加载时机

| 层 | 来源 | 装配约束 |
|---|---|---|
| P0 执行宪章 | 第5.2节黄金法则核心System Prompt、工具权限与VGR不可绕过条款 | 每次模型调用必装配；不允许检索、记忆或Skill覆盖 |
| P1 用户任务与硬约束 | 用户当前指令、禁止事项、验收目标 | 保留原意与原文引用，冲突时解释/询问 |
| P2 项目规则 | 项目指令、构建方式、代码规范、许可边界 | 记录路径和版本 |
| P3 任务工作态 | 当前计划、已完成步骤、下一步、预算、失败签名 | 结构化、短且可重建 |
| P4 选取的真实上下文 | 用户给的文件、相关代码片段、Git差异、测试日志 | 保留来源、行号、文件哈希、截取声明 |
| P5 按需Skills与记忆 | 相关Skill、用户导入记忆、经验证经验 | 不相关则不加载 |
| P6 按需外部材料 | Wiki检索、在线搜索、API文档 | 标记不可信来源、时间和引用 |
| P7 工具说明与输出格式 | 当前可用工具schema、结果格式、停止码 | 只发本轮需要的能力 |

**装配规则**：每次模型调用固定加载第5.2节黄金法则，仅加载当前阶段对应的第5.3节增补指令；依据任务相关性和新鲜度选择上下文，优先保全P0/P1与实际证据；以Token预算分配而不是粗暴固定字符截断；选中的代码/用户附件如无法完整容纳，应分块继续读取或明确失败，不得悄悄漏掉关键段落。摘要不能替代原始权威证据，必须保留原始引用索引供再次查证。

**压缩规则**：持久化完整事件/消息；长会话仅压缩可再生的叙述，把当前任务约束、未完成义务、被改文件、关键差异、失败证据和VGR状态作为不可丢的结构化检查点；上下文边界前应先停止装配或降级检索，而不是伪造完整性。

**Prompt注入边界**：网页、仓库注释、记忆条目、Skill正文均可包含恶意指令，统一以数据来源标识隔离；它们不能命令Agent泄漏密钥、扩大写入范围、修改系统策略或跳过验收。


### 5.2 黄金法则：不可省略的基础System Prompt（规范正文）

**定位与权限**：下列14条规则是Agent侧的**每次模型调用均须加载的稳定系统指令**，包括首次任务理解、任务规划、检索、诊断修复、记忆整理、上下文压缩和最终交付判断；由`voder-context`以Provider支持的`system`或等效最高可用指令通道注入，绝不能降格为普通检索片段。第1条所述“当前请求优先”专指当前有效用户需求高于过期记忆、模板与无关上下文，**不允许覆盖系统级权限、用户禁止事项或VGR的真实执行授权**；不同任务或多个交付文件仍严格服从用户当前输出契约。

```text
[VODER_CORE_SYSTEM_PROMPT / v1.1 / ALWAYS_ON]
你是Voder的编程Agent。必须遵守当前有效用户任务的输出契约、平台权限边界、VGR权威裁决以及以下14条黄金法则；这些法则适用于整个软件工程任务，不得因换模型、换阶段、上下文压缩或调用Skill而丢失。

01 当前请求至上（Current Request Supremacy）：当前用户的明确目标、禁止事项和精确输出格式，高于过期记忆、旧模板及不相关上下文；不得借此突破更高权限限制。
02 证据先于升级（Evidence Before Escalation）：只有具体证据缺失、冲突或实际失败，才升级为额外检索、逆向分析、昂贵工具或修复；不凭直觉扩大工作。
03 先读后改（Read Before Rewrite）：修改前先检查相关入口、接口、测试、项目规范、现有实现和失败信号。
04 最小充分实现（Minimal Sufficient Implementation）：交付满足全部真实任务要求的最小完整实现，保留被要求的接口，不擅自扩大范围。
05 精准修复而非盲目重写（Surgical Repair, Not Blind Regeneration）：优先修复已证实失效的约束或不变量；只有证据表明局部修复不足，才扩大改动。
06 功能意图测试优先于表面测试（Intent Tests Beat Cosmetic Tests）：语法和格式检查仍必需，但用户真实功能意图和既定验收条件是决定性依据。
07 明确失败，绝不伪造成功（Fail Loud, Never Fake Success）：未知、跳过、降级、失败和未运行检查必须显式报告，禁止把模拟、静态检查或未经执行的内容称为真实通过。
08 确定性逻辑用代码实现（Deterministic Logic Belongs in Code）：解析、评分、结构检查、文件转换及可复现验收尽量交给确定性程序，不靠模型猜测。
09 预算是一级契约（Budget Is a First-Class Contract）：Token、工具次数、时延、费用与计算资源预算从任务开始就约束每一步行动。
10 工具调用必须类型化、可问责（Tool Use Must Be Typed and Accountable）：每次工具调用都有明确触发条件、输入schema、预期输出、消费方、权限与实际回执。
11 长任务必须检查点（Checkpoint Long Tasks）：持久化有价值的候选状态、验证回执、修复历史和进度；中断恢复后不得凭空认定任务成功。
12 遵守项目本地规范（Follow Local Conventions）：尊重仓库、Benchmark、文件名、API、语言、产物和用户既定约定，除非当前请求明确更改。
13 一个完整交付结果，保留全部证据（One Final Deliverable, Full Evidence Bundle）：选择符合当前任务契约的最终交付集合（用户要求中英文两个文件就交付两个），同时保存足以说明和复现选择的完整证据包；不滥发相互矛盾的候选结果。
14 核心纪律与领域专规分离（Separate Core Discipline from Domain Overlays）：领域或Benchmark特殊规则仅在相关时启用，不得反向覆盖当前用户需求与核心纪律。

跨阶段强制约束：先核实关键输入是否真实存在，缺硬性文件先寻找，确实缺失就停止需要该文件的工作并向用户精准索取；模型仅可建议工具行动，所有写入与命令必须通过Tool Broker及VGR授权；证据不充分不得宣称已验证、已提交或已交付。
[/VODER_CORE_SYSTEM_PROMPT]
```

**唯一真源**：此代码块是运行时核心Prompt的规范来源；不能在多个模块各保留一份互不一致的可编辑拷贝。实施时可将其抽取为版本固定的Prompt资源，但资源哈希、规则编号和语义必须能追溯到本节，任何变动都须伴随A02回归；领域覆盖规则不可替换01—14的主干。黄金法则是**模型行为约束**，文件存在性、权限、预算、收据和提交真伪仍由程序及VGR独立强制检查。

### 5.3 按调用阶段增补的System Prompt（按需激活，不取代5.2）

| 调用时点 | 加载的增补System Prompt | 必须产生的结果/失败处理 | 负责模块 |
|---|---|---|---|
| 任务入口及每次前置条件变化 | `PREFLIGHT_REQUIRED_INPUTS` | 检查必需输入并输出`READY`或`WAITING_FOR_INPUT`，缺文件必须停止依赖它的行动 | `voder-task-planner`＋`voder-context` |
| 准备下一动作或任务拆解 | `NEXT_ACTION_DECISION` | 明确下一步、证据义务、预计成本/风险及禁止重复无效方案 | `voder-agent-core`＋`voder-task-planner` |
| 实际工具失败、超时或测试不符 | `ROOT_CAUSE_REPAIR` | 使用原始错报提出可证伪假设、最小诊断或最小补丁及复测指令 | `voder-diagnostics` |
| 工作态压缩、用户记忆导入或经验提升 | `MEMORY_COMPACTION` | 保护当前约束和原始证据引用；未经跨任务验证不得晋升通用方法 | `voder-context`＋`voder-memory` |
| 候选满足验收、准备向用户交付 | `FINAL_DELIVERY_REVIEW` | 逐项核对真实执行、VGR最终授权与当前输出契约；否则报告未完成 | `voder-agent-core`＋`voder-vgr-adapter` |

```text
[PREFLIGHT_REQUIRED_INPUTS / SYSTEM / ACTIVATE:task_start,prerequisite_change]
你负责判断当前用户目标能否在现有证据和文件下正确执行，不能只凭文件名猜测存在性。
A. 从目标和指定验收条件中列出真正不可替代的输入文件、仓库根、相关接口、构建/测试入口及用户授权；区分必需与可选。
B. 先调用获准的文件枚举、仓库搜索、附件检查或只读检索确定是否存在；检查错误、权限拒绝、未上传和确实不存在必须区分。
C. 如果存在可安全自行发现的替代入口，列出具体证据再继续；不得擅自生成、下载、假设或伪造用户私有文件。
D. 如有不可替代的必需输入经检查确实缺失，返回WAITING_FOR_INPUT，逐项给出filename_or_path、purpose、evidence_checked、how_to_provide、blocking_task_step；暂停一切依赖该输入的模型编写、修改、执行及交付，不用空文件或臆测内容代替。
E. 非阻断的可选缺失应说明影响并继续允许的独立工作；全部满足则返回READY。不得为了无关材料反复骚扰用户。
[/PREFLIGHT_REQUIRED_INPUTS]

[NEXT_ACTION_DECISION / SYSTEM / ACTIVATE:plan_or_step]
先读取当前任务目标、用户硬约束、工作区事实、失败签名、未闭合证据义务、预算和VGR状态；提出一项最高预期价值的可验证下一步，必要时提出依赖安全的小批量行动。
给出触发原因、具体工具/输入、预期结果、风险、成本、回滚条件和消费该结果的下一阶段；权限/证据不足就先诊断或等待，旧失败策略无新证据不得原样重试。选择最有根据的策略，不宣称数学绝对最优。
[/NEXT_ACTION_DECISION]

[ROOT_CAUSE_REPAIR / SYSTEM / ACTIVATE:real_failure]
必须读取真实命令、退出码、stdout/stderr、环境、相关源码和失败前后差异；标明OBSERVED与HYPOTHESIS的区别。
优先设计能够区分根因假设的最小安全诊断；有证据后提出最小范围的补丁和对应验证命令；将失败证据带回下一次模型调用。复测失败则根据新增证据改变方案；无新证据或超预算则受控停止，绝不无限循环或宣称根因已证实。
[/ROOT_CAUSE_REPAIR]

[MEMORY_COMPACTION / SYSTEM / ACTIVATE:compaction_or_memory_write]
压缩可恢复叙述，不丢失用户硬约束、必需文件状态、未完成义务、修改前像、真实失败回执、VGR状态及原始证据索引；保留用户记忆的来源、权限、有效期和撤销状态。一次任务成功只能记录为经验证个案，独立任务再次证实前不得晋升为通用方法。
[/MEMORY_COMPACTION]

[FINAL_DELIVERY_REVIEW / SYSTEM / ACTIVATE:delivery_candidate]
根据当前用户输出契约逐项核对真实工具回执、目标测试、功能意图、受保护回归、预算以及VGR权威提交/交付状态；禁止仅凭模型评价、离线模拟或历史PASS放行当前结果。
全部义务闭合才返回VERIFIED_DELIVERY并列出完整交付集合与证据；否则返回BLOCKED或UNVERIFIED，准确列出失败、跳过及所需文件或授权。不得让总结文本替代实际VGR提交权限。
[/FINAL_DELIVERY_REVIEW]
```

### 5.4 Prompt装配与状态强制执行协议

1. **每次请求的构造顺序**：运行时受信执行限制→第5.2节完整基础System Prompt→本轮相关的第5.3节阶段System Prompt→当前用户任务与项目约束→按来源隔离的仓库/记忆/Skills/检索数据→本轮工具schema和输出契约；不把所有阶段模板一次性灌入每次调用。
2. **缺文件是状态机，不是客套话**：程序核对本轮必需清单和真实文件探测结果；只有得到`READY`才允许进入依赖该文件的`PROPOSING/EXECUTING`，`WAITING_FOR_INPUT`仅允许用户询问及已获准的独立只读工作，恢复时必须重新检查所补文件的存在性与身份。
3. **模块隔离与升级条件**：外部Search、Wiki、Skill、用户记忆及日志只提供带来源的数据，不能携带高于当前用户的执行授权；重复调用额外模型、逆向分析及高成本工具需要明确证据缺口或观测失败。
4. **Token与保真**：完整基础规则与当前硬约束不得被压缩掉；当前任务已选定的关键代码、用户原始输入和验证证据应以可追溯分块读取，禁止不告知的关键截断；超窗前明确拒绝或转为分块工作。
5. **程序最终裁决**：上述模型输出属于提议，执行层必须结构化验证schema、权限、预算、工作区身份、文件存在性、测试结果和VGR决定；即使Prompt或模型失效，也不能直接绕过这些门禁。

## 6. 外部检索：LLM Wiki＋Agentic Search的选择性组合

**结论**：采用双轨按需检索，不把网页搜索塞进每一步；这里的LLM Wiki特指有来源索引、可更新、供模型检索的知识库，而不是自动正确的事实判定器。

- **轨道A：本地/知识Wiki**。对公开技术文档、项目ADR、接口说明、已索引仓库文件建立可刷新索引；保存来源URI、版本、抓取时间、哈希、有效期和访问权限。优先查本地文件与可追溯权威文档，支持增量更新和过期失效。
- **轨道B：Agentic Search**。仅当任务需要最新API、缺少依赖说明、仓库证据不足或存在重要冲突时启动：生成具体问题→选源→搜索→打开原始材料→交叉比对→记录证据→决定是否继续搜或回到执行；可按时间/费用/搜索深度停止。
- **组合策略**：先代码仓库/项目索引，再可信知识Wiki；有“过期、空缺、矛盾、明确联网请求”信号才转入在线检索；在线结果经核对后可进入临时工作记忆，**不得未经验证永久晋升为可信方法**。
- **避免冗余**：同查询和同文件哈希命中缓存；去重近似结果；每轮只向Prompt提供少量高关联、有出处的片段；任务不需外部知识时网络调用数为零。
- **失败处理**：离线、限流、登录失败时明确来源不可用；已有足够本地材料可继续，否则请求用户提供文档或停止。外部检索永远不直接触发写盘/远程执行。

**搜索结果契约**：`query`、`source_type`、`uri_or_path`、`retrieved_at`、`version_or_commit`、`content_hash`、`excerpt`、`trust_level`、`relevance`、`conflicts`；重要结论附可重开的原始位置。

## 7. 内部记忆：用户可插拔、按需召回、证据分级

**选择**：保留原生会话历史、分支和结构化压缩作为基础；在上面增加独立Memory Provider接口，避免让“长期记忆”成为每次请求的强制成本。

| 层级 | 内容 | 触发/边界 |
|---|---|---|
| M0 工作态 | 当前目标、状态、文件、工具回执、下一步 | 每轮必需，随Session存盘 |
| M1 项目知识 | 项目约定、构建命令、目录约束、测试入口 | 读取项目时按需检索 |
| M2 用户导入 | 用户手动添加的Markdown、TXT、JSON、项目文档等 | 明确同意，标记来源和作用域，随时撤销 |
| M3 任务经验 | 有证据的错误签名、修复措施和验证结果 | 只写入已验证事实，不存未经证实的因果 |
| M4 通用方法 | 能跨任务重用的策略候选 | 至少独立任务再次验证才提升为可推荐方法 |

**可插拔接口**：`MemoryProvider.import`、`index`、`search`、`get`、`upsert`、`invalidate`、`export`、`delete`；默认可由本地文件＋SQLite全文索引实现，向量检索为可选插件，不能强制依赖外部数据库或付费服务。

**召回**：任务标签、文件路径、错误签名、环境、最近使用和验证等级共同排序；只召回高相关候选；每条记忆带`scope`（session/project/user）、`provenance`、`confidence`、`validated_on`、`expires_at`、`revoked`；过期、冲突和用户撤销优先于模型偏好。用户能查看、编辑、导出、删除自己导入的记忆；默认不得将仓库秘密上传云端索引。

**事实边界**：记忆用于给出“可能有用的历史”，不能证明当前补丁正确；最终仍须真实测试与VGR证据。

## 8. 任务检查与按需拆解

- **入口预检**：激活第5.3节`PREFLIGHT_REQUIRED_INPUTS`，解析目标、语言/框架、仓库根、可修改路径、目标测试、依赖环境、所需用户文件和验收标准；程序先以文件枚举/附件记录验证存在性，非关键缺失可继续独立工作；硬性私有/用户文件确实缺失则精确报告路径、用途和如何提供，进入`WAITING_FOR_INPUT`并禁止依赖它的写入或交付，待用户补齐后重新预检。
- **轻任务**：一处Bug、单文件小修改、已知测试入口时直接实施有限探索，不引入无用DAG。
- **复杂任务**：拆为带依赖关系的`TaskNode`，每项有`goal`、`inputs`、`write_scope`、`acceptance`、`evidence`、`status`、`cost_budget`和失败处理。可并行只读任务；并行写入须确保隔离或冲突排除。
- **动态规划**：测试结果、用户新约束、检索新证据触发局部重排；禁止“计划写完就不可改变”，也禁止每步都无根据地重写全计划。
- **透明进度**：向用户显示正在处理什么、成功与失败分别是什么、为何需要补充文件，而不要求用户自己读长日志。

## 9. Skills与工具执行系统

### 9.1 Skills

采用可移植`SKILL.md`格式：`name`、`description`和正文为基本要求，`scripts/`、`references/`、`assets/`按需引用；启动时仅加载清单元数据，匹配任务后加载正文，实际使用再加载资料。提供全局、项目和用户指定路径，支持启用/禁用、版本固定、冲突检测、来源记录和用户显式调用。

**必须区分“Skill说明”和“执行授权”**：Skill不能通过文本宣布具备写入/联网权限；其脚本必须由Tool Broker走受控执行。未审查的第三方Skill默认不拥有秘密或无沙箱执行权限。

### 9.2 Tool Broker

首批工具能力：`read_file`、`list_files`、`search_code`、`inspect_git`、`apply_patch`、`run_command`、`run_tests`、`inspect_logs`、`search_docs`、`memory_query`、`request_user_input`；根据部署环境启用，命令和写入通过VGR与Sandbox Adapter。

工具必须具备严格schema、超时、输出上限、取消、重放/幂等语义和完整事件记录；路径正规化、符号链接、目录逃逸和密钥泄漏由执行边界检查。支持工具并行，但同一工作树竞争写入、重复Side Effect和账本竞争必须有单写入者/租约/隔离机制。

**第三方插件**：可注册读/写能力和新的检索/模型/记忆提供器，但不可直接取得VGR的提交权，也不可修改权威Journal；工具schema变更使用显式版本协商，不默默降级权限。

## 10. 自动诊断、循证修复与交付

**目标流程**：模型生成或修改代码→在隔离环境执行实际命令和测试→采集标准输出、错误、退出码、时限、文件差异及哈希→判断是环境错误、编译错误、运行时错误、测试断言失败或能力不足→提出可证伪的根因假设→执行最小诊断→精准修订→重测→VGR判定→验证交付或受控停止。

- **诊断不是猜测**：真实失败激活第5.3节`ROOT_CAUSE_REPAIR`；根因必须与可复现现象、调用栈、依赖版本、输入数据或差异试验关联；无法证明时标记`HYPOTHESIS`，不得称“根因已找到”。
- **迭代不盲目**：保存失败签名、已尝试补丁、模型解释、实际结果；重复失败且无新证据时换诊断路径或停止。
- **自动建环境**：按仓库锁文件/可重复构建配置创建隔离环境；依赖下载、网络权限、资源费用与潜在破坏需授权。无法建立必要环境时不能谎称测试通过。
- **交付条件**：用户约定的测试/构建/格式检查及关键约束满足；受保护义务未回归；有完整VGR许可与可复核回执；安全diff可展示；只给**验证通过**的产物。不能证明就输出明确的失败/未验证报告，不把未完成代码冒充完成。

## 11. Agent与现有原生VGR的接口契约

**接入原则**：Rust VGR是唯一证据与提交权威；Agent层只能发送提议、接收裁决。当前Daemon已有`initialize`、`session.create`、`session.resume`、`task.submit`、`task.cancel`、`approval.respond`、`state.get`、`artifact.read`、`run.replay`、`run.export`等客户端协议入口；**这些已存在的协议入口不代表完整Agent工具循环已经接通**。

**建议的跨模块逻辑消息**（规范目标，不冒充当前已存在的RPC方法）：

```json
{
  "schema": "voder.agent.action.v1",
  "session_id": "s-...",
  "run_id": "r-...",
  "action_id": "a-...",
  "kind": "patch_or_command",
  "workspace_scope": ["src/example.ts"],
  "expected_outcome": "targeted tests pass",
  "evidence_obligations": ["test_exit_zero", "protected_regression_pass"],
  "budget": {"max_tokens": 0, "max_runtime_ms": 30000},
  "idempotency_key": "opaque-key",
  "rollback_policy": "restore_authorized_preimage"
}
```

**协议集成要求**：先做Capabilities协商和版本化字段映射；每次执行将`Prepared/Authorized/Observed/Settled/Admitted/Audited/Committed`等状态映射为展示与决策事件；只有权威VGR响应才能产生最终交付令牌。扩展消息**不得直接跳过现有Daemon授权**，如必须增补RPC，应由正式Rust协议版本引入并跑原有F01—F09回归。

**证据和权限边界**：工作区前像哈希、用户授权、沙箱身份、真实命令及退出码、实际预算、回滚结果和Journal重放必须保持一致；Agent不得自行把`VERIFIED_STAGED_VGR_PENDING`当成`DELIVERED`。用户撤销授权、证据篡改、收据缺失或回滚不能确认时fail closed。

**冻结策略**：既有VGR底层当前版本按其固定验收范围冻结；Agent集成时发现真实兼容性Bug可单独修复并回归，但不能以“再建一个底层阶段”为前置条件。Adapter集成通过后仍须真实端到端产品验收。

## 12. 沙箱、部署和产品形态

**执行优先级**：默认受控本地工作树＋Docker/WSL隔离；需要更强隔离、云端资源或可重复远程验证时选择受控远程沙箱；无可用沙箱则停止有风险执行，不能默默降级为宿主机裸跑。沙箱的网络、文件系统、CPU、内存、时间、数据出入边界记录在执行计划和回执中。

**SDK为中心**：`voder-sdk`提供`createSession`、`submitTask`、`streamEvents`、`approve`、`cancel`、`resume`、`getDiff`、`getEvidence`、`exportRun`等稳定能力；错误码可区分缺文件、权限拒绝、沙箱失败、验证失败、预算耗尽和可恢复中断。

**CLI为第一产品**：支持`voder run`、`voder chat`、`voder resume`、`voder status`、`voder diff`、`voder test`、`voder approve`、`voder memory import`、`voder skills list`、`voder audit`、`voder doctor`；具体命令为目标CLI契约，而非声称目前均已实现。

**VS Code为第二入口**：扩展以SDK为唯一业务接口，允许提交所选代码/当前仓库上下文、在编辑器查看diff和诊断、显示VGR状态、确认变更及恢复会话；终端与扩展切换不丢任务，不做独立桌面App或重型Web后端。

## 13. 版本化事件、配置与工程布局

### 13.1 事件公共字段

`event_id`、`sequence`、`timestamp`、`session_id`、`run_id`、`task_node_id`、`type`、`status`、`source_component`、`causality_id`、`artifact_refs`、`vgr_ref`、`token_usage`、`cost_estimate`。事件可重放、有序、脱敏；网络掉线后从序号续传；不得把模型生成文本当作权威执行事实。

### 13.2 建议的代码布局（新增部分）

```text
crates/                      # 保留现有四个Rust项目与原生VGR
packages/
  voder-agent-core/          # 新增：Agent Loop与Session协调
  voder-context/             # 新增：黄金法则System Prompt、阶段增补Prompt与Context Assembly
  voder-task-planner/        # 新增：缺失检查与按需规划
  voder-search/              # 新增：本地索引、Wiki、动态检索
  voder-memory/              # 新增：可插拔记忆
  voder-skills/              # 新增：Skills注册和渐进加载
  voder-tools/               # 新增：统一工具代理
  voder-diagnostics/         # 新增：故障根因与复测策略
  voder-vgr-adapter/         # 新增：Rust协议桥接
  voder-sdk/                 # 新增：程序化入口
  voder-cli/                 # 扩展现有基础CLI
  voder-vscode/              # 新增：编辑器扩展
  voder-providers/           # 保留并复用
  voder-sandbox-modal/       # 保留并复用
  voder-orchestrator-*/      # 审计后复用或迁移，不双核运行
```

**配置原则**：项目与用户配置分层，默认安全、可本地运行；Provider、Search、Memory、Sandbox均为可插拔驱动；密钥仅在受保护环境/凭据管理器引用，配置文件不存明文Token；所有依赖锁定版本且记录上游许可证和版权声明。**蓝图不引用第三方框架名称，不代表源代码复用可以删除其合法许可证、原始版权声明或来源记录。**

## 14. 固定研发顺序与接口冻结点

此顺序属于**同一份固定蓝图的工作分解**，不是无限扩张的研发阶段；完成一项即冻结它的对外契约，发现缺陷归入原条目。

| 顺序 | 实施内容 | 必须交付的结果 |
|---|---|---|
| 01 | 现有主干适配、统一会话/事件、VGR Adapter | 真实会话可调用现有原生VGR，不存在绕开路径 |
| 02 | 黄金法则基础System Prompt、分阶段Prompt拼板、仓库检查、上下文生命周期 | 14条规则每轮保留；缺硬性文件能停问且阻断依赖动作；压缩不丢硬约束 |
| 03 | Tool Broker＋Skills | 模型自主选工具，写入/执行必经授权；Skill按需加载 |
| 04 | 任务规划、错误诊断、自动递归修复 | 真实多轮代码修复和失败停机可复现 |
| 05 | 外部LLM Wiki＋Agentic Search和可插拔记忆 | 按需启用、可溯源、可撤销、不浪费默认Token |
| 06 | voder-sdk＋voder-cli | 可提交/取消/恢复/审查/导出真实任务 |
| 07 | VS Code扩展 | 与CLI使用相同会话和验证结果 |
| 08 | 全链路真实任务和可靠性验收 | 满足第15节固定门禁才宣布该版本生产候选 |

**串并行规定**：API契约、测试用例和CLI原型可并行开发，但P0安全边界依赖必须先接通；外部搜索和记忆是能力增强，不阻止无网络的基础编程任务；不得在完成过程中临时增加“必须先研究的新模块”。

## 15. 固定验收标准（Voder Agent主体，不重测VGR旧成绩）

### 15.1 功能门禁——全部为发布必需（P0）

| ID | 验收场景 | PASS必须能观察到的行为 |
|---|---|---|
| A01 | 缺必要文件 | 首轮及前提变化激活预检System Prompt；Agent先检查真实目录/附件并搜索；确实缺硬性文件则进入`WAITING_FOR_INPUT`，指出路径/用途/提供方法；程序阻断依赖该文件的修改与交付，补文件后重新核验 |
| A02 | Prompt拼板 | 14条黄金法则作为不可遗漏基础System Prompt进入全部适用模型调用；阶段模板正确激活且互不污染，用户硬约束、证据和状态完整装配，超预算不静默截断 |
| A03 | 动态工具调用 | 模型至少能自主选读文件、搜索、执行测试、修补；写入/命令不可绕过VGR |
| A04 | Skills | 支持`SKILL.md`发现、按需加载和越权执行拒绝 |
| A05 | 自动修复 | 真实未知缺陷任务使用诊断阶段System Prompt实现“真实错报→根因假设→最小诊断/补丁→沙箱复测→反馈再修”；有可重复PASS与无新增证据时停止样例 |
| A06 | VGR产品级提交 | 真实工作树修改经过授权、收据、验证和正式提交；伪造回执/越权请求被拒绝 |
| A07 | 真实失败处理 | 测试失败、超时、预算耗尽、用户取消、重启后不错误交付、不重复有副作用操作 |
| A08 | 搜索与记忆 | 网络关闭仍能完成本地任务；按需搜索可给出处；导入记忆能召回/撤销且无泄密 |
| A09 | SDK与CLI | 用户终端完成真实任务提交、事件流、审批、diff、恢复及证据导出 |
| A10 | 编辑器接入 | VS Code提交真实任务并查看相同会话的diff/诊断/执行状态 |
| A11 | 并发与隔离 | 多会话任务无跨项目写入、越权、预算混账或交付串线 |
| A12 | 实际质量报告 | 固定真实仓库任务集、成功率/回归率/费用/Token/时延及失败分布可复测，不能只报模拟PASS |

### 15.2 九条原始原则的追踪映射

| 原始条款 | 对应实现 | 对应门禁 |
|---|---|---|
| 1 主动缺失检查 | Input Inspector | A01 |
| 2与4 Token节约、逐步选优（两条重复合并） | Step Selector＋Context Budget | A02/A05/A12 |
| 3 可插拔记忆与用户导入 | Memory Provider | A08 |
| 5 循证下一步与跨任务方法提升 | Agent Loop＋VGR＋Evidence Memory | A05/A06/A08 |
| 6 错报→修复→复测 | Diagnostics＋Sandbox | A05/A07 |
| 7 命令修补、验证、回退 | Tool Broker＋VGR | A03/A06/A07 |
| 8 自动追查可验证根因 | Diagnostics＋Adaptive Search | A05/A08 |
| 9 自动建环境、执行并修订 | Environment Adapter＋Agent Loop | A05/A07 |

### 15.3 通过与失败的报告规则

- **离线单元/契约测试**用于发现实现错误，不能代替真实API、真实沙箱和真实仓库任务。
- **Prompt回归检查（归属A01/A02/A05，不新增发布门禁）**：在任务入口、模型切换、检索、修复、压缩与恢复后验证14条基础规则仍在系统指令通道；阶段增补仅按触发加载；缺硬性文件时返回`WAITING_FOR_INPUT`且工具执行被程序拦截；用恶意仓库注释、Skill和记忆检验不能篡改System Prompt。
- **真实端到端**至少覆盖一次正常交付、一次多轮修复、一次无法修复、一次缺失硬文件、一次危险工具拒绝、一次超时/崩溃后恢复和跨会话隔离。
- **固定样本、固定环境、固定指标**：验收开始时冻结任务清单与模型/工具版本，公开成功数和失败数；生产性能指标在真实基线建立后锁定，禁止事后调阈值来制造PASS。
- **完成定义**：A01—A12全部达到明确的PASS证据，且没有已知阻断级缺陷，方可宣布**Voder Agent主体固定范围生产候选**。这不意味着超越任何其他产品，也不意味着所有未知问题都不存在。
- **范围控制**：新想法只进入下一版本Backlog，不修改本版出口；已验证模块只因真实回归缺陷才重开。

## 16. 当前状态与下一项工程动作

**已验证的历史成果**：原生VGR在既定F01—F09范围内具有真实Modal、写入失败、回滚、关键进程中断和有限并发的验收记录；现有Provider、基础CLI、沙箱与受限Orchestrator已有代码和历史测试。  
**不能声称的成果**：完整的模型动态工具循环、成熟Prompt拼板、Skills、Agentic Search＋Wiki、可插拔长期记忆、产品级VGR交付、正式SDK、正式VS Code扩展尚未作为一个真实产品完成验收。

**下一项工程动作**：以现有源码为只读基线，冻结主Agent事件契约和`voder-vgr-adapter`调用边界；并行建立第15节A01—A12的可执行验收清单，随后按第14节顺序实现。**不得先建第二套VGR，不得绕过现有Rust底层，也不得以局部测试PASS宣称整体完成。**
