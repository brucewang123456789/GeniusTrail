# Voder Agent — Rust VGR可信执行Gateway阶段交付

**定位：开发/云端联调版，不是生产级验收证书。**

## 真实代码与授权边界

- 正式源码路径：`/root/autodl-tmp/voder-agent/crates/voder-gateway`、`packages/voder-native-bridge`、`packages/voder-agent-core`、`packages/voder-sdk`。
- 第三方上游源码、Rust VGR已有核心四个crate **不修改**。
- 只有主机可信代码能设定Rust gateway二进制、操作员policy、原仓库位置、沙箱驱动及证据根目录。模型仅建议`kind=patch`、`relative_path`、`old_sha256`、`new_content`。
- 主机重写session/action/idempotency身份和预算；禁用Pi全部其他内置、扩展、隐形/递延工具。Rust再次检查确切文件范围、SHA256前像、干净Git树和可信验证器命令pin。
- Rust流程：创建隔离工作树→可信会话日志→Core冻结合同/义务/预算→prepare→fsync dispatch→写前像WAL→受控Patch→Modal沙箱观察→Runtime实测receipt→Core预算结算→固定外部验证器独立执行→密封证据→独立meta-audit→fsync final commit→重放确认。
- Node不相信工具文本：`VgrBroker`调用二次独立`voderd` journal/state/replay复核，只有唯一`vgr.final_committed`与所有身份/digest匹配才报告`VERIFIED`。
- 失败后不自动重试。Rust尝试`reconcile_interrupted_write`，崩溃场景由操作员以`--recover STATE_DIR SESSION RUN ACTION`执行，需精确身份。**被证明已提交的孤立工作树不会被恢复器擅自回滚。**
- 安全边界：目前只支持单次**现有普通文本文件替换**，且仅在独立detached Git worktree内进行，**不会把改动复制回用户原仓库**。将来推广需经过审查/原仓库合并准入门。当前没有通用shell/bash、任意写盘、跨文件变更权限。

## 云端安装

文件置于`/root/autodl-tmp/repair/`，在根目录执行：

```bash
cd /root/autodl-tmp
unzip -q -n repair/Voder_Agent_VGR_Trusted_Execution_Gateway_20261012.zip -d repair
python3 repair/Voder_Agent_VGR_Trusted_Execution_Gateway_20261012/scripts/install.py /root/autodl-tmp/voder-agent
RC=$?
echo "VODER_GATEWAY_INSTALL_RC=$RC"
echo 'CURRENT_TERMINAL_REMAINS_OPEN=True'
```

安装器：源文件SHA256、基线哈希、安全路径检查；只写既定的新/旧Agent主干目标；备份先行；真实`cargo build --offline --release`；33项Node安全回归；Pi/Rust只读云端联动；编译/测试失败恢复原文件并退出非零。真正的Modal测试**不在安装时自动花钱运行**。

## 云端真实执行

使用已设置的`VODER_MODAL_CLI`绝对路径、`MODAL_TOKEN_ID`、`MODAL_TOKEN_SECRET`、`VODER_MODAL_APP`和`VODER_MODAL_IMAGE`（必须为image@sha256:digest），**不要把凭据贴进聊天**。

先用无模型调用的真实Rust↔Modal↔验证↔回滚路径测试（有云端Modal费用）：

```bash
cd /root/autodl-tmp
source ./voder-env.sh
export VODER_ALLOW_MODAL_CHARGES=YES
node /root/autodl-tmp/voder-agent/scripts/gateway_live_smoke.mjs /root/autodl-tmp/voder-agent --native
RC=$?
echo "VODER_REAL_NATIVE_E2E_RC=$RC"
echo 'CURRENT_TERMINAL_REMAINS_OPEN=True'
```

成功后，再做**真实模型驱动工具调用**（需要Pi模型提供商的既有合法鉴权，并明确同意可能发生的API和Modal费用）：

```bash
cd /root/autodl-tmp
source ./voder-env.sh
export VODER_ALLOW_MODAL_CHARGES=YES
export VODER_ALLOW_MODEL_API=YES
node /root/autodl-tmp/voder-agent/scripts/gateway_live_smoke.mjs /root/autodl-tmp/voder-agent --model
RC=$?
echo "VODER_REAL_MODEL_VGR_E2E_RC=$RC"
echo 'CURRENT_TERMINAL_REMAINS_OPEN=True'
```

每次烟测会新建独立、干净的临时Git仓库、操作员固定policy和唯一证据路径，旧项目源码不会被更改。成功回显`RUST_VGR_AUTHORIZED_PATCH_AND_FINAL_COMMIT=PASS`或`REAL_MODEL_TO_NATIVE_VGR_FINAL_COMMIT=PASS`，失败返回非零并保留证据。远端模态沙箱不可达、模型没有凭据、Rust编译异常都必须记为失败，不能跳过或称完成。

## 实际验收门

- `NODE_REGRESSION`：此沙箱可执行，36/36 PASS。
- `RUST_COMPILE`：此沙箱没有`cargo`/`rustc`，**NOT_TESTED**，在AutoDL安装时必测。
- `REAL_NATIVE_MODAL_EXECUTION`：**NOT_TESTED**。
- `REAL_MODEL_PI_VGR_E2E`：**NOT_TESTED**。
- `SIGKILL_RECOVERY`/`CONCURRENCY`/`MULTI_FILE`/`PRODUCTION_SECURITY`：**NOT_TESTED/NOT_COMPLETE**。

此包解决真实接线实现的工程缺口，但**不宣称达到Codex/Claude Code等的生产成熟度**；真实跑通后仍需在失败/崩溃/多动作情形下补齐工程与评测。
