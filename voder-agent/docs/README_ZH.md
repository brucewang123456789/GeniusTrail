# Voder Agent Kernel + voder-sdk · 开发版交接包

**这是完整的现有源码交接，不是生产版发布。** 此ZIP包括用户提交的VGR底层源码快照、先前已经写出的Agent Kernel＋SDK源码、对应离线测试、双语正式蓝图及安全部署指令；源码不会因为当前沙箱无法跑真实远端而被扣留。

## 一次性部署（AutoDL）

1. **先手动上传本ZIP到** `/root/autodl-tmp/repair/`；不要解压覆盖现有源码。
2. 执行以下命令，脚本只会新增两个目录，不修改原VGR，并在模块已经存在时拒绝覆盖：

```bash
cd /root/autodl-tmp
mkdir -p repair
unzip -q -n repair/Voder_Agent_Kernel_SDK_Development_Handoff_20261012.zip -d repair
bash repair/Voder_Agent_Kernel_SDK_Development_Handoff_20261012/scripts/install_to_autodl.sh /root/autodl-tmp
RC=$?
echo "VODER_AGENT_SDK_DEV_INSTALL_RC=$RC"
echo "CURRENT_TERMINAL_REMAINS_OPEN=True"
```

如要只跑压缩包内部现有测试，不安装：

```bash
cd /root/autodl-tmp/repair/Voder_Agent_Kernel_SDK_Development_Handoff_20261012
bash scripts/test_offline.sh
```

## 官方Agent依赖核对（**不代表可运行整链**）

已核对官方GitHub `https://github.com/earendil-works/pi` 的 `v1.1.0` Tag和npm包名 `@earendil-works/pi-coding-agent@1.1.0`；真实Node版本要求不低于22.19.0。

当前包**不会自动下载安装**官方Agent或处理密钥（避免在云端失控修改用户环境）；待准备好独立Node依赖安装与外部API后，可以安装并运行只读的 `node scripts/probe_pi_api.mjs`，确认其导出的函数名，但该检查仍不等于模型/API/Modal联调。

## 明确的阻断项

- `executeAuthorized`只有Node接口约定，尚未可靠对接原生Rust VGR授权/提交；仅测试替身。
- Agent工具schema与Pi动态行为、真实模型、Modal回滚、沙箱联调未验收。
- voder-sdk仅覆盖开发版基本会话方法；不应宣称已实现完整版流式SDK。
- **10/10 Node测试通过不意味着完成生产级验收，也不允许真实执行绕过VGR。**

完整状态请阅读 `docs/ACCEPTANCE_STATUS.md`；绝不在当前安装中清理用户任何原文件。
