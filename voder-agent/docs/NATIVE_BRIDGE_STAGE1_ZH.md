# Voder Pi-VGR通信边界第一阶段（开发版）

本补丁是独立、**只加文件**的真实VGR协议观察适配器，严禁将其声明为 `executeAuthorized`，也不授予Rust prepare/dispatch/commit权限。Pi v1.1.0已在用户AutoDL的 `packages/voder-agent-core/node_modules` 安装，未收录第三方Pi源码；本补丁不修改现有VGR Rust源码、journal、workspace或凭据。

## 安装和运行

将ZIP上传到 `/root/autodl-tmp/repair/`，运行：

```bash
cd /root/autodl-tmp
mkdir -p repair
unzip -q -n repair/Voder_Pi_VGR_Stdio_Bridge_Stage1_20261012.zip -d repair
B=/root/autodl-tmp/repair/Voder_Pi_VGR_Stdio_Bridge_Stage1_20261012
node --test "$B/packages/voder-native-bridge/test/observer.test.mjs"
bash "$B/scripts/run_on_autodl.sh" /root/autodl-tmp
```

`run_on_autodl.sh`仅在不存在debug daemon二进制时对现有源码运行 `cargo build -p voderd`（不改动源码）；所有真实daemon协议操作在 `mkdtemp` 的临时独立git库和状态目录中进行，结束即清理。此阶段不能跑模型写文件、不能提交VGR证据、没有端到端验证；下一阶段必须新增受保护的Rust原生executor宿主，接受模型的**无权限提议**并由Rust独立创建/验证VGR证书后授权执行，不能仅给stdio增加一个模型可传入的commit命令。

源码匹配当前用户提供的Rust VGR协议（v1 JSONL），非生产级验收；Node离线测试与云端真实rust gate分别报告。请保留之前Agent Kernel/SDK交接包，**不要**用本补丁覆盖或重新安装它。
