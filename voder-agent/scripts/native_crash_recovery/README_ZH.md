# Voder T08 原生Rust SIGKILL回滚验收：Node真实路径 + PASS后根目录同步

## 已确认的根因

旧测试把`/root/autodl-tmp/.voder-tools/node/bin/node`（符号链接）直接传给原生Rust`PluginSandbox::new`，而Rust安全契约明确要求`VODER_NODE_BINARY`指向绝对路径的普通文件，因此旧测试提前退出并非崩溃恢复失败。

新版只将**传给Rust的Node路径**解析为`.voder-tools`目录内经过校验的真实、可执行普通文件；Cargo仍以原始`cargo`代理路径调用，绝不解析成`rustup`去执行。Modal TypeScript插件本身必须为普通文件，且不需要执行位。

## 验收范围和根目录同步

在AutoDL云端运行`run_t08_only.sh`：先进行正反测试和源码哈希冻结检查，离线构建已有原生Rust示例，然后分别在`POST_PATCH`与`POST_META_AUDIT`检查点启动专用示例，在**临时Git仓库**触发真实SIGKILL，由**正式Rust Gateway**恢复回滚，并验证旧值、Git干净状态、Journal中无最终提交、没有授权交付、重复恢复被拒绝。

**只有两个阶段全部PASS且CRASH.json中的原始Journal哈希与恢复身份同时通过交叉校验**，才会将这份已验证的测试脚本安装到正式根目录的`voder-agent/scripts/native_crash_recovery/`；任何失败均不安装。此目录属于**验收/测试脚本**，不是对原生VGR、Pi、SDK或CLI源码的变更；若已有不同内容则拒绝覆盖。

**SDK自动续跑依然NOT_SUPPORTED**；本次验收只能证明原生Rust中断安全回滚，不代表SDK已能自动重新规划并继续执行，也不能代替24小时原生稳定性测试。

## AutoDL运行

将ZIP上传`/root/autodl-tmp/repair/`，校验ZIP SHA256并解压后执行：

```bash
cd /root/autodl-tmp
VODER_ALLOW_MODAL_CHARGES=YES VODER_ENABLE_NATIVE_SIGKILL=YES bash repair/Voder_T08_Node_Realpath_And_RootSync_20261013/run_t08_only.sh
```

如失败，查看输出的`EVIDENCE_ROOT`中的`T08-run.log`、`rust-build.log`、`CRASH.json`及两个独立case目录。不要重跑任何旧的已提交动作。
