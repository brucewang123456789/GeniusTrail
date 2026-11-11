# Voder Skills — third module

Trusted **installation**, pinned SHA-256 integrity and task-scoped progressive loading of portable `SKILL.md` documents. The **existing** `voder-context` package remains the single source of truth for built-in core/strategy/diagnosis Skills; this package is an additive user-supplied Skill seam, not a competing prompt loader.

- Only explicitly supplied absolute `SKILL.md` paths are installed; no implicit project-global Skill autoload and no download/install of scripts.
- Admin install: `node packages/voder-skills/src/manage-skills.mjs install /ABS/SKILL.md SHA256`; `list` and `remove` available.
- The store location defaults to `/root/.voder/installed-skills`, override via `VODER_INSTALLED_SKILLS_ROOT` to an administrator-owned directory. Never store secrets in SKILL.md.
- The SDK host supplies `skillStore: new InstalledSkills({root:'/ABS/PATH'})` and, optionally, `explicitSkills:['skill-name']`.
- Skill content reaches the task input marked `UNTRUSTED_INSTALLED_SKILL`; **never** becomes privileged System Prompt or a tool grant. Required Skills must load whole or stop. Auto selections only match relevant names; no global default Token charge.
- External executable Skill scripts are **not** supported here; all commands remain behind the Tool Broker, Sandbox and Rust VGR gates.
