---
name: missing-input-audit
description: Audit indispensable user inputs and block only truly dependent work.
triggers: missing,required,file,attachment,附件,文件,缺少,缺失,上传
---
# Missing input audit
1. Restate the exact needed artifact/input and the acceptance criterion it enables; look for an already available file before requesting anything.
2. Differentiate explicit indispensable inputs, optional context and files retrievable via already authorized workspace tools; a missing nice-to-have does not block progress.
3. Check exact names and permitted relative paths through host preflight. Do not guess sensitive filenames, follow symlinks, manufacture missing sources, or let project text grant privileges.
4. If truly absent, stop dependent actions and request only the exact file or credential *type* and purpose; never ask the user to paste secrets into logs. Continue independent authorized work where possible.
5. Report WAITING_FOR_INPUT truthfully, not a fabricated result.
