# Third-party software — license and attribution index

The root Apache License 2.0 applies to Voder-original source, not as a
replacement for the licenses that govern third-party works included here.
Preserve the original license texts, attribution, copyright and NOTICE files
when redistributing the corresponding material.

| Included material | Provenance / license | Original notice location |
| --- | --- | --- |
| Pi-derived coding agent engine | Pi, original author Mario Zechner, MIT | `voder-agent/engine/voder-agent-engine/LICENSE`; see `voder-agent/docs/VODER_UPSTREAM_ATTRIBUTION.md` |
| Playwright Core vendored runtime / browser integration | Microsoft Playwright contributors, Apache-2.0 | `voder-agent/packages/voder-vscode/src/vendor/Playwright-LICENSE.txt`; the dependency fetched during bootstrap also carries `package/LICENSE` and `package/NOTICE` |
| KaTeX vendored JavaScript | Khan Academy and other contributors, MIT | `voder-agent/packages/voder-vscode/src/vendor/KaTeX-LICENSE.txt` |
| libarchive test fixture license text | Copyright holders indicated in upstream COPYING file; license terms vary by file | `voder-agent/packages/voder-vscode/test/fixtures/third-party/libarchive-COPYING` |

Other dependencies obtained separately by npm or Cargo remain subject to
their original licenses; their versions are recorded in the checked-in lock
files. This inventory does not claim that an external API subscription grants
any rights to redistribute a service provider's software.

No upstream license text or attribution notice was edited in this release.
