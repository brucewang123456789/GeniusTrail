'use strict';
/** Presentation-only renderer. Must not mutate SDK messages, authorization, or evidence. */
function createAssistantPresentation(document, mathRenderer = null) {
  const MAX_DISPLAY = 131072;
  function unwrap(raw) {
    const original = String(raw == null ? '' : raw);
    let candidate = original.trim().replace(/^\uFEFF/, '');
    let didUnwrap = false;
    const fenced = candidate.match(/^```(?:json)?\s*\n([\s\S]*?)\n```\s*$/i);
    if (fenced) candidate = fenced[1].trim();
    for (let i = 0; i < 2; i++) {
      if (!candidate.startsWith('{') || !candidate.endsWith('}') || candidate.length > MAX_DISPLAY) break;
      let parsed;
      try { parsed = JSON.parse(candidate); } catch { break; }
      if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object' || typeof parsed.answer !== 'string') break;
      if (Object.keys(parsed).some(k => k !== 'answer' && k !== 'scope')) break;
      candidate = parsed.answer; didUnwrap = true;
    }
    return didUnwrap ? candidate : original;
  }
  function redact(raw) {
    let text = String(raw == null ? '' : raw);
    text = text.replace(/-----BEGIN (?:[A-Z ]* )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z ]* )?PRIVATE KEY-----/g, '[private key redacted]');
    text = text.replace(/\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|AIza[\w-]{30,}|xox[baprs]-[\w-]{15,})\b/g, '[credential redacted]');
    text = text.replace(/\b(Bearer\s+)[A-Za-z0-9._~+\/-]{16,}/gi, '$1[credential redacted]');
    text = text.replace(/\b((?:api[_-]?key|password|access[_-]?token|client[_-]?secret)\s*[:=]\s*["'])([^\s"']{8,})(["'])/gi, '$1[credential redacted]$3');
    text = text.replace(/(https?:\/\/)[^\s/@]+:[^\s/@]+@/gi, '$1[credentials redacted]@');
    // Local user/root prefixes are implementation details, not evidence-bearing basenames.
    text = text.replace(/\/(?:root|home\/[A-Za-z0-9._-]+)\/(?=[A-Za-z0-9._-])/g, '~/');
    text = text.replace(/\b[A-Za-z]:\\Users\\[^\\\s]+\\/g, '~\\');
    return text;
  }
  function element(tag, className) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }
  // Math is parsed only outside code. Each expression has fresh macros and bounded work.
  function mathToken(text, start) {
    let open, close, display;
    if (text.startsWith('\\(', start)) { open = '\\('; close = '\\)'; display = false; }
    else if (text.startsWith('\\[', start)) { open = '\\['; close = '\\]'; display = true; }
    else if (text.startsWith('$$', start)) { open = '$$'; close = '$$'; display = true; }
    else if (text[start] === '$' && !/\s/.test(text[start + 1] || ' ')) { open = '$'; close = '$'; display = false; }
    else return null;
    for (let end = start + open.length; end < Math.min(text.length, start + 8192); end++) {
      if (text[end] === '\\' && !text.startsWith(close, end)) { end++; continue; }
      if (!text.startsWith(close, end)) continue;
      const body = text.slice(start + open.length, end);
      if (!body.trim() || (open === '$' && (/\s$/.test(body) || /\d/.test(text[end + 1] || '')))) return null;
      return {body, display, end: end + close.length};
    }
    return null;
  }
  function appendMath(parent, token, original) {
    const node = element('span', token.display ? 'response-math response-math-display' : 'response-math');
    try {
      if (!mathRenderer || typeof mathRenderer.render !== 'function') throw Error('Math renderer unavailable');
      mathRenderer.render(token.body, node, {output: 'mathml', displayMode: token.display,
        throwOnError: true, trust: false, strict: 'error', maxExpand: 200, maxSize: 10, macros: {}});
    } catch {
      // Invalid or unsupported input stays readable; a formula must never erase the response.
      node.className = 'response-math-fallback'; node.textContent = original;
      node.title = 'Formula could not be typeset; original source preserved.';
    }
    parent.append(node);
  }
  function inline(parent, raw, depth = 0) {
    const text = redact(raw); let plain = '';
    const flush = () => { if (plain) { parent.append(document.createTextNode(plain)); plain = ''; } };
    for (let i = 0; i < text.length;) {
      if (text[i] === '`') {
        let end = i + 1; while (text[end] === '`') end++;
        const fence = text.slice(i, end); let close = text.indexOf(fence, end);
        while (close >= 0 && (text[close - 1] === '`' || text[close + fence.length] === '`')) close = text.indexOf(fence, close + fence.length);
        if (close >= 0) { flush(); const code = element('code'); code.textContent = text.slice(end, close); parent.append(code); i = close + fence.length; continue; }
        plain += fence; i = end; continue;
      }
      const token = mathToken(text, i);
      if (token) { flush(); appendMath(parent, token, text.slice(i, token.end)); i = token.end; continue; }
      if (text[i] === '\\' && ['\\', '$', '`'].includes(text[i + 1])) { plain += text.slice(i, i + 2); i += 2; continue; }
      const tail = text.slice(i);
      const bold = depth < 8 && tail.match(/^\*\*([^*\n]+)\*\*/);
      if (bold) { flush(); const node = element('strong'); inline(node, bold[1], depth + 1); parent.append(node); i += bold[0].length; continue; }
      const link = tail.match(/^\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/);
      if (link) {
        flush(); const node = element('a'); node.textContent = link[1];
        try { const url = new URL(link[2]); if (['https:', 'http:'].includes(url.protocol)) { node.href = url.href; node.rel = 'noopener noreferrer'; node.target = '_blank'; } } catch {}
        parent.append(node); i += link[0].length; continue;
      }
      plain += text[i++];
    }
    flush();
  }
  function tableCells(row) {
    const source = row.trim(), runs = [], next = new Map(), nearest = new Map();
    // Pair equal-length backtick runs in linear time. An unmatched run stays literal.
    for (let i = 0; i < source.length;) {
      if (source[i] !== '`') { i++; continue; }
      let end = i + 1;
      while (source[end] === '`') end++;
      runs.push({start: i, length: end - i}); i = end;
    }
    for (let i = runs.length - 1; i >= 0; i--) {
      const run = runs[i];
      if (nearest.has(run.length)) next.set(run.start, nearest.get(run.length));
      nearest.set(run.length, run.start);
    }
    const cells = []; let parts = [], codeEnd = -1, codeWidth = 0, lastDelimiter = -1;
    for (let i = 0; i < source.length;) {
      const ch = source[i];
      if (i === codeEnd) {
        parts.push(source.slice(i, i + codeWidth)); i += codeWidth; codeEnd = -1; continue;
      }
      if (codeEnd < 0) {
        const token = mathToken(source, i);
        if (token) { parts.push(source.slice(i, token.end)); i = token.end; continue; }
      }
      if (ch === '\\' && i + 1 < source.length) {
        const following = source[i + 1];
        if (following === '|') { parts.push('|'); i += 2; continue; }
        if (following === '\\' || (codeEnd < 0 && following === '`')) {
          parts.push(source.slice(i, i + 2)); i += 2; continue;
        }
      }
      if (codeEnd < 0 && ch === '`') {
        let end = i + 1;
        while (source[end] === '`') end++;
        if (next.has(i)) { codeEnd = next.get(i); codeWidth = end - i; }
        parts.push(source.slice(i, end)); i = end; continue;
      }
      if (codeEnd < 0 && ch === '|') {
        cells.push(parts.join('').trim()); parts = []; lastDelimiter = i;
      } else parts.push(ch);
      i++;
    }
    cells.push(parts.join('').trim());
    if (source.startsWith('|')) cells.shift();
    if (lastDelimiter === source.length - 1) cells.pop();
    return cells;
  }
  function render(raw, container) {
    const text = unwrap(raw).replace(/\r\n?/g, '\n').slice(0, MAX_DISPLAY);
    container.replaceChildren();
    const lines = text.split('\n');
    let i = 0;
    const isList = line => /^\s{0,3}(?:[-*+]\s+|\d{1,3}[.)]\s+)/.test(line);
    const isSpecial = line => /^(?:\s*\\\[|\s*\$\$)/.test(line) || /^\s{0,3}(?:#{1,4}\s+|>|```|~~~|---+\s*$)/.test(line) || isList(line);
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      let m = line.match(/^\s{0,3}(```|~~~)([A-Za-z0-9_+#.-]{0,32})\s*$/);
      if (m) {
        const fence = m[1], lang = m[2], code = []; i++;
        while (i < lines.length && !lines[i].trimStart().startsWith(fence)) code.push(lines[i++]);
        if (i < lines.length) i++;
        const wrap = element('div', 'response-code');
        if (lang) { const label = element('div', 'response-code-lang'); label.textContent = lang; wrap.append(label); }
        const pre = element('pre'), child = element('code'); child.textContent = redact(code.join('\n')); pre.append(child); wrap.append(pre); container.append(wrap); continue;
      }
      if (/^\s*(?:\\\[|\$\$)/.test(line)) {
        let combined = line.trim(), end = i + 1, token = mathToken(combined, 0);
        while (!token && end < lines.length && combined.length < 8192 && lines[end].trim() && !/^\s*(?:```|~~~)/.test(lines[end])) {
          combined += '\n' + lines[end++]; token = mathToken(combined, 0);
        }
        if (token && !combined.slice(token.end).trim()) {
          const block = element('div', 'response-math-block'); appendMath(block, token, combined); container.append(block); i = end; continue;
        }
      }
      m = line.match(/^\s{0,3}(#{1,4})\s+(.+)$/);
      if (m) { const h = element(m[1].length <= 2 ? 'h3' : 'h4'); inline(h, m[2]); container.append(h); i++; continue; }
      if (/^\s{0,3}---+\s*$/.test(line)) { container.append(element('hr')); i++; continue; }
      m = line.match(/^\s{0,3}>\s?(.*)$/);
      if (m) {
        const q = element('blockquote'); const collected = [];
        while (i < lines.length && /^\s{0,3}>/.test(lines[i])) collected.push(lines[i++].replace(/^\s{0,3}>\s?/, ''));
        collected.forEach((s, j) => { if (j) q.append(document.createElement('br')); inline(q, s); });
        container.append(q); continue;
      }
      // Render a pipe table only when the next line is a valid Markdown separator.
      if (/^\s*\|.+\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$/.test(lines[i + 1]) && tableCells(line).length === tableCells(lines[i + 1]).length) {
        const heads = tableCells(line), tableWrap = element('div', 'response-table-wrap'), table = element('table');
        const thead = element('thead'), hr = element('tr');
        heads.forEach(t => { const th = element('th'); inline(th, t); hr.append(th); });
        thead.append(hr); table.append(thead);
        const tbody = element('tbody'); i += 2;
        while (i < lines.length && /^\s*\|.+\|\s*$/.test(lines[i])) {
          const tr = element('tr'), rawRow = lines[i++], items = tableCells(rawRow);
          if (items.length > heads.length) {
            // Ambiguous rows must remain visible; never silently discard surplus columns.
            const td = element('td'), pre = element('pre'); td.colSpan = heads.length;
            pre.textContent = redact(rawRow); td.append(pre); tr.append(td);
          } else {
            for (let j = 0; j < heads.length; j++) { const td = element('td'); inline(td, items[j] || ''); tr.append(td); }
          }
          tbody.append(tr);
        }
        table.append(tbody); tableWrap.append(table); container.append(tableWrap); continue;
      }
      if (isList(line)) {
        const ordered = /^\s{0,3}\d/.test(line), list = element(ordered ? 'ol' : 'ul');
        const numberOf = value => { const match = value.match(/^\s{0,3}(\d{1,3})[.)]\s+/); return match ? Number(match[1]) : null; };
        let expected = ordered ? numberOf(line) : null;
        if (ordered) list.start = expected;
        while (i < lines.length) {
          // Markdown allows blank lines between items of the same list. Preserve explicit numbering
          // and avoid merging an intentionally restarted ordered list into its predecessor.
          if (!lines[i].trim()) {
            let next = i;
            while (next < lines.length && !lines[next].trim()) next++;
            if (next >= lines.length || !isList(lines[next]) || /^\s{0,3}\d/.test(lines[next]) !== ordered) break;
            if (ordered && numberOf(lines[next]) !== expected) break;
            i = next;
          }
          if (!isList(lines[i]) || /^\s{0,3}\d/.test(lines[i]) !== ordered) break;
          const sourceNumber = ordered ? numberOf(lines[i]) : null;
          const li = element('li');
          if (ordered) { li.value = sourceNumber; expected = sourceNumber + 1; }
          inline(li, lines[i++].replace(/^\s{0,3}(?:[-*+]\s+|\d{1,3}[.)]\s+)/, ''));
          list.append(li);
        }
        container.append(list); continue;
      }
      const p = element('p');
      while (i < lines.length && lines[i].trim() && (i === 0 || !isSpecial(lines[i]) || p.childNodes.length === 0)) {
        if (p.childNodes.length) p.append(document.createElement('br'));
        inline(p, lines[i++]);
        if (i < lines.length && isSpecial(lines[i])) break;
      }
      container.append(p);
    }
    if (!container.childNodes.length) { const empty = element('p'); empty.textContent = 'No response content returned.'; container.append(empty); }
    return container;
  }
  return {render, redact, unwrap};
}
module.exports = {createAssistantPresentation};
