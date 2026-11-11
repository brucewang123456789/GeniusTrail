"""Pinned structural acceptance only. Does not execute JavaScript or certify visuals."""
from pathlib import Path
from html.parser import HTMLParser
import re

class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.scripts = []
        self.styles = []
        self.canvas = 0
        self.title = 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        assert tag not in ('base', 'iframe', 'object', 'embed', 'form'), 'unsupported embedded authority'
        for key, value in attrs.items():
            assert not key.lower().startswith('on'), 'use app.js event listeners'
            if key in ('src', 'href', 'action', 'poster', 'srcset'):
                assert value in ('app.js', './app.js', 'style.css', './style.css') or (value or '').startswith(('#', 'data:image/')), 'external or unknown resource'
        if tag == 'script':
            assert attrs.get('src') in ('app.js', './app.js'), 'all scripts must use app.js'
            self.scripts.append(attrs['src'])
        if tag == 'link':
            assert attrs.get('rel') == 'stylesheet' and attrs.get('href') in ('style.css', './style.css'), 'stylesheet must use style.css'
            self.styles.append(attrs['href'])
        if tag == 'title': self.title += 1
        if tag in ('canvas', 'svg'): self.canvas += 1
    def handle_startendtag(self, tag, attrs): self.handle_starttag(tag, attrs)

def check():
    files = {}
    for name in ('index.html', 'app.js', 'style.css'):
        p = Path(name)
        assert p.is_file() and not p.is_symlink(), 'missing regular ' + name
        raw = p.read_bytes()
        assert 32 <= len(raw) and b'\x00' not in raw, 'invalid size or NUL: ' + name
        files[name] = raw.decode('utf-8')
    html = files['index.html']
    assert re.search(r'<!doctype\s+html\s*>', html, re.I), 'HTML5 doctype required'
    assert re.search(r'<head\b', html, re.I) and re.search(r'</body\s*>', html, re.I), 'explicit head and body required'
    assert re.search(r'<html\b', html, re.I) and re.search(r'</html\s*>', html, re.I), 'complete HTML required'
    page = Page(); page.feed(html); page.close()
    assert len(page.scripts) == len(page.styles) == 1 and page.title == 1, 'one app.js, stylesheet and title required'
    assert page.canvas >= 1, 'canvas or SVG drawing surface required'
    assert 'requestAnimationFrame' in files['app.js'], 'animation loop required'
    assert re.search(r'@import|url\s*\(\s*[\x22\x27]?\s*(?:https?:|//|file:)', files['style.css'], re.I) is None, 'external stylesheet resource'
    print('PASS: HTML/resource structure, bounded UTF-8 files and animation entry. JS execution and visuals NOT tested.')

if __name__ == '__main__': check()
