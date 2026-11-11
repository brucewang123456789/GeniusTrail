"""Passive validation only. Never execute SVG code or resolve external resources."""
import sys, json, re, math
import xml.etree.ElementTree as ET
text=sys.stdin.read(131073)
if re.search(r'<!\s*(?:DOCTYPE|ENTITY)|<\?xml-stylesheet',text,re.I):
    raise ValueError('SVG size or external XML declaration')
root=ET.fromstring(text)
if root.tag.split('}')[-1]!='svg': raise ValueError('Expected SVG root')
ids=set();refs=[]
for e in root.iter():
    tag=e.tag.split('}')[-1].lower()
    if tag in ('script','foreignobject','image','feimage','iframe','audio','video','animate','animatetransform','animatemotion','set'):
        raise ValueError('Only static editable vector content is supported by this renderer')
    values=list(e.attrib.values())+([''.join(e.itertext())] if tag=='style' else [])
    for value in values:
        if '\\' in value: raise ValueError('Escaped CSS is unsupported by the isolated SVG preview')
        if re.search(r'@import|(?:https?|file|data|ftp):|//',value,re.I): raise ValueError('External SVG resource')
        for link in re.findall(r'url\(\s*[\'\"]?([^\)\'\"]+)',value,re.I):
            if not link.startswith('#'): raise ValueError('External SVG URL')
            refs.append(link[1:].strip())
    for key,value in e.attrib.items():
        name=key.split('}')[-1].lower()
        if name.startswith('on'):raise ValueError('SVG event handler')
        if name=='id':
            if not value or value in ids:raise ValueError('Empty/duplicate SVG ID')
            ids.add(value)
        if name in ('href','src'):
            if not value.startswith('#'):raise ValueError('External SVG reference')
            refs.append(value[1:])
if any(ref not in ids for ref in refs):raise ValueError('Unresolved SVG reference')
box=[float(x) for x in re.split('[ ,]+',root.attrib.get('viewBox','').strip())]
if len(box)!=4 or not all(math.isfinite(v) for v in box) or min(box[2:])<=0:raise ValueError('Valid viewBox required for preview')
width,height=round(box[2]),round(box[3])
if not 1<=width<=2048 or not 1<=height<=1536:raise ValueError('Preview dimensions exceed 2048x1536')
print(json.dumps({'width':width,'height':height,'viewBox':box,'unique_ids':len(ids),'references':len(refs),'title':any(e.tag.split('}')[-1]=='title' and ''.join(e.itertext()).strip() for e in root),'description':any(e.tag.split('}')[-1]=='desc' and ''.join(e.itertext()).strip() for e in root)}))
