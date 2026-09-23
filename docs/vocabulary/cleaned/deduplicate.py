"""Extract source entries and merge duplicate headwords, preserving every source variant."""
from pathlib import Path
import subprocess, xml.etree.ElementTree as ET, re, unicodedata, json
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
NS={'h':'http://www.w3.org/1999/xhtml'}
records=[]
for level,filename in [('B2','B2-Aspekte-neu-Alphabetische-Wortliste.pdf'),('C1','C1-Aspekte-neu-Wortliste.pdf')]:
 xml=subprocess.check_output(['pdftotext','-bbox-layout',str(ROOT/filename),'-'])
 doc=ET.fromstring(xml)
 for page_no,page in enumerate(doc.findall('.//h:page',NS),1):
  for left,right,refleft in ([(70,290,225),(315,550,475)] if level=='B2' else [(70,295,235),(315,550,480)]):
   words=[w for w in page.findall('.//h:word',NS) if left<=float(w.get('xMin'))<right and float(w.get('yMin'))<770]
   refs=[w for w in words if float(w.get('xMin'))>=refleft and re.fullmatch(r'\d+/(?:[A-Za-z0-9/]+)',w.text or '')]
   refs.sort(key=lambda w:float(w.get('yMin')))
   for i,ref in enumerate(refs):
    y=float(ref.get('yMax')); end=float(refs[i+1].get('yMax'))-.5 if i+1<len(refs) else 780
    content=[w for w in words if float(w.get('xMin'))<refleft and y-.5<=float(w.get('yMax'))<end]
    content.sort(key=lambda w:(round(float(w.get('yMax')),0),float(w.get('xMin'))))
    text=' '.join(w.text or '' for w in content)
    assert text,(level,page_no,ref.text)
    records.append(dict(level=level,text=text,source=f'{filename}, page {page_no}, {ref.text}'))
for line in (ROOT/'C2-RadicalRampage-wordlist.txt').read_text().splitlines():
 if re.match(r'^\d+ \|',line):
  num,word,translation=line.split(' | ',2)
  records.append(dict(level='C2',text=word,translation=translation,source=f'RadicalRampage entry {num}'))
def headword(s):
 s=s.replace('Kommunikationsmanage- ment','Kommunikationsmanagement').replace('Zusammengehörigkeits- gefühl','Zusammengehörigkeitsgefühl')
 s=re.sub(r'^(?:der/die|der|die|das)\s+','',s)
 s=re.split(r'[,;(]',s,1)[0].strip()
 return unicodedata.normalize('NFC',s)
groups={}
for rec in records:
 h=headword(rec['text']); key=h.casefold()
 assert h and not re.search(r'\d',h),(h,rec)
 rec['headword']=h
 if key not in groups: groups[key]={'headword':h,'level':rec['level'],'variants':[]}
 groups[key]['variants'].append(rec)
for level in ['B2','C1','C2']:
 selected=[v for v in groups.values() if v['level']==level]
 lines=[f'{level}: unique headwords', 'Repeated headwords across levels occur only in the earliest source level (B2, then C1, then C2).', 'All source examples, forms and translations are retained as variants. Source levels are textbook/site labels, not independently verified CEFR assignments.','']
 for g in selected:
  lines.append(g['headword'])
  seen=set()
  for r in g['variants']:
   detail=r['text']+(' | '+r['translation'] if 'translation' in r else '')
   if detail not in seen: lines.append('  '+detail); seen.add(detail)
  lines.append('')
 (ROOT/'cleaned'/f'{level}-unique.txt').write_text('\n'.join(lines))
(ROOT/'cleaned'/'entries-with-provenance.json').write_text(json.dumps(list(groups.values()),ensure_ascii=False,indent=2))
counts=Counter(r['level'] for r in records); unique=Counter(g['level'] for g in groups.values())
report=['# Duplicate audit','', 'Method: Unicode NFC normalization and case-insensitive headword comparison. Leading noun articles and text following commas or parentheses are excluded from the matching key. Distinct phrases remain separate. Shared headwords are assigned to the earliest source level; all variants and provenance are retained. Original PDFs and downloaded text are unchanged.','','| Level | Source entries | Unique headwords assigned |','|---|---:|---:|']
for level in ['B2','C1','C2']: report.append(f'| {level} | {counts[level]} | {unique[level]} |')
report+=['',f'Total: {len(records)} source entries merged into {len(groups)} headword records. {len(records)-len(groups)} repeated occurrences merged.','', '## Repeated headwords','']
for key,g in groups.items():
 if len(g['variants'])>1: report.append(f"- {g['headword']}: "+', '.join(r['level'] for r in g['variants']))
(ROOT/'cleaned'/'AUDIT.md').write_text('\n'.join(report)+'\n')
assert sum(len(g['variants']) for g in groups.values())==len(records)
assert len({headword(g['headword']).casefold() for g in groups.values()})==len(groups)
print(counts,unique,'Merged occurrences:',len(records)-len(groups))
print('Longest headwords:',sorted((g['headword'] for g in groups.values()),key=len)[-12:])
