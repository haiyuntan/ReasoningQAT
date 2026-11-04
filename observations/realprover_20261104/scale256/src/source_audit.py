"""Read pinned papers and code; report actual matches rather than assumed scripts."""
import re
from common import OBS, atomic, read, sha

def main():
    acquisition=read(OBS/'manifests/acquisition.json')
    sections=['# Pinned source and figure audit','\nOnly Figure 2(b), conceptual REAL-Prover domain transfer. No calibration sweep or training.']
    patterns={'ReasoningQAT':r'tsne|t.sne|use_teacher_weight|topk|top_k','EfficientQAT':r'def train|def main|loss.backward','ReQAT':r'class SEMTrainer|quantile|sem_loss|entropy','REAL-Prover':r'build_local_prompt_str|build_local_incontext_prompt_str'}
    matches={}
    for name,pattern in patterns.items():
        source=OBS/'upstream'/name
        hits=[]
        for p in source.rglob('*.py'):
            if '.git' in p.parts:continue
            for n,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                if re.search(pattern,line,re.I):hits.append({'path':str(p.relative_to(OBS)),'line':n,'text':line.strip()[:220]})
        matches[name]=hits
        commit=acquisition.get('source:'+name,{}).get('result',{}).get('commit','UNKNOWN')
        sections += [f'\n## {name}\n\nCommit: `{commit}`.']
        for h in hits[:45]:sections.append(f'- `{h["path"]}:{h["line"]}`: `{h["text"]}`')
        if name=='ReasoningQAT' and not any(re.search('tsne|t.sne',h['text'],re.I) for h in hits):
            sections.append('\nFigure 2(b) extraction/t-SNE script: NOT_FOUND_IN_CHECKED_COMMIT. RECONSTRUCTION_NOT_EXACT_AUTHOR_SCRIPT; mean pooling is our preregistered choice.')
    sections.append('\n## PDF observations\n')
    for name in ['ReasoningQAT','EfficientQAT','ReQAT','REAL-Prover']:
        text=OBS/'papers'/(name+'.txt')
        if text.exists():
            lines=text.read_text().splitlines(); hits=[(i+1,l) for i,l in enumerate(lines) if re.search(r'Figure 2|14th|14-th|low.entropy|tail mass|75th|state.tactic',l,re.I)]
            sections.append(f'\n{name}: PDF SHA `{sha(OBS/"papers"/(name+".pdf"))}`. Extracted text evidence:')
            sections.extend(f'- text line {n}: {line.strip()}' for n,line in hits[:25])
        else:sections.append(f'{name}: SOURCE_BLOCKED, PDF text not parsed.')
    atomic(OBS/'source_audit/code_matches.json',matches)
    (OBS/'source_audit/methods_and_figures.md').write_text('\n'.join(sections)+'\n')

if __name__=='__main__':main()
