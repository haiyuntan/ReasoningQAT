"""Use measured PDF heading positions and locally decoded embedded glyphs.

Font binaries remain local. Coordinates are never moved per domain.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import re,json,warnings
from pathlib import Path
from explore import ROOT,OUT,IDS,save
import fitz
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties

def heading_font(document,page):
    record=next(f for f in document.get_page_fonts(2,full=True) if 'YuGothic-Bold' in f[3])
    name,ext,kind,buffer=document.extract_font(record[0]);local=OUT/'reference_heading_font.ttf';local.write_bytes(buffer)
    font=TTFont(local);old=font['cmap'].tables[0].cmap
    kind,ref=document.xref_get_key(record[0],'ToUnicode');assert kind=='xref'
    mapping=document.xref_stream(int(ref.split()[0])).decode('ascii')
    unicode_map={}
    for first,last,target in re.findall(r'<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>',mapping):
        for offset,code in enumerate(range(int(first,16),int(last,16)+1)):
            if code in old:unicode_map[int(target,16)+offset]=old[code]
    needed=set('Dataset DomainPre-trainingCodeMathScience')
    assert all(ord(c) in unicode_map for c in needed)
    for platform,encoding in [(0,3),(3,1)]:
        table=CmapSubtable.newSubtable(4);table.platformID=platform;table.platEncID=encoding;table.language=0;table.cmap=unicode_map.copy();font['cmap'].tables.append(table)
    fixed=OUT/'reference_heading_unicode.ttf';font.save(fixed)
    return fixed,{'source':'Embedded font subset from original PDF; ToUnicode CMap remapped to Unicode for local rendering',
                  'original_font':name,'required_characters_present':True,'binary_published':False}

def main():
    # Matplotlib measures the synthetic string "lp" for baseline metrics.
    # Those unused letters are absent from this subset; all actual label letters
    # are explicitly verified above. Do not add invented glyphs to the font.
    warnings.filterwarnings('ignore',message=r'Glyph (108|112) .*missing from font',category=UserWarning)
    doc=fitz.open(ROOT/'papers/reasoningqat_iclr2026.pdf');page=doc[2]
    font,font_audit=heading_font(doc,page)
    evidence=json.loads((OUT/'reference_audit.json').read_text());rect=evidence['embedded_image_pdf_rect'];ratio=488/(rect[2]-rect[0])
    spans=[]
    for b in page.get_text('dict')['blocks']:
        for line in b.get('lines',[]):
            for s in line['spans']:
                if s['text'] in ['Dataset Domain','Pre-training','Code','Math','Science'] and s['bbox'][0]>320:spans.append(s)
    lines=[d for d in page.get_drawings() if d['rect'].x0>320 and d['rect'].y0>80 and d['rect'].y1<103]
    assert len(spans)==5 and len(lines)==5
    out=OUT/'paper_style';out.mkdir(exist_ok=True);checks=[]
    for coordinate in sorted((OUT/'panels').glob('*.npz')):
        name=coordinate.stem
        with np.load(coordinate) as data:xy=data['display_xy'];labels=data['labels']
        metadata=json.loads(coordinate.with_suffix('.json').read_text())
        fig=plt.figure(figsize=(4.88,4.07),dpi=100);ax=fig.add_axes([49/488,33/407,372/488,309/407]);cax=fig.add_axes([445/488,33/407,16/488,309/407])
        order=np.concatenate([np.flatnonzero(labels==d) for d in ['Code','Math','Science','Pre-training']])
        scatter=ax.scatter(xy[order,0],xy[order,1],c=np.array([IDS[d] for d in labels[order]]),cmap='viridis',norm=Normalize(0,5),alpha=.2,s=36,linewidths=.5)
        if metadata['points_outside_display_axes'] is not None:
            ax.set_xlim(-40,40);ax.set_ylim(-40,40);ax.set_xticks(np.arange(-40,41,20));ax.set_yticks(np.arange(-40,41,10))
            assert not (np.abs(xy)>40.000001).any()
        ax.tick_params(labelsize=10,width=.8)
        for spine in ax.spines.values():spine.set_linewidth(.8)
        cb=fig.colorbar(scatter,cax=cax,ticks=range(6));cb.ax.tick_params(labelsize=10,width=.8)
        for s in spans:
            x=(s['origin'][0]-rect[0])*ratio;baseline=(s['origin'][1]-rect[1])*ratio+50
            color='#'+format(s['color'],'06x')
            props=FontProperties(fname=str(font),size=s['size']*ratio*72/100)
            fig.text(x/488,1-baseline/407,s['text'],fontproperties=props,color=color,ha='left',va='baseline')
        for d in lines:
            r=d['rect'];left=(r.x0-rect[0])*ratio;top=(r.y0-rect[1])*ratio+50
            fig.add_artist(plt.Rectangle((left/488,1-(top+r.height*ratio)/407),r.width*ratio/488,r.height*ratio/407,color=d['fill'],linewidth=0,transform=fig.transFigure))
        for ext in ['png','pdf','svg']:
            destination=out/(name+'.'+ext)
            fig.savefig(destination,dpi=200)
            if ext=='svg':
                # Matplotlib path attributes include trailing spaces by default.
                original=destination.read_text()
                normalized='\n'.join(line.rstrip() for line in original.splitlines())+'\n'
                assert original.split()==normalized.split()
                destination.write_text(normalized)
        transformed=metadata['points_outside_display_axes'] is not None
        annotation='Global orientation + uniform display scaling; native coordinates retained' if transformed and metadata['angle_radians'] else ('Uniform display scaling only; native coordinates retained' if transformed else 'Native t-SNE coordinates; no display transform')
        fig.text(.5,.008,annotation,ha='center',va='bottom',fontsize=4.5,color='#444444')
        fig.savefig(out/(name+'_annotated.png'),dpi=200);plt.close(fig)
        checks.append({'file':name,'points':len(xy),'display_axis_clipped_points':0 if transformed else None})
    save('exact_style_validation.json',{'status':'PASS','font':font_audit,'pdf_based_heading_spans':len(spans),'pdf_based_underlines':len(lines),
        'rendered_panels':len(checks),'checks':checks,'limits':'Underlying scatter marker size/order/alpha and categorical ID meanings remain hypotheses, not recovered author code.'})
if __name__=='__main__':main()
