from pathlib import Path
from PIL import Image
import fitz,json,hashlib
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
source=ROOT/'papers/reasoningqat_iclr2026.pdf';doc=fitz.open(source)
item=next(i for i in doc[2].get_images(full=True) if i[2:4]==(590,462))
a=doc.extract_image(item[0]);original=OUT/('paper_fig2a.'+a['ext']);original.write_bytes(a['image'])
paths=[original,ROOT/'fig2b_qwen3_v3_20261105/paper_reference_with_heading.png',ROOT/'representation_expanded_v4_20261105/qwen/plots/qwen_calibration_chat_expanded512x4_seed42.png',ROOT/'representation_expanded_v4_20261105/real/plots/real_raw128_expanded512x4_seed42.png']
w,h,gap=1440,1204,32
canvas=Image.new('RGB',(w*2+gap,h*2+gap),'white');records=[]
for i,p in enumerate(paths):
 im=Image.open(p).convert('RGB');scale=min(w/im.width,h/im.height);size=(round(im.width*scale),round(im.height*scale));resized=im.resize(size,Image.Resampling.LANCZOS)
 x=(i%2)*(w+gap)+(w-size[0])//2;y=(i//2)*(h+gap)+(h-size[1])//2;canvas.paste(resized,(x,y))
 records.append({'panel':i,'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_size':list(im.size),'display_size':list(size),'paste_xy':[x,y]})
canvas.save(OUT/'fig2_four_panel_expanded.png',dpi=(200,200));canvas.save(OUT/'fig2_four_panel_expanded.pdf',resolution=200)
(OUT/'provenance.json').write_text(json.dumps({'layout':'top: paper Fig2a, paper Fig2b; bottom: Qwen expanded512x4 noPCA seed42, REAL-Prover five-domain expanded512x4 raw128 seed42','sample_scope':'Lower panels use512 groups/domain ×4tokens: Qwen8192points and REAL10240points. All27 expanded-study plots complete.','operation':'Aspect-preserving resize, white padding; no added text; no cropped plot content','panels':records,'size':list(canvas.size)},indent=2))
print(OUT/'fig2_four_panel_expanded.png')
