"""CPU-only t-SNE controls and disclosed paper-style presentation transforms."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import sys,json,hashlib,time,itertools
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.append(str(ROOT/'observation_20261104_all_layers/envs/data/lib/python3.11/site-packages'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
import fitz
from PIL import Image

DOMAINS=['Pre-training','Code','Math','Science']
IDS={'Pre-training':5,'Code':0,'Math':1,'Science':2}
LABEL_RGB={'Pre-training':'#d2bf1f','Code':'#440054','Math':'#414487','Science':'#2a788e'}
def save(name,value):
    path=OUT/name;path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit_reference():
    document=fitz.open(ROOT/'papers/reasoningqat_iclr2026.pdf');page=document[2]
    # Locate the embedded panel using its visible x/y axis/colorbar aspect.
    candidates=[]
    for item in page.get_images(full=True):
        if item[2:4]==(488,357):candidates.append(item)
    assert len(candidates)==1
    item=candidates[0];image=document.extract_image(item[0]);rect=page.get_image_rects(item[0])[0]
    (OUT/'paper_scatter_reference.jpeg').write_bytes(image['image'])
    spans=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                if span['text'] in DOMAINS+['Dataset Domain'] and span['bbox'][0]>320:
                    spans.append({key:span[key] for key in ['text','bbox','font','size','color']})
    assert len(spans)==5
    pixels=np.array(Image.open(OUT/'paper_scatter_reference.jpeg').convert('RGB'))
    ys=np.arange(25,315);observed=pixels[ys,451].astype(float)
    predicted=plt.colormaps['viridis'](1-(ys-15)/309)[:,:3]*255
    alpha=float(np.sum((observed-255)*(predicted-255))/np.sum((predicted-255)**2))
    fit=255*(1-alpha)+alpha*predicted
    evidence={'embedded_image_size':[488,357],'embedded_image_pdf_rect':list(rect),
        'pdf_sha256':sha(ROOT/'papers/reasoningqat_iclr2026.pdf'),'label_spans':spans,
        'axes':{'x_limits':[-40,40],'x_tick_step':20,'y_limits':[-40,40],'y_tick_step':10},
        'colorbar':{'ticks':[0,1,2,3,4,5],'cmap':'viridis','estimated_alpha':alpha,'fit_rgb_rmse':float(np.sqrt(np.mean((observed-fit)**2))),
            'interpretation':'Domain-color encoding inferred from caption and matching hues; numeric semantics not explained by authors. Not entropy or activation magnitude.',
            'inferred_reproduction_ids':IDS,'ids_3_and_4':'Unused in our four-domain data; retained only to match the visible original 0..5 bar.'},
        'font':'Original headings use YuGothic-Bold. Local export uses DejaVu Sans; font glyphs are not exact.',
        'scatter_alpha':{'value':.2,'status':'Hypothesis from common scatter-colorbar alpha linkage; only bar alpha is measured.'},
        'marker_size':{'value':36,'status':'Matplotlib default-sized marker hypothesis; original value unavailable.'}}
    save('reference_audit.json',evidence)
    return evidence

def display_transform(xy,labels,orient=True,scale=True):
    xy=np.asarray(xy,dtype=np.float64);rotation=np.eye(2);reflected=False
    if orient:
        vector=xy[labels=='Pre-training'].mean(0)-xy[labels!='Pre-training'].mean(0)
        theta=-np.pi/2-np.arctan2(vector[1],vector[0])
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        rotated=xy@rotation.T
        if rotated[labels=='Code',0].mean()>rotated[labels=='Science',0].mean():
            rotation=np.diag([-1,1])@rotation;reflected=True
    else:theta=0
    rotated=xy@rotation.T
    center=(rotated.min(0)+rotated.max(0))/2 if scale else np.zeros(2)
    factor=38/np.max(np.abs(rotated-center)) if scale else 1.
    result=(rotated-center)*factor
    rng=np.random.default_rng(2);a=rng.integers(len(xy),size=256);b=rng.integers(len(xy),size=256)
    old=np.linalg.norm(xy[a]-xy[b],axis=1);new=np.linalg.norm(result[a]-result[b],axis=1)
    error=float(np.max(np.abs(new-old*factor)));assert error<1e-10
    if scale:assert np.max(np.abs(result))<=38.00000001
    return result,{'orientation':'Pre-training below pooled reasoning; one global optional reflection places Code left of Science' if orient else 'native t-SNE orientation',
        'angle_radians':float(theta),'reflected':reflected,'orthogonal_matrix':rotation.tolist(),'subtract_center_after_rotation':center.tolist(),
        'uniform_scale':float(factor),'random_pair_distance_max_abs_error':error,'points_outside_display_axes':int((np.abs(result)>40).any(1).sum()) if scale else None,
        'warning':'Presentation transform only, fitted from category means. Not evidence of independent coordinate replication. No per-domain translation/scaling or point removal.'}

def render(xy,labels,name,evidence,orient=True,scale=True,mixed_order=False):
    panel=OUT/'panels';panel.mkdir(exist_ok=True)
    shown,transform=display_transform(xy,labels,orient,scale)
    save('panels/'+name+'.json',transform)
    np.savez_compressed(panel/(name+'.npz'),native_xy=xy,display_xy=shown,labels=labels)
    # Pixel geometry measured from original 488x357 image plus its PDF headings.
    width=488;height=407;header=50
    fig=plt.figure(figsize=(width/100,height/100),dpi=100)
    ax=fig.add_axes([49/width,33/height,372/width,309/height])
    cax=fig.add_axes([445/width,33/height,16/width,309/height])
    if mixed_order:order=np.random.default_rng(2).permutation(len(xy))
    else:order=np.concatenate([np.flatnonzero(labels==d) for d in ['Code','Math','Science','Pre-training']])
    scatter=ax.scatter(shown[order,0],shown[order,1],c=np.array([IDS[d] for d in labels[order]]),
                       cmap='viridis',norm=Normalize(0,5),s=36,alpha=.2,linewidths=.5)
    if scale:
        ax.set_xlim(-40,40);ax.set_ylim(-40,40);ax.set_xticks(np.arange(-40,41,20));ax.set_yticks(np.arange(-40,41,10))
    ax.tick_params(labelsize=10,width=.8)
    for spine in ax.spines.values():spine.set_linewidth(.8)
    bar=fig.colorbar(scatter,cax=cax,ticks=np.arange(6));bar.ax.tick_params(labelsize=10,width=.8)
    rect=evidence['embedded_image_pdf_rect'];ratio=488/(rect[2]-rect[0]);texts=[]
    for span in evidence['label_spans']:
        x=(span['bbox'][0]-rect[0])*ratio
        top=(span['bbox'][1]-rect[1])*ratio+header
        font=span['size']*ratio*72/100
        text=span['text'];artist=fig.text(x/width,1-top/height,text,ha='left',va='top',fontsize=font,
            fontweight='bold',fontfamily='DejaVu Sans',color=LABEL_RGB.get(text,'black'))
        texts.append(artist)
    # Original PDF colored headings are underlined; match with measured text widths.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in texts:
        box=text.get_window_extent(renderer=renderer)
        fig.add_artist(plt.Line2D([box.x0/width,box.x1/width],[(box.y0-1)/height]*2,
                                 color=text.get_color(),linewidth=.6,transform=fig.transFigure))
    for ext in ['png','pdf','svg']:fig.savefig(panel/(name+'.'+ext),dpi=200)
    # A labeled version makes display transforms explicit when shared independently.
    fig.text(.5,.008,'Display-aligned: global rotation/reflection + uniform scale' if orient and scale else
             ('Display-scaled only; native t-SNE orientation' if scale else 'Native t-SNE coordinates; no presentation transform'),
             ha='center',va='bottom',fontsize=5,color='#444444')
    fig.savefig(panel/(name+'_annotated.png'),dpi=200);plt.close(fig)
    return transform

def main():
    evidence=audit_reference()
    protocol={'inputs':'v2 calibration_chat block_out_token, first4 of16 uniformly sampled tokens per source; same128 groups/domain',
        'point_count':2048,'feature_normalization':'none','seed':42,'iterations':1000,
        'factorial':{'pca_components':[None,50],'init':['pca','random'],'learning_rate':['auto',200],'perplexity':[30]},
        'extra_perplexity_controls':[{'pca':None,'init':'random','learning_rate':200,'perplexity':100},
                                      {'pca':50,'init':'pca','learning_rate':'auto','perplexity':100}],
        'frozen_primary':'Existing v2 seed2 primary stays primary; no score-based winner selected',
        'presentation':'Measured header/axes/bar geometry; explicit orthogonal orientation + one common scale for all points; native-coordinate controls retained.',
        'versions':{'sklearn':sklearn.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__},'script_sha256':sha(Path(__file__))}
    save('protocol.json',protocol)
    rows=[json.loads(s) for s in (ROOT/'fig2b_qwen3_v2_20261105/inputs.jsonl').read_text().splitlines()]
    labels=np.repeat(np.array([r['domain'] for r in rows]),4)
    with np.load(ROOT/'fig2b_qwen3_v2_20261105/calibration_chat_features.npz') as data:values=data['block_out_token'][:,:4].reshape(-1,2048)
    with np.load(ROOT/'fig2b_qwen3_v2_20261105/plots/calibration_chat_block_out_token_none_seed2.npz') as data:
        oldxy=data['xy'];oldlabels=data['labels']
        render(oldxy,oldlabels,'v2_primary_paper_style',evidence)
        render(oldxy,oldlabels,'v2_primary_unoriented',evidence,orient=False)
        render(oldxy,oldlabels,'v2_primary_native_coordinates',evidence,orient=False,scale=False)
    cases=[]
    for pca,init,lr in itertools.product([None,50],['pca','random'],['auto',200]):
        cases.append(dict(pca=pca,init=init,lr=lr,perplexity=30))
    cases.extend([dict(pca=None,init='random',lr=200,perplexity=100),dict(pca=50,init='pca',lr='auto',perplexity=100)])
    pca_values=PCA(n_components=50,random_state=2).fit_transform(values);results=[]
    for case in cases:
        tag=f"pca{case['pca']}_init{case['init']}_lr{case['lr']}_perp{case['perplexity']}"
        started=time.monotonic();z=values if case['pca'] is None else pca_values
        fit=TSNE(perplexity=case['perplexity'],init=case['init'],learning_rate=case['lr'],random_state=42,max_iter=1000,n_jobs=2)
        xy=fit.fit_transform(z)
        transform=render(xy,labels,tag,evidence)
        var=lambda a:float(np.mean(np.sum((a-a.mean(0))**2,axis=1)))
        results.append({'tag':tag,**case,'elapsed_s':time.monotonic()-started,'kl_divergence':float(fit.kl_divergence_),
            'native_bounds':[xy.min(0).tolist(),xy.max(0).tolist()],
            'two_dimensional_reasoning_to_pretraining_dispersion':var(xy[labels!='Pre-training'])/var(xy[labels=='Pre-training']),
            'display_transform':transform})
        save('results.json',results);print('DONE',tag,flush=True)
    save('completed.json',{'status':'DONE','tsne_controls':len(results),'primary':'panels/v2_primary_paper_style_annotated.png',
        'exact_replication':False,'reason':'Author figure script, selected samples, activation definition and seed remain unknown.'})
if __name__=='__main__':main()
