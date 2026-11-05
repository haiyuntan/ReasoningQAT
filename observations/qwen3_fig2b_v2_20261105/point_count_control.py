"""Explicit point-count sensitivity; does not replace the frozen primary panel."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import sys,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
sys.path.append(str(OUT.parent/'observation_20261104_all_layers/envs/data/lib/python3.11/site-packages'))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    rows=[json.loads(s) for s in (OUT/'inputs.jsonl').read_text().splitlines()]
    labels=np.array([r['domain'] for r in rows]);domains=['Pre-training','Code','Math','Science']
    with np.load(OUT/'calibration_chat_features.npz') as data:values=data['block_out_token']
    controls=OUT/'point_count_controls';controls.mkdir(exist_ok=True)
    protocol={'primary_unchanged':'8192 points (16/source), fixed seed2',
              'controls':[{'tokens_per_source':4,'seeds':[2,3,42]},{'tokens_per_source':8,'seeds':[2]}],
              'selection':'first4/8 of16 already uniformly sampled positions; all128 sources/domain retained',
              'justification':'Original point count not published; t-SNE topology depends on the number of points and neighborhood scale.'}
    (controls/'protocol.json').write_text(json.dumps(protocol,indent=2));stats=[]
    for n,seeds in [(4,[2,3,42]),(8,[2])]:
        x=values[:,:n].reshape(-1,values.shape[-1]);y=np.repeat(labels,n)
        z=PCA(n_components=50,random_state=2).fit_transform(x)
        for seed in seeds:
            xy=TSNE(perplexity=30,init='pca',learning_rate='auto',random_state=seed,max_iter=1500,n_jobs=2).fit_transform(z)
            tag=f'calibration_chat_block_out_{len(x)}points_seed{seed}'
            np.savez_compressed(controls/(tag+'.npz'),xy=xy,labels=y)
            fig,ax=plt.subplots(figsize=(5,5))
            order=np.random.default_rng(2).permutation(len(y));colors=np.array([['#e2cf18','#6a3495','#425fa3','#2495a7'][domains.index(d)] for d in y])
            ax.scatter(xy[order,0],xy[order,1],s=9,alpha=.45,c=colors[order],linewidths=0)
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([],[],marker='o',color=c,linestyle='',label=d) for d,c in zip(domains,['#e2cf18','#6a3495','#425fa3','#2495a7'])],loc='upper center',ncol=2,frameon=False,fontsize=8)
            ax.set(xticks=[],yticks=[],title='Qwen3-1.7B BF16 | Block 14')
            fig.text(.5,.01,f'Point-count sensitivity | {len(x)} points | seed{seed}\nSame128 sources/domain, calibration chat; primary unchanged',ha='center',fontsize=7)
            fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(controls/(tag+'.png'),dpi=180);plt.close(fig)
            var=lambda a:float(np.square(a-a.mean(0)).sum(1).mean())
            stats.append({'points':len(x),'seed':seed,'two_dimensional_reasoning_to_pretraining_dispersion':var(xy[y!='Pre-training'])/var(xy[y=='Pre-training'])})
            (controls/'statistics.json').write_text(json.dumps(stats,indent=2));print(tag,flush=True)
    (controls/'completed.json').write_text(json.dumps({'status':'DONE','panels':4}))
if __name__=='__main__':main()
