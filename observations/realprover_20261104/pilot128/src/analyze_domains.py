"""BF16 Figure 2(b) only, high-dimensional diagnostics, paired W2 tables."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,StratifiedKFold
from common import OBS, atomic, read

def main():
    rows=[json.loads(s) for s in (OBS/'manifests/domain_inputs.jsonl').read_text().splitlines()]
    base=OBS/'metrics/raw_summaries/domains'
    if not (base/'P0/completed.json').exists():raise RuntimeError('BF16 not complete')
    features=np.stack([np.load(base/'P0'/(r['sample_id']+'.npz'))['mean_14'] for r in rows])
    labels=np.array([r['domain'] for r in rows]);normalized=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    reduced=PCA(n_components=50,random_state=2).fit_transform(normalized)
    embeddings={seed:TSNE(n_components=2,perplexity=30,random_state=seed,init='pca',learning_rate='auto',max_iter=1500).fit_transform(reduced) for seed in [2,3]}
    metrics=[]
    for seed,embedding in embeddings.items():
        metrics.append({'metric':'tsne_silhouette','seed':seed,'value':silhouette_score(embedding,labels)})
    metrics.append({'metric':'high_dim_cosine_silhouette','seed':2,'value':silhouette_score(normalized,labels,metric='cosine')})
    probe=cross_val_score(LogisticRegression(max_iter=2000),normalized,labels,cv=StratifiedKFold(5,shuffle=True,random_state=2))
    metrics.append({'metric':'domain_probe_cv_accuracy','seed':2,'value':float(probe.mean())})
    for a in sorted(set(labels)):
        for b in sorted(set(labels)):
            if a>=b:continue
            x=normalized[labels==a];y=normalized[labels==b];gamma=1.
            value=rbf_kernel(x,x,gamma=gamma).mean()+rbf_kernel(y,y,gamma=gamma).mean()-2*rbf_kernel(x,y,gamma=gamma).mean()
            metrics.append({'metric':'biased_rbf_mmd2_'+a+'_'+b,'seed':2,'value':float(value)})
    pd.DataFrame(metrics).to_csv(OBS/'metrics/domain_statistics.csv',index=False)
    out=OBS/'plots/figure2_right';out.mkdir(exist_ok=True,parents=True)
    fig,ax=plt.subplots(figsize=(8,6))
    for domain in ['General','Math','Code','Lean']:
        mask=labels==domain;xy=embeddings[2][mask];ax.scatter(xy[:,0],xy[:,1],s=16,alpha=.75,label=domain)
    ax.set_title('Figure 2(b)-style: REAL-Prover BF16, Block 14')
    ax.set_xticks([]);ax.set_yticks([]);ax.legend(frameon=False)
    fig.text(.5,.01,'Pilot: 128 source groups/domain; same 128-token window + chat wrapper; seed 2\nConceptual reconstruction; Lean is in training distribution; t-SNE is not a knowledge map',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(out/'realprover_bf16_layer14_tsne.png',dpi=220);plt.close(fig)
    pd.DataFrame({'sample_id':[r['sample_id'] for r in rows],'domain':labels,'x_seed2':embeddings[2][:,0],'y_seed2':embeddings[2][:,1],'x_seed3':embeddings[3][:,0],'y_seed3':embeddings[3][:,1]}).to_csv(OBS/'metrics/tsne_coordinates.csv',index=False)
    comparisons=[]
    for method in ['P3','P4','P6','P5','P1','P2']:
        if not (base/method/'completed.json').exists():continue
        for row in rows:
            with np.load(base/'P0'/(row['sample_id']+'.npz')) as teacher,np.load(base/method/(row['sample_id']+'.npz')) as student:
                for layer in [1,7,14,21,28]:
                    t=teacher['mean_'+str(layer)].astype(float);q=student['mean_'+str(layer)].astype(float)
                    comparisons.append({'method':method,'domain':row['domain'],'sample_id':row['sample_id'],'layer':layer,'nmse':float(np.sum((q-t)**2)/max(np.sum(t*t),1e-12)),'cosine':float(np.dot(q,t)/max(np.linalg.norm(q)*np.linalg.norm(t),1e-12)),'relative_norm':float(np.linalg.norm(q)/max(np.linalg.norm(t),1e-12))})
    if comparisons:pd.DataFrame(comparisons).to_csv(OBS/'metrics/domain_sensitivity.csv',index=False)
    # Raw/no-template control and last-position sensitivity use original high-dimensional space.
    controls=[]
    for key in ['raw_mean_14','last_14']:
        x=np.stack([np.load(base/'P0'/(r['sample_id']+'.npz'))[key] for r in rows])
        x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        score=cross_val_score(LogisticRegression(max_iter=2000),x,labels,cv=StratifiedKFold(5,shuffle=True,random_state=2))
        controls.append({'pooling':key,'domain_probe_cv_accuracy':float(score.mean())})
    pd.DataFrame(controls).to_csv(OBS/'metrics/domain_template_controls.csv',index=False)
    (OBS/'reports/02_fig2_right_method.md').write_text('# Figure 2(b) reconstruction\n\nBF16 REAL-Prover, actual decoder block hook 13 (one-based 14), pre-final-norm. Mean pooling, same 128-token raw-text windows with identical qwen wrapper; pilot 128 groups/domain. Raw/no-wrapper and last-position controls in CSV. Shared combined L2 normalization → PCA(50) → t-SNE(perplexity30, seeds2/3, 1500 iterations). Only seed2 plotted; seed3 coordinates and statistics retained.\n\nRECONSTRUCTION_NOT_EXACT_AUTHOR_SCRIPT. Lean is training-distribution. Non-Lean first-eligible sequential source selection and raw-code punctuation remain confounders; probe CV is source-group separated but not source-corpus held-out. Geometry does not establish a separate knowledge module. No training or Figure2(a).\n')
    atomic(OBS/'manifests/figure_gate.json',{'status':'PILOT_DONE','samples':len(rows)})

if __name__=='__main__':main()
