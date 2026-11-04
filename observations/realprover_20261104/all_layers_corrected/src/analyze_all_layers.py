"""All 28 BF16 layers; representative layer chosen before seeing 2D projection.

Discovery/confirmation here are grouped subsets of training-distribution inputs,
not independent corpora or held-out Lean theorem evaluation.
"""
import argparse
import concurrent.futures
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from common import OBS,read,atomic,sha

def normalized(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30)

def records():
    return [json.loads(s) for s in (OBS/'manifests/domain_inputs.jsonl').read_text().splitlines()]

def represent():
    rows=records();base=OBS/'metrics/raw_summaries/domains/P0'
    assert (base/'completed.json').exists()
    labels=np.array([r['domain'] for r in rows]);rng=np.random.default_rng(2)
    discovery=np.zeros(len(rows),dtype=bool)
    for domain in sorted(set(labels)):
        indices=np.flatnonzero(labels==domain);rng.shuffle(indices)
        discovery[indices[:len(indices)//2]]=True
    fields=[prefix+str(k) for k in range(1,29) for prefix in ['mean_','raw_mean_','last_']]
    arrays={key:[] for key in fields}
    def load(row):
        with np.load(base/(row['sample_id']+'.npz')) as data:return {key:data[key] for key in fields}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(load,rows):
            for key in fields:arrays[key].append(result[key])
    arrays={key:normalized(np.stack(values)) for key,values in arrays.items()}
    stats=[]
    for layer in range(1,29):
        row={'layer':layer}
        for prefix,tag in [('mean_','chat'),('raw_mean_','raw'),('last_','last')]:
            x=arrays[prefix+str(layer)]
            distance=cosine_distances(x);np.fill_diagonal(distance,0)
            row[tag+'_silhouette']=float(silhouette_score(distance,labels,metric='precomputed'))
            row[tag+'_discovery_silhouette']=float(silhouette_score(distance[np.ix_(discovery,discovery)],labels[discovery],metric='precomputed'))
            row[tag+'_confirmation_silhouette']=float(silhouette_score(distance[np.ix_(~discovery,~discovery)],labels[~discovery],metric='precomputed'))
            within=[];between=[]
            for domain in sorted(set(labels)):
                mask=labels==domain;n=mask.sum()
                within.append(distance[np.ix_(mask,mask)].sum()/(n*(n-1)))
                if domain!='Lean':
                    lean=labels=='Lean';row[tag+'_Lean_'+domain+'_distance']=float(distance[np.ix_(lean,mask)].mean())
            for a in sorted(set(labels)):
                for b in sorted(set(labels)):
                    if a<b:between.append(distance[np.ix_(labels==a,labels==b)].mean())
            row[tag+'_within_distance']=float(np.mean(within));row[tag+'_between_distance']=float(np.mean(between))
            if tag in ['chat','raw']:
                scores=cross_val_score(LogisticRegression(max_iter=1000),x,labels,cv=StratifiedKFold(5,shuffle=True,random_state=2))
                row[tag+'_probe_cv_mean']=float(scores.mean());row[tag+'_probe_cv_std']=float(scores.std(ddof=1))
        stats.append(row)
        pd.DataFrame(stats).to_csv(OBS/'metrics/all_layer_domain_statistics.csv',index=False)
        print('Layer statistics',layer,'/28',flush=True)
    table=pd.DataFrame(stats)
    selected=int(table.sort_values(['raw_discovery_silhouette','layer'],ascending=[False,True]).iloc[0].layer)
    winner=table[table.layer==selected].iloc[0].to_dict()
    atomic(OBS/'manifests/representative_layer.json',{'layer_one_based':selected,'block_zero_based':selected-1,
           'selection':'maximum discovery-half raw/no-chat high-dimensional cosine silhouette; tie to lower layer',
           'exploratory':True,'statistics':winner,'discovery_groups':int(discovery.sum()),'confirmation_groups':int((~discovery).sum()),
           'input_sha256':sha(OBS/'manifests/domain_inputs.jsonl'),'warning':'Training-distribution internal split, not independent replication; 28-layer selection induces selection bias.'})
    out=OBS/'plots/figure2_right';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for tag in ['chat','raw','last']:axes[0].plot(table.layer,table[tag+'_silhouette'],label=tag)
    for tag in ['chat','raw']:axes[1].plot(table.layer,table[tag+'_probe_cv_mean'],label=tag)
    for ax in axes:ax.axvline(selected,color='gray',linestyle='--');ax.set_xlabel('Decoder block (one based)');ax.legend()
    axes[0].set_ylabel('High-dimensional cosine silhouette');axes[1].set_ylabel('5-fold domain probe accuracy')
    fig.tight_layout();fig.savefig(out/'all_layer_domain_metrics.png',dpi=200);plt.close(fig)
    for layer in sorted({14,selected}):
        x=arrays['mean_'+str(layer)]
        pca=PCA(n_components=50,random_state=2).fit_transform(x)
        coordinates={seed:TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=seed,max_iter=1500).fit_transform(pca) for seed in [2,3]}
        xy=coordinates[2];fig,ax=plt.subplots(figsize=(8,6))
        for domain in ['General','Math','Code','Lean']:
            mask=labels==domain;ax.scatter(xy[mask,0],xy[mask,1],s=12,alpha=.7,label=domain)
        ax.set_title(f'Figure 2(b)-style: REAL-Prover BF16, Block {layer}')
        ax.set_xticks([]);ax.set_yticks([]);ax.legend(frameon=False)
        role='Exploratory selection by discovery raw high-dimensional silhouette' if layer==selected else 'Preregistered Block 14 reference'
        fig.text(.5,.01,f'{role}\n256 groups/domain; same 128-token windows; seed 2; Lean in training distribution',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(out/f'realprover_bf16_layer{layer}_tsne.png',dpi=220);plt.close(fig)
        pd.DataFrame({'sample_id':[r['sample_id'] for r in rows],'domain':labels,'discovery':discovery,
                      **{f'{axis}_seed{seed}':value[:,j] for seed,value in coordinates.items() for j,axis in enumerate(['x','y'])}}).to_csv(OBS/'metrics'/f'layer{layer}_tsne_coordinates.csv',index=False)
    (OBS/'reports/all_layer_representation.md').write_text('# All-layer representation\n\nAll 28 pre-final-norm block hooks. Same frozen 256 groups/domain and 128-token raw windows. Mean, last and raw/no-chat controls. Layer selection is exploratory on a seeded internal discovery half, using raw high-dimensional cosine silhouette. Confirmation half is reported without reselecting. This does not establish a knowledge module, semantic independence, or a corpus-held-out effect.\n\nSelected block: '+str(selected)+'. Layer 14 reference retained. Figure 2(b)-style reconstruction, not exact author plotting script.\n')

def quantization():
    rows=records();base=OBS/'metrics/raw_summaries/domains';values=[]
    for method in ['P3','P4','P6','P5']:
        if not (base/method/'completed.json').exists():continue
        for row in rows:
            with np.load(base/method/(row['sample_id']+'.npz')) as data:
                for layer in range(1,29):
                    nmse,cosine,relative=data['paired_'+str(layer)]
                    values.append({'method':method,'layer':layer,'domain':row['domain'],'sample_id':row['sample_id'],'nmse':nmse,'cosine':cosine,'relative_norm':relative})
    if not values:return
    table=pd.DataFrame(values);table.to_csv(OBS/'metrics/all_layer_quantization_samples.csv',index=False)
    summary=table.groupby(['method','layer','domain'])[['nmse','cosine','relative_norm']].agg(['mean','std','count'])
    summary.to_csv(OBS/'metrics/all_layer_quantization_statistics.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(method,group) in zip(axes.flat,table.groupby('method')):
        for domain,part in group.groupby('domain'):
            curve=part.groupby('layer').nmse.mean();ax.plot(curve.index,curve.values,label=domain)
        ax.set_title(method);ax.set_xlabel('Decoder block');ax.set_ylabel('Mean pooled activation NMSE');ax.legend()
    fig.tight_layout();fig.savefig(OBS/'plots/figure2_right/all_layer_quantization_nmse.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--quantization',action='store_true');args=parser.parse_args()
    quantization() if args.quantization else represent()
