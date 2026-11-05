"""No-PCA seed/convergence checks and PCA-neighborhood audit; CPU only."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from explore import ROOT,OUT,save,render
def main():
    save('robustness_protocol.json',{'source':'same2048 token vectors as v3 factorial','no_pca_seed_controls':[2,3],
        'initialization':'random','learning_rate':200,'perplexity':30,'seed_check_iterations':1000,
        'longer_run':{'seed':42,'iterations':3000},'neighborhood_audit_k':15,
        'warning':'Exploratory method diagnosis, not original-author configuration discovery.'})
    rows=[json.loads(s) for s in (ROOT/'fig2b_qwen3_v2_20261105/inputs.jsonl').read_text().splitlines()]
    labels=np.repeat(np.array([r['domain'] for r in rows]),4)
    with np.load(ROOT/'fig2b_qwen3_v2_20261105/calibration_chat_features.npz') as data:x=data['block_out_token'][:,:4].reshape(-1,2048)
    pca=PCA(n_components=50,random_state=2);z=pca.fit_transform(x)
    neighbors=[]
    for value in [x,z]:
        index=NearestNeighbors(n_neighbors=16,algorithm='brute',n_jobs=2).fit(value).kneighbors(value,return_distance=False)
        neighbors.append([set(row[row!=i][:15]) for i,row in enumerate(index)])
    var=lambda a:float(np.mean(np.sum((a-a.mean(0))**2,axis=1)))
    audit={'pca50_explained_variance_fraction':float(pca.explained_variance_ratio_.sum()),
        'mean_original_vs_pca50_15nn_overlap':float(np.mean([len(a&b)/15 for a,b in zip(*neighbors)])),
        'original_high_dimensional_reasoning_to_pretraining_dispersion':var(x[labels!='Pre-training'])/var(x[labels=='Pre-training']),
        'pca50_reasoning_to_pretraining_dispersion':var(z[labels!='Pre-training'])/var(z[labels=='Pre-training']),
        'warning':'KL values from t-SNE before and after PCA optimize different neighborhood graphs; do not compare them as a common accuracy score.'}
    save('pca_neighborhood_audit.json',audit)
    results=[];reference=json.loads((OUT/'reference_audit.json').read_text())
    for seed,iterations in [(2,1000),(3,1000),(42,3000)]:
        fit=TSNE(init='random',learning_rate=200,perplexity=30,random_state=seed,max_iter=iterations,n_jobs=2)
        xy=fit.fit_transform(x);name=f'nopca_random_lr200_seed{seed}_iter{iterations}'
        transform=render(xy,labels,name,reference)
        results.append({'tag':name,'seed':seed,'iterations':iterations,'kl_divergence':float(fit.kl_divergence_),
            'two_dimensional_reasoning_to_pretraining_dispersion':var(xy[labels!='Pre-training'])/var(xy[labels=='Pre-training']),
            'display_transform':transform})
        save('robustness_results.json',results);print('DONE',name,flush=True)
    save('robustness_completed.json',{'status':'DONE','additional_tsne_runs':3})
if __name__=='__main__':main()
