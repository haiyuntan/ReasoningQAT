"""Report the full fixed comparison grid; no similarity-based selection."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
sys.path.append(str(OUT.parent/'observation_20261104_all_layers/envs/data/lib/python3.11/site-packages'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
DOMAINS=['Pre-training','Code','Math','Science']
COLORS=['#e2cf18','#6a3495','#425fa3','#2495a7']

def dispersion(x):
    return float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
def grid_stats(base,labels):
    stats=[]
    for key,value in base.items():
        shape=value.shape;token=len(shape)==3
        raw=value.reshape(-1,shape[-1]).astype(np.float64)
        y=np.repeat(labels,shape[1]) if token else labels
        for norm in ['none','l2']:
            x=raw if norm=='none' else raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
            pre=dispersion(x[y=='Pre-training']);pooled=dispersion(x[y!='Pre-training'])
            stats.append(dict(feature=key,norm=norm,pretraining_dispersion=pre,pooled_reasoning_dispersion=pooled,
                              pooled_reasoning_to_pretraining=pooled/pre,
                              individual_domain_ratio={d:dispersion(x[y==d])/pre for d in DOMAINS[1:]}))
    return stats
def bootstrap_main(x,labels):
    # Resample independent source groups, retaining within-source token dependence.
    x=x.astype(np.float64);means=x.mean(1);norms=np.square(x).sum(2).mean(1)
    rng=np.random.default_rng(2);ids={d:np.flatnonzero(labels==d) for d in DOMAINS}
    ratios=[]
    for _ in range(1000):
        draw={d:rng.choice(i,len(i),replace=True) for d,i in ids.items()}
        pre=draw['Pre-training'];reason=np.concatenate([draw[d] for d in DOMAINS[1:]])
        var=lambda a:float(norms[a].mean()-np.square(means[a].mean(0)).sum())
        ratios.append(var(reason)/var(pre))
    return {'resampling':'1000 source-group bootstrap, stratified domains, seed2',
            'pooled_reasoning_to_pretraining_95ci':np.quantile(ratios,[.025,.975]).tolist()}
def main():
    rows=[json.loads(s) for s in (OUT/'inputs.jsonl').read_text().splitlines()]
    labels=np.array([r['domain'] for r in rows]);allstats=[]
    for view in ['calibration_chat','raw_conversation','response_only']:
        with np.load(OUT/(view+'_features.npz')) as data:
            group=grid_stats(data,labels)
            for r in group:r['view']=view
            allstats.extend(group)
            if view=='calibration_chat':bootstrap=bootstrap_main(data['block_out_token'],labels)
    (OUT/'pooled_dispersion.json').write_text(json.dumps({'statistics':allstats,'primary_bootstrap':bootstrap},indent=2))
    # Reassess v1 with the same pooled estimand: within-domain vs union differs.
    old=OUT.parent/'fig2b_qwen3_20261105'
    oldlabels=np.array([json.loads(s)['domain'] for s in (old/'inputs.jsonl').read_text().splitlines()])
    with np.load(old/'features.npz') as data:oldstats=grid_stats(data,oldlabels)
    (OUT/'v1_pooled_reanalysis.json').write_text(json.dumps(oldstats,indent=2))
    # All views and locations, fixed token/raw/seed2; randomized painter order.
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    for i,view in enumerate(['calibration_chat','raw_conversation','response_only']):
        for j,hook in enumerate(['block_in','block_out','mlp_in']):
            p=OUT/'plots'/f'{view}_{hook}_token_none_seed2.npz'
            with np.load(p) as data:xy=data['xy'];y=data['labels']
            order=np.random.default_rng(2).permutation(len(y));color=np.array([COLORS[DOMAINS.index(d)] for d in y])
            ax=axes[i,j];ax.scatter(xy[order,0],xy[order,1],s=3,alpha=.4,c=color[order],linewidths=0)
            ax.set(xticks=[],yticks=[],title=f'{view}\n{hook}',aspect='equal')
    fig.legend(handles=[Line2D([],[],marker='o',color=c,linestyle='',label=d) for d,c in zip(DOMAINS,COLORS)],loc='upper center',ncol=4,frameon=False)
    fig.suptitle('Qwen3-1.7B | Block 14 | Fixed calibration-aligned comparisons',y=.96)
    fig.text(.5,.01,'All panels: 128 groups/domain, 2048 tokens/input, 16 sampled tokens/group, PCA50, t-SNE seed2\nDifferent panels have unrelated axes. Author plotting protocol unavailable.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(OUT/'comparison_grid.png',dpi=180);plt.close(fig)
    lines=['# 修正版结果：完整问答、固定2048 token\n',
           '固定第14层，128组/域；已验证 hook 位置及14层截断与完整前向一致。主图预先指定，无挑选层或随机种子。\n',
           '域内方差与“将三个推理域合并后的总体方差”不是同一个指标。以下同时报告两者；后者也包含推理域之间的均值差异，不能单独推断每一域内部都更分散。\n',
           '## 固定主表：逐token、无归一化\n',
           '| 输入 | 位置 | 推理合并/预训练离散度 | Code/预训练 | Math/预训练 | Science/预训练 |',
           '|---|---|---:|---:|---:|---:|']
    for r in allstats:
        if r['norm']=='none' and r['feature'].endswith('_token'):
            ratios=r['individual_domain_ratio'];lines.append(f"| {r['view']} | {r['feature']} | {r['pooled_reasoning_to_pretraining']:.3f} | "+' | '.join(f'{ratios[d]:.3f}' for d in DOMAINS[1:])+' |')
    lines+=['\n主图合并推理/预训练离散度的按源组 bootstrap 95% CI：'+str(bootstrap['pooled_reasoning_to_pretraining_95ci']),
            '\n![固定比较网格](comparison_grid.png)',
            '\n![预先指定主图](plots/calibration_chat_block_out_token_none_seed2.png)',
            '\n局限：Fig2b的原始绘图脚本未公开；沿用官方校准输入构造不等于已知原图就是该构造。推理样本为长回答子集，FineWeb仍按顺序采样。t-SNE不能保留所有全局距离，图形相似不等于科学结论复现。统计区间仅反映当前样本组的重采样，不包含语料来源选择不确定性。',
            '\n[全部高维统计](pooled_dispersion.json) · [旧版合并推理指标重分析](v1_pooled_reanalysis.json) · [层位置验证](hook_validation.json)']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
