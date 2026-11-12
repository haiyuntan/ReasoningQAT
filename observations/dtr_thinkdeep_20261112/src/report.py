"""Frozen-result plots, paired same-token shifts and honest coverage reports."""
from common import *
def main():
    import numpy as np,collections
    from run import bootstrap
    source={p.name:read(p) for p in (W/'metrics/per_tactic').glob('*.json') if read(p).get('status')=='DONE'}
    matched=[]
    for p in (W/'metrics/paired_replay').glob('*.json'):
        q=read(p);r=source[p.name]
        assert q['generated_token_ids_sha']==r['generated_token_ids_sha'] and q['input_ids_sha']==r['input_ids_sha']
        if q['dtr'] is not None and r['dtr'] is not None:matched.append(dict(theorem_id=r['theorem_id'],split=r['split'],seed=r['seed'],P4_dtr=r['dtr'],P0_same_token_dtr=q['dtr'],delta_dtr=r['dtr']-q['dtr'],Lean_status=r['lean_status']))
    shifts={}
    for split in ['D-DEV','D-CONFIRM']:
        ss=[r for r in matched if r['split']==split];theorems=collections.defaultdict(list)
        for r in ss:theorems[r['theorem_id']].append(r['delta_dtr'])
        shifts[split]=dict(estimate=bootstrap([sum(v)/len(v) for v in theorems.values()]),matched_tactics=len(ss),origin='P4 only; P0 replays identical tokens',interpretation='DTR shift is not capacity or proof-success change')
    save(W/'metrics/summaries/paired_shift.json',shifts);jsonl(W/'metrics/summaries/paired_shift_per_tactic.jsonl',matched)
    primary=read(W/'metrics/summaries/D-CONFIRM.json');dev=read(W/'metrics/summaries/D-DEV.json')
    save(W/'metrics/summaries/primary.json',primary)
    save(W/'metrics/summaries/secondary.json',dict(dev=dev,paired_shift=shifts,P0_natural='NOT_RUN_TIME',P3_P6_P5='NOT_RUN_TIME',intermediate_rollouts='NOT_RUN_TIME',proof_search='NOT_RUN',ranking='NOT_RUN_UNLESS_PRIMARY_SUPPORT',sensitivity='raw, bits, g/rho and persistent values per tactic; exploratory'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Illustration is fixed first DEV theorem/first seed, never chosen by label.
    c=rows(W/'manifests/frozen_cases.jsonl')[0];key=digest(['P4',c['theorem_id'],protocol()['seeds'][0]])+'.json';r=source.get(key)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    if r:
        t=[x for x in read(W/r['token_file']) if not x['stop_flag']];qpath=W/'metrics/paired_replay'/key
        if t:
            axs[0,0].imshow(np.array([x['jsd'] for x in t]).T,aspect='auto',vmin=0,vmax=np.log(2));axs[0,0].axhline(22.5,color='white',lw=.7);axs[0,0].set_title('P4 normalized lens: first frozen DEV sample')
        if qpath.exists() and read(qpath)['tokens']:
            qt=read(qpath)['tokens'];axs[0,1].imshow(np.array([x['jsd'] for x in qt]).T,aspect='auto',vmin=0,vmax=np.log(2));axs[0,1].axhline(22.5,color='white',lw=.7);axs[0,1].set_title('P0 normalized lens: identical P4 token sequence')
    for split in ['D-DEV','D-CONFIRM']:
        vals=sorted(x['dtr'] for x in source.values() if x['split']==split and x['dtr'] is not None)
        if vals:axs[1,0].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=split)
    axs[1,0].legend();axs[1,0].set_xlabel('P4 DTR');axs[1,0].set_title('Short-tactic DTR ECDF; ties retained')
    metrics=list(primary['baselines'])
    for i,m in enumerate(metrics):
        a=primary['baselines'][m]
        if a['mean'] is not None:axs[1,1].plot(i,a['mean'],'o');axs[1,1].vlines(i,*a['ci'])
    axs[1,1].set_xticks(range(len(metrics)),metrics,rotation=45);axs[1,1].axhline(.5,color='gray');axs[1,1].set_title('Confirm within-state concordance; theorem bootstrap')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(W/'plots'/('DTR_summary.'+ext),dpi=200)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for split in ['D-DEV','D-CONFIRM']:
        rr=[r for r in source.values() if r['split']==split and r['dtr'] is not None]
        axs[0].scatter([-r['negative_entropy'] for r in rr],[r['dtr'] for r in rr],s=8,alpha=.4,label=split)
    axs[0].set_xlabel('Final predictive entropy');axs[0].set_ylabel('DTR');axs[0].legend()
    for i,split in enumerate(['D-DEV','D-CONFIRM']):
        v=shifts[split]['estimate']
        if v['mean'] is not None:axs[1].plot(i,v['mean'],'o');axs[1].vlines(i,*v['ci'])
    axs[1].set_xticks([0,1],['DEV','CONFIRM']);axs[1].axhline(0,color='gray');axs[1].set_title('Same-token P4−P0 DTR; not ability ranking')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(W/'plots'/('DTR_entropy_and_shift.'+ext),dpi=200)
    plt.close(fig)
    report='# DTR 机制观察结果\n\n独立确认集20题与 REQAT 的64题互斥。主要结果为同状态、按定理等权的 DTR−Self-Certainty 辨别率差。有效正负配对定理不足20时明确低功效；不将 token 数当独立样本。\n\n```json\n'+json.dumps(dict(confirm=primary,dev=dev,same_token_shift=shifts),indent=2,ensure_ascii=False)+'\n```\n\nDTR 是短 tactic 的层间预测修订统计；未训练模型、未运行 Think@n 或 theorem pass@64。P0 仅对 P4 实际生成 token 做重放，不是独立 P0 生成。EOS/空白、raw lens、bits 和 g/rho 敏感性保留在逐 tactic 文件。\n'
    (W/'reports/08_RESULTS.zh.md').write_text(report)
    (W/'reports/09_DECISION_MEMO.md').write_text('# Decision\n\nPrimary status: '+primary['status']+'; measurement: '+primary['measurement']+'.\n\nDo not use DTR as a training target based only on this observation. Examine same-state baseline advantage, mixed-label theorem count, short-tactic ties, unknowns and same-token quantization shift. No independent proof-search utility or causal benefit has been established.\n')
    for name,content in [('02_NUMERIC_PARITY.md',read(W/'manifests/G2.json')),('03_DATA_SPLITS_LEAKAGE.md',read(W/'configs/split_registry.json')),('05_QUANTIZATION_SHIFT.md',shifts)]:
        (W/'reports'/name).write_text('# '+name+'\n\n```json\n'+json.dumps(content,indent=2)+'\n```\n')
    for name in ['06_TACTIC_RERANKING.md','07_PROOF_SEARCH.md']:(W/'reports'/name).write_text('# '+name+'\n\nNOT_RUN. No positive independent primary signal or additional budget established.\n')
    (W/'reports/00_EXECUTION_STATUS.md').write_text('# Execution coverage\n\nP4 actual generated tactics: '+str(len(source))+'; P0 same-token replays: '+str(len(matched))+'.\n\nIndependent confirm coverage: '+str(primary['theorems'])+'/20 roots; complete8-candidate pools: '+str(primary['complete_pools'])+'. DEV coverage: '+str(dev['theorems'])+'/24. Model and pipeline receipts record exact stages. One-hour resource deadline respected; no production retrieval changes. Formal evaluations remain paused by user request.\n')
    paths=[]
    for directory in ['configs','src','tests','reports','metrics/summaries','plots']:
        paths += [p for p in (W/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    save(W/'SHA256SUMS.json',{str(p.relative_to(W)):sha(p) for p in sorted(paths)})
    status('report','DONE',primary=primary['primary'],status_science=primary['status'])
if __name__=='__main__':main()
