"""Freeze BF16-only thresholds, then state-cluster paired diagnostics."""
import argparse
import json
import numpy as np
import pandas as pd
from common import OBS, atomic, read, sha

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    inputs=OBS/'manifests/entropy_inputs.jsonl'
    rows=[json.loads(s) for s in inputs.read_text().splitlines()]
    base=OBS/'metrics/raw_summaries/entropy'
    assert (base/'P0/completed.json').exists()
    calibration=np.concatenate([np.load(base/'P0'/(r['sample_id']+'.npz'))['entropy'] for r in rows if r['subset']=='threshold'])
    q25,q75=np.quantile(calibration,[.25,.75])
    thresholds={'q25':float(q25),'q75':float(q75),'calibration_tokens':len(calibration),'input_sha256':sha(inputs),'source':'P0 threshold-only states','units':'nats/token'}
    path=OBS/'manifests/entropy_thresholds.json'
    if path.exists():assert read(path)==thresholds,'Threshold redefinition forbidden'
    else:atomic(path,thresholds)
    if args.freeze_only:return
    token_rows=[];states=[];summaries=[]
    rng=np.random.default_rng(2)
    for method in ['P3','P4','P6','P5','P1','P2']:
        if not (base/method/'completed.json').exists():continue
        for row in rows:
            if row['subset']!='analysis':continue
            with np.load(base/'P0'/(row['sample_id']+'.npz')) as t,np.load(base/method/(row['sample_id']+'.npz')) as q:
                entropy=t['entropy'];binning=np.where(entropy<=q25,'Low',np.where(entropy>=q75,'High','Mid'))
                ratio=q['own_log_tail']-t['own_log_tail']
                for i,label in enumerate(binning):
                    record={'method':method,'state_hash':row['state_hash'],'bin':label,'entropy':float(entropy[i]),'log_rho':float(ratio[i]),'delta_nll':float(q['delta_nll'][i]),'top1_flip':int(t['top1'][i]!=q['top1'][i]),'teacher_correct':int(t['top1'][i]==row['gold_ids'][i]),'category':row['categories'][i],'position':row['prediction_positions'][i],'length':len(row['input_ids']),'raw_rr':float(q['rr'][i]),'top20_kd':float(q['kl_top20'][i]),'ce':-float(q['gold_logp'][i]),'sample_tail_delta':float(q['sample_fixed_tail'][i]-t['sample_fixed_tail'][i]),
                            'teacher_tail':float(t['own_tail'][i]),'student_tail':float(q['own_tail'][i]),'absolute_tail_delta':float(q['own_tail'][i]-t['own_tail'][i]),
                            'legacy_teacher_tail':float(t['legacy_own_tail'][i]),'legacy_student_tail':float(q['legacy_own_tail'][i]),
                            'teacher_log_tail':float(t['own_log_tail'][i]),'student_log_tail':float(q['own_log_tail'][i]),
                            'fp32_log_tail_abs_error':float(abs(t['stable_fp32_log_tail'][i]-t['own_log_tail'][i])),
                            'sampler_teacher_tail':float(t['sample_fixed_tail'][i]),'sampler_student_tail':float(q['sample_fixed_tail'][i])}
                    for eps in [1e-12,1e-10,1e-8]:record['log_rho_eps_'+str(eps)]=float(np.logaddexp(q['own_log_tail'][i],np.log(eps))-np.logaddexp(t['own_log_tail'][i],np.log(eps)))
                    record['legacy_log_rho_eps_1e-08']=float(np.log((q['legacy_own_tail'][i]+1e-8)/(t['legacy_own_tail'][i]+1e-8)))
                    token_rows.append(record)
                for label in ['Low','Mid','High']:
                    mask=binning==label
                    if mask.any():states.append({'method':method,'state_hash':row['state_hash'],'bin':label,'tokens':int(mask.sum()),'mean_log_rho':float(ratio[mask].mean())})
    if not token_rows:return
    tokens=pd.DataFrame(token_rows);table=pd.DataFrame(states)
    tokens.to_csv(OBS/'metrics/entropy_tokens.csv',index=False)
    precision=tokens.groupby(['method','bin']).agg(tokens=('log_rho','size'),teacher_zero=('teacher_tail',lambda x:int((x==0).sum())),legacy_teacher_zero=('legacy_teacher_tail',lambda x:int((x==0).sum())),max_fp32_log_tail_error=('fp32_log_tail_abs_error','max'),mean_absolute_tail_delta=('absolute_tail_delta','mean'),sampler_tail_delta=('sample_tail_delta','mean'))
    precision.to_csv(OBS/'metrics/tail_numeric_audit.csv')
    epsilon_effects=[]
    for method,group in tokens.groupby('method'):
        for metric in ['log_rho','log_rho_eps_1e-12','log_rho_eps_1e-10','log_rho_eps_1e-08','legacy_log_rho_eps_1e-08']:
            wide=group.groupby(['state_hash','bin'])[metric].mean().unstack()
            if not {'Low','High'}.issubset(wide.columns):continue
            effects=(wide.Low-wide.High).dropna().to_numpy()
            if len(effects)<2:continue
            local_rng=np.random.default_rng(2)
            draws=[local_rng.choice(effects,size=len(effects),replace=True).mean() for _ in range(2000)]
            lo,hi=np.quantile(draws,[.025,.975]);epsilon_effects.append({'method':method,'metric':metric,'state_pairs':len(effects),'effect':effects.mean(),'ci_low':lo,'ci_high':hi})
    pd.DataFrame(epsilon_effects).to_csv(OBS/'metrics/epsilon_sensitivity.csv',index=False)
    table.to_csv(OBS/'metrics/entropy_by_state.csv',index=False)
    tokens.groupby(['method','bin','category']).agg(tokens=('log_rho','size'),states=('state_hash','nunique'),mean_log_rho=('log_rho','mean'),delta_nll=('delta_nll','mean'),top1_flip=('top1_flip','mean')).to_csv(OBS/'metrics/entropy_by_token_category.csv')
    for method,group in table.groupby('method'):
        wide=group.pivot(index='state_hash',columns='bin',values='mean_log_rho')
        if not {'Low','High'}.issubset(wide.columns):continue
        paired=wide[['Low','High']].dropna();effects=(paired.Low-paired.High).to_numpy()
        if len(effects)<2:continue
        draws=np.array([rng.choice(effects,size=len(effects),replace=True).mean() for _ in range(2000)])
        low,high=np.quantile(draws,[.025,.975])
        summaries.append({'method':method,'state_pairs':len(effects),'low_minus_high_log_rho':float(effects.mean()),'ci_low':float(low),'ci_high':float(high),'population':'in_training_distribution','estimand':'equal-weight states having both Low and High eligible positions'})
    pd.DataFrame(summaries).to_csv(OBS/'metrics/entropy_bootstrap.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=OBS/'plots/entropy';target.mkdir(parents=True,exist_ok=True)
    teacher_h=np.concatenate([np.load(base/'P0'/(r['sample_id']+'.npz'))['entropy'] for r in rows if r['subset']=='analysis'])
    fig,ax=plt.subplots();ax.hist(teacher_h,bins=60);ax.axvline(q25,color='orange');ax.axvline(q75,color='red');ax.set_xlabel('BF16 predictive entropy (nats/token)');ax.set_title('In-training-distribution; thresholds from separate states');fig.tight_layout();fig.savefig(target/'entropy_hist_bf16.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots()
    means=tokens.groupby(['method','bin']).log_rho.mean().unstack().reindex(columns=['Low','Mid','High'])
    means.plot.bar(ax=ax);ax.set_ylabel('Mean log own-tail mass ratio');ax.set_title('W2 numerical weights, dense BF16 forward; train distribution');fig.tight_layout();fig.savefig(target/'low_high_tail_mass_ratio.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots()
    for method,group in tokens.groupby('method'):
        group=group.copy();group['decile']=pd.qcut(group.entropy,10,labels=False,duplicates='drop');a=group.groupby('decile')[['entropy','delta_nll']].mean();ax.plot(a.entropy,a.delta_nll,marker='.',label=method)
    ax.legend();ax.set_xlabel('BF16 predictive entropy');ax.set_ylabel('Paired delta NLL');ax.set_title('Exploratory deciles; training distribution');fig.tight_layout();fig.savefig(target/'entropy_vs_delta_nll.png',dpi=180);plt.close(fig)
    (OBS/'reports/03b_entropy_failure_modes.md').write_text('# Corrected paired entropy diagnostics\n\nPrimary: Low minus High mean (log_tail_Q-log_tail_T), no epsilon, equal-weight state groups with both bins; 2000 cluster bootstrap draws, seed2, 95% CI. Thresholds recomputed using stable FP64 entropy from separate BF16-only calibration state groups. Full-vocabulary statistics, BF16 model, gold-prefix teacher forcing.\n\n'+pd.DataFrame(summaries).to_csv(index=False)+'\n\nLegacy float32 cancellation is reconstructed on the SAME new logits in tail_numeric_audit.csv and epsilon_sensitivity.csv; legacy results are superseded for tail conclusions. Report absolute probability changes and actual T1.5/top_p.9 sampler tails alongside relative ratios. A relative change does not establish sampling or verifier harm. Token categories remain lexical heuristics. Independent dev verification, frequency controls and native-scale gradients remain incomplete; no theorem ranking inferred.\n')
    tokens.groupby('method')[['ce','raw_rr','top20_kd']].mean().to_csv(OBS/'metrics/loss_diagnostics.csv')
    (OBS/'reports/05_loss_objective_analysis.md').write_text('# Objective diagnostic status\n\nSame frozen development prefixes, raw RR=p_teacher(gold)*CE; P4 objective=.2*RR+top20 KD, full-vocab-normalized top20 without re-normalization. Values in loss_diagnostics.csv. These are not full-vocab standard KL.\n\nPARTIAL: native quantizer-scale gradients not run on shared retrieval GPU; logits-only or dense-weight gradients must not substitute for them. WAITING_FINAL_EVAL: no theorem success/loss ordering claim.\n')

if __name__=='__main__':main()
