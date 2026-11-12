"""Traceable, CPU-only publication figures from immutable DTR observations."""
import argparse,csv,hashlib,json,math,random,shutil,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

W=Path(__file__).resolve().parents[2]
COLORS={'P0':'#0072B2','P4':'#D55E00','P3':'#009E73','P6':'#CC79A7'}
SEED=2;B=10000
MANIFEST_ROWS=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def jsonl(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
def loadrows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def csvfile(path,data,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(data)
def depths(tokens):
    return [next(i+1 for i,v in enumerate(t['jsd']) if v<=.5) for t in tokens]
def category(text):
    import re
    t=text.strip()
    if t in {'exact','apply','rw','simp','simpa','intro','rintro','have','refine','constructor','linarith','nlinarith','ring','norm_num','aesop','rfl'}:return 'keyword'
    if not t:return 'whitespace'
    if re.fullmatch(r'\d+(\.\d+)?',t):return 'numeral'
    if re.fullmatch(r'[\[\](){}⟨⟩,:;]+',t):return 'punctuation'
    if re.fullmatch(r'[+*/^=<>≤≥≠↔→∧∨¬-]+',t):return 'operator'
    if re.fullmatch(r'[\w]+(?:\.[\w]+)+',t):return 'lemma-like'
    if re.fullmatch(r'[\w]+',t):return 'identifier'
    return 'other'
def interval(x,alpha=.05):
    x=np.asarray(x,dtype=float)
    if not len(x):return {'n_theorems':0,'mean':None,'ci':None}
    rng=np.random.default_rng(SEED);bs=x[rng.integers(0,len(x),(B,len(x)))].mean(1)
    return {'n_theorems':len(x),'mean':float(x.mean()),'ci':list(map(float,np.quantile(bs,[alpha/2,1-alpha/2])))}
def micro_interval(rows,key='is_valid_transition'):
    ids=sorted({r['theorem_id'] for r in rows});ys=np.array([sum(r[key] for r in rows if r['theorem_id']==i) for i in ids]);ns=np.array([sum(r['theorem_id']==i for r in rows) for i in ids])
    rng=np.random.default_rng(SEED);ix=rng.integers(0,len(ids),(B,len(ids)));bs=ys[ix].sum(1)/ns[ix].sum(1)
    return {'mean':float(ys.sum()/ns.sum()),'ci':list(map(float,np.quantile(bs,[.025,.975]))),'n_tactics':len(rows),'n_theorems':len(ids)}
def export(out):
    primary={};rows=[];tokenrows=[];matrices=[];sources={};case_tokens={};excluded=[]
    optional_counts={m:len(list((W/'metrics'/('secondary_replay_'+m)).glob('*.json'))) for m in ['P3','P6']}
    for p in sorted((W/'metrics/per_tactic').glob('*.json')):
        r=read(p)
        if r.get('status')!='DONE':excluded.append({'file':str(p.relative_to(W)),'reason':r.get('status')});continue
        sourceid=p.stem;primary[sourceid]=r;sources[str(p.relative_to(W))]=digest(p)
        tp=W/r['token_file'];sources[str(tp.relative_to(W))]=digest(tp);assert digest(tp)==r['token_sha']
        tt=[t for t in read(tp) if not t['stop_flag']]
        assert [t['token_id'] for t in tt]==r['generated_ids']
        available={'P4':(r,tt)};pair=W/'metrics/paired_replay'/p.name
        assert pair.exists(),'Missing paired P0 replay: '+p.name
        q=read(pair);assert q['generated_token_ids_sha']==r['generated_token_ids_sha'] and q['input_ids_sha']==r['input_ids_sha']
        sources[str(pair.relative_to(W))]=digest(pair);available['P0']=(q,q['tokens'])
        for model in ['P3','P6']:
            if optional_counts[model]==192 and r['split']=='D-DEV':
                pp=W/'metrics'/('secondary_replay_'+model)/p.name
                if pp.exists():s=read(pp);sources[str(pp.relative_to(W))]=digest(pp);available[model]=(s,s['tokens'])
        for model,(q,tokens) in available.items():
            assert len(tokens)==len(tt)
            if not tokens:excluded.append({'candidate_id':sourceid,'model':model,'reason':'INVALID_EMPTY_SEQUENCE'});continue
            ds=depths(tokens);dtr=sum(x>=24 for x in ds)/len(ds);assert abs(dtr-q['dtr'])<1e-10
            known=r['lean_status'] in ['VALID_TRANSITION','INVALID_TACTIC'];closed=r.get('goals_remaining')==0 and r['lean_status']=='VALID_TRANSITION'
            status='GOALS_CLOSED' if closed else 'VALID_TRANSITION' if r['lean_status']=='VALID_TRANSITION' else 'INVALID' if r['lean_status']=='INVALID_TACTIC' else r['lean_status']
            row=dict(theorem_id=r['theorem_id'],state_id=r['state_id'],candidate_id=sourceid,model=model,split=r['split'],tactic_text=r['tactic_text'],token_length=len(tokens),dtr=dtr,self_certainty=q['self_certainty'],avg_logprob=q['mean_logprob'],negative_entropy=q['negative_entropy'],lean_status=status,is_valid_transition=int(r['lean_status']=='VALID_TRANSITION') if known else None,is_goal_closed=int(closed) if known else None,token_count_used_in_dtr=len(tokens),deep_token_count=sum(x>=24 for x in ds),generation_origin='P4',measurement='natural P4' if model=='P4' else 'same-token replay',seed=r['seed'])
            rows.append(row);case_tokens[(sourceid,model)]=tokens
            matrix=dict(theorem_id=r['theorem_id'],state_id=r['state_id'],candidate_id=sourceid,model=model,split=r['split'],token_strings=[t.get('token_text',tt[i]['token_text']) for i,t in enumerate(tokens)],token_ids=[t['token_id'] for t in tokens],layer_ids=list(range(1,29)),jsd_matrix=np.array([t['jsd'] for t in tokens]).T.tolist(),dtr=dtr,lean_status=status,tactic_text=r['tactic_text'],generation_origin='P4')
            matrices.append(matrix)
            for i,t in enumerate(tokens):
                assert len(t['jsd'])==28 and abs(t['jsd'][-1])<1e-5
                text=t.get('token_text',tt[i]['token_text'])
                tokenrows.append(dict(theorem_id=r['theorem_id'],state_id=r['state_id'],candidate_id=sourceid,model=model,split=r['split'],token_pos=i,token_str=text,token_id=t['token_id'],jsd_deep=float(np.mean(t['jsd'][23:])),is_deep_thinking_token=int(ds[i]>=24),first_crossing_depth=ds[i],final_logprob=t['logprob'],token_type=category(text)))
    for name in ['configs/protocol.json','configs/split_registry.json','metrics/summaries/D-CONFIRM.json','metrics/summaries/D-DEV.json','manifests/frozen_cases.jsonl']:
        sources[name]=digest(W/name)
    csvfile(W/'metrics/per_tactic_metrics.csv',rows,list(rows[0]));csvfile(W/'metrics/per_token_metrics.csv',tokenrows,list(tokenrows[0]));jsonl(W/'metrics/per_layer_jsd.jsonl',matrices)
    write(out/'tables/data_snapshot.json',{'optional_DEV_model_receipts':optional_counts,'optional_rule':'include only complete192; no development candidates in main figures','rows_per_model_split':{m:{s:sum(r['model']==m and r['split']==s for r in rows) for s in ['D-DEV','D-CONFIRM']} for m in sorted({r['model'] for r in rows})},'sources':sources,'excluded':excluded})
    return rows,tokenrows,matrices,case_tokens,sources
def binned(rows,metric):
    known=[r for r in rows if r['is_valid_transition'] is not None];vals=np.array([r[metric] for r in known]);edges=np.unique(np.quantile(vals,np.linspace(0,1,6)));bins=[]
    assignments=np.searchsorted(edges[1:-1],vals,side='right')
    for i in range(max(1,len(edges)-1)):
        rr=[r for r,j in zip(known,assignments) if j==i]
        if not rr:continue
        bins.append(dict(micro_interval(rr),x_mean=float(np.mean([r[metric] for r in rr])),bin=i+1,lower=float(edges[min(i,len(edges)-1)]),upper=float(edges[min(i+1,len(edges)-1)])))
    rho=None if len(set(vals))<2 or len({r['is_valid_transition'] for r in known})<2 else float(spearmanr(vals,[r['is_valid_transition'] for r in known]).statistic)
    return dict(bins=bins,rho_spearman=rho,edges=list(map(float,edges)),tie_rule='all equal values in same bin; no jitter or outcome-based edges',bin_collapse=len(bins)<5)
def savefig(fig,out,name,stats,manifest,sources,note):
    target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(target.with_suffix('.'+ext),bbox_inches='tight',dpi=300,metadata={'Creator':'REAL-Prover deterministic DTR figure scripts'})
    plt.close(fig)
    if name.endswith('Fig2_jsd_heatmaps_p0_p4') or name.endswith('FigA2_case_panels'):
        ids={r['candidate_id'] for r in stats['cases'].values()}
        selected=[r for r in MANIFEST_ROWS if r['candidate_id'] in ids and r['model'] in ['P0','P4']]
    elif name.endswith('FigA3_per_model_binned_curves'):selected=MANIFEST_ROWS
    else:selected=[r for r in MANIFEST_ROWS if r['split']=='D-CONFIRM' and r['model'] in ['P0','P4']]
    ids={r['candidate_id'] for r in selected}
    case_prefixes=['metrics/per_tactic/','metrics/per_token/','metrics/paired_replay/','metrics/secondary_replay_']
    files=[p for p in sources if not any(p.startswith(prefix) for prefix in case_prefixes) or Path(p).stem in ids]
    manifest[name]=dict(data_files=files,sample_counts=dict(unique_candidates=len(ids),model_measurements=len(selected),unique_theorems=len({r['theorem_id'] for r in selected})),models=stats.get('models',['P0','P4']),statistics=stats,note=note,script_sha=digest(__file__))
def fig1(rows,out,stats,manifest,sources):
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.7));result={'models':['P0','P4'],'split':'D-CONFIRM','metrics':{}}
    for ax,metric,label in zip(axs,['token_length','dtr'],['Token length','Deep-Thinking Ratio (DTR)']):
        for model in ['P0','P4']:
            rr=[r for r in rows if r['model']==model and r['split']=='D-CONFIRM'];a=binned(rr,metric);result['metrics'][model+'_'+metric]=a
            bins=a['bins'];x=[b['x_mean'] for b in bins];y=[b['mean'] for b in bins];err=np.array([[b['mean']-b['ci'][0],b['ci'][1]-b['mean']] for b in bins]).T
            rho=a['rho_spearman'];text=f'{model} ρ={rho:.3f}' if rho is not None else model+' ρ=undefined'
            ax.errorbar(x,y,yerr=err,color=COLORS[model],marker='o' if model=='P4' else 's',linestyle='-' if model=='P4' else '--',capsize=3,label=text,alpha=.9)
            if model=='P4':
                for b in bins:ax.annotate(f"n={b['n_tactics']}; T={b['n_theorems']}",(b['x_mean'],b['mean']),xytext=(0,10),textcoords='offset points',fontsize=7,ha='center')
        ax.set_xlabel(label);ax.set_ylabel('Lean tactic validity');ax.set_ylim(-.05,1.13);ax.legend(fontsize=8);ax.grid(alpha=.18)
    fig.suptitle('Confirmation: 20 theorems · shared P4-generated candidate pool',fontsize=12)
    fig.text(.5,-.02,'Tie-preserving quintile boundaries; 95% theorem-cluster bootstrap. P0 is same-token replay; length curves overlap.',ha='center',fontsize=8)
    fig.tight_layout();stats['Fig1']=result;savefig(fig,out,'main/Fig1_length_vs_dtr_validity',result,manifest,sources,'Descriptive binned micro-validity, not theorem proof success; no natural P0 generation.')
def select_cases(rows,out):
    rr=sorted([r for r in rows if r['model']=='P4' and r['split']=='D-CONFIRM'],key=lambda r:r['candidate_id']);rng=random.Random(SEED)
    valid=[r for r in rr if r['is_valid_transition']==1];invalid=[r for r in rr if r['is_valid_transition']==0]
    assert valid and invalid
    selected={'V0':rng.choice(valid),'V3':rng.choice(invalid)}
    data=dict(seed=SEED,selection='Uniform seeded choice among complete confirmation P4 candidates by observed validity only; never JSD, DTR or appearance',V0=selected['V0'],V3=selected['V3'],
              V1=dict(status='NOT_APPLICABLE',reason='No independently generated P0 candidate pool. Same tactic on same Lean state has model-invariant verdict; replay cannot supply P0-valid/P4-invalid outcomes.',fallback='V3 used for main invalid row'),
              V2=dict(status='NOT_APPLICABLE',reason='No independently generated P0 candidates; cannot assign different verdicts to identical replayed tokens.',fallback='V0 used for main valid row'))
    write(out/'tables/selected_case_manifest.json',data);return selected
def tokenlabel(t,token_id=None):
    # A byte-BPE piece can be an incomplete UTF-8 character in isolation.
    # Preserve native IDs rather than display a misleading replacement glyph.
    if '\ufffd' in t:return '#'+str(token_id) if token_id is not None else '[byte]'
    return t.replace('\n','\\n').replace(' ','·').replace('\t','\\t')[:14]
def fig2(rows,matrices,cases,out,stats,manifest,sources):
    index={(r['candidate_id'],r['model']):r for r in matrices};fig,axs=plt.subplots(2,2,figsize=(16,9.5),layout='constrained');details={};im=None
    for i,(tag,case) in enumerate(cases.items()):
        for j,model in enumerate(['P0','P4']):
            r=index[case['candidate_id'],model];matrix=np.array(r['jsd_matrix']);N=min(32,matrix.shape[1]);ax=axs[i,j]
            im=ax.imshow(matrix[:,:N],origin='lower',aspect='auto',vmin=0,vmax=math.log(2),cmap='viridis',extent=(-.5,N-.5,.5,28.5),interpolation='nearest')
            ax.axhline(23.5,color='white',linestyle='--',linewidth=1.1);ax.set_yticks([1,7,14,21,24,28]);ax.set_ylabel('Decoder layer');ax.set_xticks(range(N),[tokenlabel(t,r['token_ids'][k]) for k,t in enumerate(r['token_strings'][:N])],rotation=55,ha='right',fontsize=8)
            ax.set_title(f"{model} {'BF16' if model=='P0' else 'W2G128'} · DTR={r['dtr']:.3f} · {case['lean_status']}",fontsize=10)
            ax.set_xlabel(f"{tag}: {case['theorem_id']} · {N}/{matrix.shape[1]} tokens shown",fontsize=9);details[tag+'_'+model]={'candidate_id':case['candidate_id'],'DTR':r['dtr'],'tokens':matrix.shape[1],'shown_tokens':N,'status':case['lean_status']}
    fig.colorbar(im,ax=axs,shrink=.8,label='Full-vocabulary JSD (nats)',ticks=[0,.2,.4,.6,math.log(2)])
    fig.suptitle('Identical prompt and tactic tokens · normalized logit lens · deep regime: layers 24–28',fontsize=13)
    result={'models':['P0','P4'],'cases':details,'color_range':[0,math.log(2)],'split':'D-CONFIRM'};stats['Fig2']=result;savefig(fig,out,'main/Fig2_jsd_heatmaps_p0_p4',result,manifest,sources,'V0/V3 seeded cases. Discordant natural-generation cases unavailable; see selected_case_manifest.json.')
def concordance(rows,metric,target):
    vals=[];per=[]
    for theorem in sorted({r['theorem_id'] for r in rows}):
        ss=[]
        for sid in sorted({r['state_id'] for r in rows if r['theorem_id']==theorem}):
            group=[r for r in rows if r['theorem_id']==theorem and r['state_id']==sid];pos=[r for r in group if r[target]==1];neg=[r for r in group if r[target]==0]
            if not pos or not neg:continue
            ss.append(sum(float(a[metric]>b[metric])+.5*float(a[metric]==b[metric]) for a in pos for b in neg)/(len(pos)*len(neg)))
        if ss:v=float(np.mean(ss));vals.append(v);per.append(dict(theorem_id=theorem,value=v))
    return dict(interval(vals),per_theorem=per)
def fig3(rows,out,stats,manifest,sources):
    fig,axs=plt.subplots(1,2,figsize=(12,5));result={'models':['P0','P4'],'split':'D-CONFIRM','targets':{}};primary=read(W/'metrics/summaries/D-CONFIRM.json');metrics=['dtr','self_certainty','avg_logprob','token_length'];labels=['DTR','Self-Certainty','Avg logprob','Length'];mapping={'dtr':'dtr','self_certainty':'self_certainty','avg_logprob':'mean_logprob','token_length':'length'}
    for ax,target,title in zip(axs,['is_valid_transition','is_goal_closed'],['Valid vs invalid tactic','Goals closed vs other known outcomes']):
        result['targets'][target]={}
        for j,model in enumerate(['P0','P4']):
            rr=[r for r in rows if r['model']==model and r['split']=='D-CONFIRM' and r[target] is not None];record={}
            for i,m in enumerate(metrics):
                a=concordance(rr,m,target)
                if model=='P4' and target=='is_valid_transition':
                    saved=primary['baselines'][mapping[m]];assert abs(a['mean']-saved['mean'])<1e-10 and a['n_theorems']==saved['n']
                    a['ci']=saved['ci'];a['interval_source']='frozen primary analysis, reused exactly'
                record[m]=a
                if a['mean'] is not None:
                    x=i+(j-.5)*.3;ax.bar(x,a['mean'],width=.28,color=COLORS[model],label=model if i==0 else None,alpha=.85);ax.vlines(x,*a['ci'],color='black',lw=1);ax.text(x,.03,f"T={a['n_theorems']}",rotation=90,ha='center',va='bottom',fontsize=7)
            result['targets'][target][model]=record
        ax.set_xticks(range(4),labels,rotation=15);ax.set_ylim(0,1.06);ax.axhline(.5,color='gray',ls='--',lw=.8);ax.set_ylabel('Within-state concordance');ax.set_title(title,fontsize=11);ax.legend()
    d=primary['primary'];fig.suptitle(f"P4 primary Δ(DTR−SC)={d['mean']:.3f}; 95% CI [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] · {primary['status']}",fontsize=11)
    fig.text(.5,-.01,'Theorems with both labels contribute; goals-closed is secondary. P0 scores identical P4-origin candidates.',ha='center',fontsize=8);fig.tight_layout();stats['Fig3']=result;savefig(fig,out,'main/Fig3_same_state_metric_comparison',result,manifest,sources,'No new primary significance test. P4 VALID intervals and hypothesis result copied from frozen main analysis.')
def groupshift(rows):
    paired={}
    for r in rows:
        if r['split']=='D-CONFIRM' and r['model'] in ['P0','P4']:paired.setdefault(r['candidate_id'],{})[r['model']]=r
    result=[]
    for pair in paired.values():
        if set(pair)!= {'P0','P4'}:continue
        a,b=pair['P4'],pair['P0'];assert a['lean_status']==b['lean_status'] and a['state_id']==b['state_id']
        result.append(dict(theorem_id=a['theorem_id'],outcome=a['lean_status'],delta_dtr=a['dtr']-b['dtr'],delta_self_certainty=a['self_certainty']-b['self_certainty']))
    return result
def violin(ax,sets,positions,colors):
    for values,x,col in zip(sets,positions,colors):
        if not values:continue
        if len(set(values))>1:
            v=ax.violinplot(values,positions=[x],showextrema=False,widths=.65)
            for body in v['bodies']:body.set_facecolor(col);body.set_alpha(.25)
        ax.boxplot([values],positions=[x],widths=.15,showfliers=False,medianprops={'color':col},boxprops={'color':col},whiskerprops={'color':col},capprops={'color':col})
def fig4(rows,out,stats,manifest,sources):
    data=groupshift(rows);groups=['VALID_TRANSITION','GOALS_CLOSED','INVALID'];fig,axs=plt.subplots(1,2,figsize=(12,4.7));result={'models':['P0','P4'],'split':'D-CONFIRM','groups':{}}
    for ax,metric,title in zip(axs,['delta_dtr','delta_self_certainty'],['DTR(P4) − DTR(P0)','Self-Certainty(P4) − Self-Certainty(P0)']):
        vv=[[r[metric] for r in data if r['outcome']==g] for g in groups];violin(ax,vv,range(3),['#0072B2','#009E73','#D55E00'])
        for x,g in enumerate(groups):
            rr=[r for r in data if r['outcome']==g];vals=[np.mean([r[metric] for r in rr if r['theorem_id']==i]) for i in sorted({r['theorem_id'] for r in rr})];a=interval(vals);a['n_tactics']=len(rr);result['groups'][g+'_'+metric]=a
            if vals:ax.scatter(np.array([x]*len(vals))+np.linspace(-.13,.13,len(vals)),vals,s=13,color='black',alpha=.55,zorder=3)
        ax.set_xticks(range(3),['Valid, goals remain','Goals closed','Invalid'],rotation=12);ax.axhline(0,color='gray',lw=.8);ax.set_ylabel(title);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Same-token prediction-convergence shift · confirmation set',fontsize=12);fig.text(.5,-.01,'Violin: tactic distribution; black points: theorem means. Higher DTR does not imply greater reasoning ability.',ha='center',fontsize=8);fig.tight_layout();stats['Fig4']=result;savefig(fig,out,'main/Fig4_dtr_shift_p0_vs_p4',result,manifest,sources,'Descriptive paired weights comparison; all strings and Lean labels fixed.')
def appendix(rows,tokens,matrices,cases,out,stats,manifest,sources):
    groups=['VALID_TRANSITION','GOALS_CLOSED','INVALID'];fig,ax=plt.subplots(figsize=(9,4.5))
    outcome_stats={}
    for j,model in enumerate(['P0','P4']):
        rr=[r for r in rows if r['model']==model and r['split']=='D-CONFIRM'];sets=[[r['dtr'] for r in rr if r['lean_status']==g] for g in groups];violin(ax,sets,[x+(j-.5)*.32 for x in range(3)],[COLORS[model]]*3);ax.plot([],[],color=COLORS[model],label=model)
        for g,values in zip(groups,sets):outcome_stats[model+'_'+g]=dict(n_tactics=len(values),mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None,n_theorems=len({r['theorem_id'] for r in rr if r['lean_status']==g}))
    ax.set_xticks(range(3),['Valid, goals remain','Goals closed','Invalid']);ax.set_ylabel('DTR');ax.set_ylim(-.03,1.03);ax.legend();ax.set_title('Confirmation outcome groups · P0 is same-token replay');fig.tight_layout();savefig(fig,out,'appendix/FigA1_dtr_by_outcome',{'models':['P0','P4'],'groups':groups,'distribution_summary':outcome_stats},manifest,sources,'Goals-closed and goals-remain are mutually exclusive.')
    idx={(r['candidate_id'],r['model']):r for r in matrices};fig,axs=plt.subplots(2,1,figsize=(16,9));caseinfo={}
    for ax,(tag,r) in zip(axs,cases.items()):
        p0=idx[r['candidate_id'],'P0'];p4=idx[r['candidate_id'],'P4'];a=(np.array(p0['jsd_matrix']).T<=.5).argmax(1)+1;b=(np.array(p4['jsd_matrix']).T<=.5).argmax(1)+1
        ax.imshow(np.array([a>=24,b>=24],dtype=int),aspect='auto',vmin=0,vmax=1,cmap='Blues',interpolation='nearest');ax.set_yticks([0,1],['P0 deep','P4 deep']);ax.set_xticks(range(len(a)),[tokenlabel(t,p4['token_ids'][k]) for k,t in enumerate(p4['token_strings'])],rotation=55,ha='right',fontsize=7)
        full='\n'.join(textwrap.wrap(r['tactic_text'],120));ax.set_title(f"{tag} · {r['theorem_id']} · {r['lean_status']}\n{full}",fontfamily='DejaVu Sans Mono',fontsize=9);caseinfo[tag]={'candidate_id':r['candidate_id'],'full_tactic':r['tactic_text'],'token_count':len(a)}
    fig.suptitle('Seeded confirmation cases · full tactic text and token deep flags');fig.tight_layout();savefig(fig,out,'appendix/FigA2_case_panels',{'models':['P0','P4'],'cases':caseinfo},manifest,sources,'No selection by heatmap appearance; full token labels supplied in exported JSONL and CSV.')
    fig,axs=plt.subplots(1,2,figsize=(12,4.7));curves={}
    for ax,split,title in zip(axs,['D-CONFIRM','D-DEV'],['Confirmation: paired P0/P4','Development only: available complete model replays']):
        for model in ['P0','P3','P4','P6']:
            rr=[r for r in rows if r['model']==model and r['split']==split]
            if not rr:continue
            a=binned(rr,'dtr');curves[split+'_'+model]=a;ax.plot([b['x_mean'] for b in a['bins']],[b['mean'] for b in a['bins']],'-o',label=model,color=COLORS[model])
        ax.set_xlabel('DTR');ax.set_ylabel('Lean tactic validity');ax.set_ylim(-.03,1.03);ax.legend();ax.set_title(title,fontsize=10)
    fig.tight_layout();savefig(fig,out,'appendix/FigA3_per_model_binned_curves',{'models':sorted({r['model'] for r in rows}),'curves':curves,'P5':'NOT_RUN'},manifest,sources,'DEV and confirm are separated; partial optional-model replays omitted, never mixed into main curves.')
    tt=[r for r in tokens if r['split']=='D-CONFIRM' and r['model'] in ['P0','P4']];types=sorted({r['token_type'] for r in tt});fig,axs=plt.subplots(1,2,figsize=(13,4.8));breakdown={}
    for j,model in enumerate(['P0','P4']):
        means=[]
        for typ in types:
            rr=[r for r in tt if r['model']==model and r['token_type']==typ];vals=[np.mean([r['is_deep_thinking_token'] for r in rr if r['theorem_id']==i]) for i in sorted({r['theorem_id'] for r in rr})];a=interval(vals);a['n_tokens']=len(rr);breakdown[model+'_'+typ]=a;means.append(a['mean'])
        axs[0].bar(np.arange(len(types))+(j-.5)*.35,means,.33,label=model,color=COLORS[model])
    for i,typ in enumerate(types):
        a={r['theorem_id']:[] for r in tt if r['token_type']==typ}
        for theorem in a:
            p0=[r['is_deep_thinking_token'] for r in tt if r['theorem_id']==theorem and r['token_type']==typ and r['model']=='P0'];p4=[r['is_deep_thinking_token'] for r in tt if r['theorem_id']==theorem and r['token_type']==typ and r['model']=='P4'];a[theorem]=float(np.mean(p4)-np.mean(p0))
        result=interval(list(a.values()));breakdown['delta_'+typ]=result
        if result['mean'] is not None:axs[1].plot(i,result['mean'],'o',color='black');axs[1].vlines(i,*result['ci'])
    for ax in axs:ax.set_xticks(range(len(types)),types,rotation=35,ha='right')
    axs[0].set_ylabel('Theorem-macro deep-token fraction');axs[0].legend();axs[1].set_ylabel('P4 − P0 deep-token fraction');axs[1].axhline(0,color='gray');fig.suptitle('Heuristic BPE token categories · confirmation only');fig.tight_layout();savefig(fig,out,'appendix/FigA4_token_type_breakdown',{'models':['P0','P4'],'categories':breakdown},manifest,sources,'Categories describe individual BPE pieces, not certified Lean semantic roles; all support counts in table.')
    stats['appendix']={'model_curves':curves,'token_types':breakdown,'case_panels':caseinfo,'outcome_distributions':outcome_stats}
def main():
    global MANIFEST_ROWS
    p=argparse.ArgumentParser();p.add_argument('--input',default=str(W/'metrics/per_tactic_metrics.csv'));p.add_argument('--outdir',default=str(W/'plots_final'));p.add_argument('--figure',default='all');a=p.parse_args();out=Path(a.outdir).resolve()
    if out.name in ['main','appendix']:out=out.parent
    requested_input=Path(a.input);requested_input=requested_input if requested_input.is_absolute() else W/requested_input
    assert requested_input.resolve()==(W/'metrics/per_tactic_metrics.csv').resolve(),'Only the frozen DTR CSV adapter is supported; do not substitute or silently mix a different candidate pool'
    assert out.is_relative_to(W.resolve()),'Output must remain within isolated DTR workspace'
    for d in ['main','appendix','tables','reports','scripts']:(out/d).mkdir(parents=True,exist_ok=True)
    rows,tokens,matrices,case_tokens,sources=export(out);stats={'seed':SEED,'bootstrap_replicates':B,'bootstrap_unit':'theorem_id','main_split':'D-CONFIRM','generation_origin':'P4 only; P0 and optional models replay exact same tokens','models_natural_generation':['P4'],'models_paired_replay':sorted({r['model'] for r in rows if r['model']!='P4'})};manifest={}
    MANIFEST_ROWS=rows
    cases=select_cases(rows,out)
    if a.figure in ['all','1']:fig1(rows,out,stats,manifest,sources)
    if a.figure in ['all','2']:fig2(rows,matrices,cases,out,stats,manifest,sources)
    if a.figure in ['all','3']:fig3(rows,out,stats,manifest,sources)
    if a.figure in ['all','4']:fig4(rows,out,stats,manifest,sources)
    if a.figure in ['all','appendix']:appendix(rows,tokens,matrices,cases,out,stats,manifest,sources)
    write(out/'tables/figure_statistics_summary.json',stats)
    write(out/'tables/figure_source_manifest.json',{'figures':manifest,'source_sha256':sources,'derived_tactic_rows':len(rows),'derived_token_rows':len(tokens),'derived_layer_records':len(matrices),'script_sha256':digest(__file__),'input_option':str(requested_input.relative_to(W)),'input_note':'CSV is generated by the immutable-record adapter; source JSON SHA identities are authoritative'})
    (out/'reports/FIGURE_README.md').write_text('''# DTR publication figures\n\nRun `bash src/figures/run_all_figures.sh` from the isolated DTR workspace. Canonical scripts live outside plots_final so the whole output directory can be regenerated after deletion. Generated plots_final/scripts wrappers also support `python build_fig1.py --input metrics/per_tactic_metrics.csv --outdir plots_final/main`. CPU only; no new model forward, Lean evaluation or experiment change.\n\nMain figures use all160 confirmation P4 tactics across20 roots, with aligned P0 same-token scores. P0 is NOT an independently generated pool: its length and Lean labels are identical. Figures1/2 are descriptive adaptations of the paper's Figure1/2, not a replication on long CoT.\n\nFig1: tie-preserving quintile boundaries, micro tactic validity, theorem-cluster95% CI,10000 resamples seed2; Spearman is descriptive. Equal values never split by labels. Fig2: seeded uniform choice among confirmation valid/invalid cases; shared[0,ln2] JSD nats scale, normalized lens, g=.5/rho=.85, cutoff24; at most32 tokens in main, full text in A2. V1/V2 discordant natural-generation classes are NOT_APPLICABLE, because only P4 was generated; never fabricate different Lean verdicts for replayed identical strings.\n\nFig3: same-state concordance averaged equally by theorem. P4 validity point estimates and intervals reused exactly from frozen primary analysis; the primary effective count is8, below the prespecified20 threshold, so INCONCLUSIVE_LOW_POWER even if the displayed interval is negative. Goals-closed is secondary. Fig4: identical-token shift grouped into mutually exclusive valid-goals-remain/goals-closed/invalid; violin shows tactic distribution, black points theorem means. Higher DTR is not automatically better reasoning.\n\nA1: outcome distributions; A2: full tactic cases; A3: separate confirmation and development panels, only complete192-candidate P3/P6 DEV pools if available, P5 NOT_RUN; A4: heuristic BPE-level categories, not verified semantic labels.\n\nInputs: metrics/per_tactic/*.json, metrics/per_token/*.json, metrics/paired_replay/*.json, optional complete secondary_replay_P3/P6, frozen case/split manifests and primary summary. Derived metrics/per_tactic_metrics.csv, per_token_metrics.csv, per_layer_jsd.jsonl are exported and validated without re-tokenization. Every displayed value and data/script SHA is in tables/. No unknown verdict is counted as invalid. These are tactic-level measurements, not theorem pass@64 or evidence that SEM training works.\n''')
    (out/'reports/figure_decision_log.md').write_text('''# Figure decisions\n\n- Binning fixed to empirical quintile boundaries, ties preserved and duplicate boundaries collapsed; no outcome-optimized bins, scatter exploration or alternative favorable layouts were tried. Flat or negative curves remain.\n- P0-only generation is unavailable. P0/P4 paired plotting is explicitly on P4-origin candidates; length curves intentionally overlap.\n- V0 and V3 selected by seed2 uniform choice from sorted confirmation valid/invalid candidates. V1/V2 are structurally unsupported, not evidence of zero discordance between independently generated models. No new generation was added for prettier cases.\n- Main heatmap uses two real tactics × two models with one common nats scale. Full text and token-ID source records remain available. Individual BPE bytes that cannot decode as UTF-8 show #token_id on ticks instead of replacement glyphs; no token IDs or statistics are changed.\n- Fig3 displays the prespecified primary result unchanged; effective mixed-label theorem support and low-power status retained.\n- Violin/box distribution and theorem mean dots convey repeated candidates without implying token/candidate independence. Optional incomplete model pools omitted and recorded in the data snapshot.\n- No empirical results, threshold, lens, dataset split or primary endpoint was changed. Validation found isolated byte-BPE fragments decoding to replacement glyphs. Tick labels now use #token_id for these fragments; full tactic text is preserved in A2 and exact token IDs in CSV/JSONL. No values or case choices changed.\n''')
    for path in (W/'src/figures').glob('*'):
        if path.is_file() and path.suffix in ['.py','.sh']:shutil.copy2(path,out/'scripts'/path.name)
    print('FIGURES_DONE',len(manifest),'tactics',len(rows),'tokens',len(tokens),'out',out,flush=True)
if __name__=='__main__':main()
