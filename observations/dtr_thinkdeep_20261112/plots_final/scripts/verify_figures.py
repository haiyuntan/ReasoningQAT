"""Independent stdlib check of exports, immutable sources and plotted summaries."""
import argparse,csv,json,hashlib,math
from pathlib import Path
W=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',default=str(W/'plots_final'));a=p.parse_args();out=Path(a.outdir)
    manifest=read(out/'tables/figure_source_manifest.json');stats=read(out/'tables/figure_statistics_summary.json')
    for name,h in manifest['source_sha256'].items():assert sha(W/name)==h,name
    assert sha(W/'src/figures/build_all.py')==manifest['script_sha256']
    with (W/'metrics/per_tactic_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    with (W/'metrics/per_token_metrics.csv').open() as f:tokens=list(csv.DictReader(f))
    layers=[json.loads(s) for s in (W/'metrics/per_layer_jsd.jsonl').read_text().splitlines()]
    assert len(rows)==manifest['derived_tactic_rows'] and len(tokens)==manifest['derived_token_rows'] and len(layers)==manifest['derived_layer_records']
    index={(r['candidate_id'],r['model']):r for r in rows}
    for r in layers:
        matrix=r['jsd_matrix'];assert len(matrix)==28 and all(len(x)==len(r['token_ids']) for x in matrix)
        assert all(abs(x)<1e-5 for x in matrix[-1]);assert all(math.isfinite(v) and -1e-5<=v<=math.log(2)+1e-5 for line in matrix for v in line)
        depths=[next(i+1 for i in range(28) if matrix[i][t]<=.5) for t in range(len(matrix[0]))]
        value=sum(d>=24 for d in depths)/len(depths);assert abs(value-r['dtr'])<1e-10
        csvrow=index[r['candidate_id'],r['model']];assert abs(float(csvrow['dtr'])-value)<1e-10
        if r['model']=='P4':
            paired=index[r['candidate_id'],'P0'];assert paired['theorem_id']==csvrow['theorem_id'] and paired['state_id']==csvrow['state_id'] and paired['lean_status']==csvrow['lean_status'] and paired['token_length']==csvrow['token_length']
    for figure,r in manifest['figures'].items():
        for ext in ['png','pdf']:assert (out/(figure+'.'+ext)).stat().st_size>1000,figure
        for name in r['data_files']:assert name in manifest['source_sha256']
    assert len(manifest['figures'])==8
    for key,result in stats['Fig1']['metrics'].items():assert sum(r['n_tactics'] for r in result['bins'])==160,key
    primary=read(W/'metrics/summaries/D-CONFIRM.json')
    for name,m in [('dtr','dtr'),('self_certainty','self_certainty'),('avg_logprob','mean_logprob'),('token_length','length')]:
        drawn=stats['Fig3']['targets']['is_valid_transition']['P4'][name];saved=primary['baselines'][m]
        assert abs(drawn['mean']-saved['mean'])<1e-12 and drawn['ci']==saved['ci'] and drawn['n_theorems']==saved['n']
    cases=read(out/'tables/selected_case_manifest.json');assert cases['V1']['status']=='NOT_APPLICABLE' and cases['V2']['status']=='NOT_APPLICABLE'
    for tag in ['V0','V3']:assert cases[tag]['split']=='D-CONFIRM'
    result=dict(status='PASS',source_hashes=len(manifest['source_sha256']),tactic_rows=len(rows),token_rows=len(tokens),full_layer_records=len(layers),figures=8,P4_primary_intervals='UNCHANGED_EXACT',same_token_pairing='PASS')
    (out/'tables/validation_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
