"""Explicit baseline contract entry; reject execution until its adapter is verified."""
import argparse
import json
from pathlib import Path

def main(run):
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','formal'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    config=json.loads((repo/'repro_configs/qat_baselines.json').read_text())
    contract={'run':run,'mode':args.mode,'output':args.output,
              'shared_contract':config['shared_contract'],'method_contract':config[run]}
    print(json.dumps(contract,indent=2),flush=True)
    if args.plan_only:
        return
    raise SystemExit('BLOCKED: '+ '; '.join(config[run]['blockers']))
