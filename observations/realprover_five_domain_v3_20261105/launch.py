import sys,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
OLD=ROOT/'observation_20261104_all_layers'
sys.path.insert(0,str(OLD/'src'))
import common
common.OBS=OUT
import resource_guard as guard
guard.CAMPAIGN=ROOT/'runs/realprover_formal_four_gpu_v15'
for name in ['configs','manifests','logs','tmp','cache']:(OUT/name).mkdir(exist_ok=True)
protocol=json.loads((OLD/'configs/observation_protocol.json').read_text())
(OUT/'configs/observation_protocol.json').write_text(json.dumps(protocol,indent=2))
assert (OUT/'inputs.jsonl').exists()
result=guard.supervise([str(OLD/'envs/forward/bin/python'),str(OUT/'experiment.py')],guard.baseline(),'realprover_five_domain')
print(json.dumps(result),flush=True)
