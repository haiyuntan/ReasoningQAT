import json,sys
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
python=OLD/'envs/forward/bin/python'
result=guard.supervise([str(python),str(OUT/'run.py')],guard.baseline(),'qwen3_fig2b_v2')
print(json.dumps(result),flush=True)
