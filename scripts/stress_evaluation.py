"""Isolated vLLM scheduling benchmark with retrieval resident; never an evaluation receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
PROVER = '/HOME/sysu_qling/sysu_qling_3/HDD_POOL/anaconda3/envs/PROVER/bin/python'


def gpu():
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    return [float(x) for x in line.split(',')]


def worker(a):
    from vllm import LLM, SamplingParams
    samples = []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            samples.append(gpu())
            stop.wait(0.25)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    result = dict(memory_utilization=a.memory, max_num_seqs=a.seqs, status='RUNNING', cases=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        started = time.monotonic()
        llm = LLM(model=str(ROOT/'models/REAL-Prover-v1-bf16'), seed=2,
                  gpu_memory_utilization=a.memory, max_model_len=8192,
                  max_num_seqs=a.seqs, max_num_batched_tokens=8192,
                  trust_remote_code=True)
        result['init_seconds'] = time.monotonic()-started
        # Fixed token lengths; synthetic requests isolate scheduler/VRAM capacity.
        # Native formal shape is one state with n=64, never extra search samples.
        for label, length, count, n in [('smoke',1024,1,4), ('formal',1024,1,64),
                                      ('long_formal',7936,1,64), ('saturated',4096,32,4)]:
            prompts = [{'prompt_token_ids': ([151644]+[1000,2000,3000,4000]*((length+3)//4))[:length]}
                       for _ in range(count)]
            params = SamplingParams(n=n, temperature=1.5, top_p=0.9, logprobs=1,
                                    max_tokens=256, min_tokens=256, ignore_eos=True, seed=2)
            start_index = len(samples)
            started = time.monotonic()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            seconds = time.monotonic()-started
            tokens = sum(len(o.token_ids) for request in outputs for o in request.outputs)
            assert len(outputs)==count and all(len(x.outputs)==n for x in outputs)
            assert tokens==count*n*256
            measurements = samples[start_index:]
            row = dict(case=label, prompt_tokens=length, requests=count, n=n, output_tokens=tokens,
                       seconds=seconds, output_tokens_per_second=tokens/seconds,
                       peak_total_mib=max((x[0] for x in measurements), default=gpu()[0]),
                       mean_gpu_utilization=sum(x[2] for x in measurements)/max(1,len(measurements)))
            result['cases'].append(row)
            a.output.write_text(json.dumps(result,indent=2))
            print(json.dumps(row),flush=True)
        result['status']='PASS'
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        result['peak_total_mib']=max((x[0] for x in samples),default=0)
        result['minimum_free_mib']=min((x[1]-x[0] for x in samples),default=0)
        result['samples']=len(samples)
        a.output.write_text(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker',action='store_true')
    ap.add_argument('--memory',type=float)
    ap.add_argument('--seqs',type=int)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.worker:
        worker(a)
        return
    import fcntl
    with (ROOT/'runs/realprover_pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.output.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                 TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        results=[]
        for memory,seqs in [(0.65,64),(0.70,128),(0.75,256)]:
            name=f'm{memory}_s{seqs}'
            with (a.output/(name+'.log')).open('w') as log:
                code=subprocess.call([PROVER,str(Path(__file__).resolve()),'--worker','--memory',str(memory),
                                      '--seqs',str(seqs),'--output',str(a.output/(name+'.json'))],
                                     env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            results.append(dict(profile=name,exit_code=code))
            (a.output/'status.json').write_text(json.dumps(dict(status='RUNNING',profiles=results),indent=2))
            # Stop escalating if capacity has failed; no deliberate OOM search.
            if code:
                break
        (a.output/'status.json').write_text(json.dumps(dict(status='COMPLETE',profiles=results),indent=2))


if __name__=='__main__':
    main()
