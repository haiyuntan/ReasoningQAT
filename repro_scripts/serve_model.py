"""One local generation protocol for BF16, GPTQ and materialized W2 QAT/AWQ."""
import argparse
import json
import threading
import time
import uuid
from pathlib import Path
import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM,AutoTokenizer
import uvicorn

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--backend',choices=['hf','gptq'],default='hf')
p.add_argument('--port',type=int,default=18080)
p.add_argument('--telemetry',required=True)
a=p.parse_args()
tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
if a.backend=='gptq':
    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND
    model=GPTQModel.load(a.model,backend=BACKEND.TORCH,device='cuda:0')
else:
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
app=FastAPI()
lock=threading.Lock()

@app.get('/v1/models')
def models():
    return {'object':'list','data':[{'id':'rq-model','object':'model'}]}

@app.post('/v1/chat/completions')
def completions(body:dict):
    if body.get('stream'):
        raise HTTPException(400,'This serial validation server supports stream=false')
    with lock:
        seed=int(body.get('seed',42))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        thinking=body.get('chat_template_kwargs',{}).get('enable_thinking',True)
        prompt=tokenizer.apply_chat_template(body['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
        inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
        start=time.monotonic()
        with torch.no_grad():
            generated=model.generate(**inputs,max_new_tokens=int(body.get('max_tokens',32768)),do_sample=True,
                                     temperature=float(body.get('temperature',0.6)),top_k=int(body.get('top_k',20)),
                                     top_p=float(body.get('top_p',1.0)),pad_token_id=tokenizer.eos_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
        new=generated[0,inputs['input_ids'].shape[1]:]
        content=tokenizer.decode(new,skip_special_tokens=True)
        stops = body.get('stop') or []
        if isinstance(stops, str):
            stops = [stops]
        for stop in stops:
            content=content.split(stop,1)[0]
        telemetry={'peak_memory_bytes':torch.cuda.max_memory_allocated(),'generation_seconds':time.monotonic()-start,
                   'prompt_tokens':inputs['input_ids'].shape[1],'completion_tokens':len(new),'seed':seed,
                   'temperature':body.get('temperature',0.6),'top_k':body.get('top_k',20),'thinking':thinking}
        with Path(a.telemetry).open('a') as f:
            f.write(json.dumps(telemetry)+'\n')
        # The final answer and reasoning remain in raw output; scoring uses the same EvalScope adapters.
        return {'id':'chatcmpl-'+uuid.uuid4().hex,'object':'chat.completion','created':int(time.time()),'model':'rq-model',
                'choices':[{'index':0,'message':{'role':'assistant','content':content},
                            'finish_reason':'stop' if new[-1].item()==tokenizer.eos_token_id else 'length'}],
                'usage':{'prompt_tokens':inputs['input_ids'].shape[1],'completion_tokens':len(new),
                         'total_tokens':generated.shape[1]}}

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=a.port)
