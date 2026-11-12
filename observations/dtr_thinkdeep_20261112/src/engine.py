import torch,time,math
from common import *
from dtr import jsd,certainty
class Engine:
    def __init__(self,name):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        assert torch.cuda.device_count()==1;self.name=name
        torch.set_num_threads(2);torch.manual_seed(2);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.80)
        self.inventory=read(W/'manifests/models.json');path=self.inventory[name]['path']
        self.tokenizer=AutoTokenizer.from_pretrained(self.inventory['P0']['path'],local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval();self.model.requires_grad_(False)
        ids=self.model.generation_config.eos_token_id;self.eos=set(ids if isinstance(ids,list) else [ids]);self.hidden={};self.hooks=[]
        for i,layer in enumerate(self.model.model.layers):
            def capture(module,args,out,i=i):
                h=out[0] if isinstance(out,tuple) else out;self.hidden[i+1]=h[:,-1:].clone()
            self.hooks.append(layer.register_forward_hook(capture))
        self.cache={}
    @torch.inference_mode()
    def step(self,prefix,verify=False):
        key=digest(prefix)
        if not verify and key in self.cache:return self.cache[key]
        start=time.monotonic();self.hidden={};x=torch.tensor([prefix],device='cuda')
        final=self.model.model(input_ids=x,use_cache=False).last_hidden_state[:,-1:];z=self.model.lm_head(final)[0,0].float()
        normalized=[];raw=[]
        for layer in range(1,29):
            h=self.hidden[layer];zn=self.model.lm_head(self.model.model.norm(h))[0,0].float();zr=self.model.lm_head(h)[0,0].float()
            normalized.append(zn);raw.append(zr)
        delta=(normalized[-1]-z).abs();assert float(delta.max())<=.125 and float(delta.mean())<=.01,'FINAL_NORM_LENS_PARITY'
        lp=z.log_softmax(-1);lr=raw[-1].log_softmax(-1)
        dn=[float(jsd(v.log_softmax(-1),lp)) for v in normalized];dr=[float(jsd(v.log_softmax(-1),lr)) for v in raw]
        assert dn[-1]<1e-5 and dr[-1]<1e-5
        out=dict(logits=z.cpu(),jsd=dn,jsd_raw=dr,entropy=float(-(lp.exp()*lp).sum()),self_certainty=float(certainty(lp)),
            lens_max_abs=float(delta.max()),lens_mean_abs=float(delta.mean()),logits_sha=digest(z.cpu().numpy().tobytes()),forward_s=time.monotonic()-start)
        if verify:
            d64=float(jsd(normalized[0].double().log_softmax(-1),z.double().log_softmax(-1)));assert abs(d64-dn[0])<1e-5
            out['jsd_fp64_abs_error']=abs(d64-dn[0])
        if len(self.cache)>=256:self.cache.pop(next(iter(self.cache)))
        self.cache[key]=out
        return out
    @torch.inference_mode()
    def native_parity(self):
        if self.name=='P0':return dict(status='REFERENCE')
        import ast
        tree=ast.parse((ROOT/'research/adapters/quantization.py').read_text())
        prompts=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPTS' for t in n.targets))
        path=ROOT/'runs/realprover_formal_four_gpu_v16/formal/P4/train/native_probe.pt';ref=torch.load(path,map_location='cpu',weights_only=True);records=[]
        for p,expected in zip(prompts,ref['logits']):
            actual=self.step(self.tokenizer.encode(p))['logits'];d=(actual-expected.float().flatten()).abs()
            record=dict(max_abs=float(d.max()),mean_abs=float(d.mean()),top1=int(actual.argmax())==int(expected.argmax()));records.append(record)
        assert all(r['max_abs']<=.125 and r['mean_abs']<=.01 and r['top1'] for r in records),'NATIVE_PARITY_FAILED'
        return dict(status='PASS',records=records,probe_sha=sha(path))
