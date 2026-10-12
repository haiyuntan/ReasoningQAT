"""Load pinned upstream operators without importing their legacy training stacks."""
import ast
import gc
import importlib.util
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def efficient_ops():
    # A private package avoids accidentally loading ReasoningQAT's quantizer.
    path=ROOT/'qat_baselines/EfficientQAT/quantize'
    import types
    package=types.ModuleType('rq_efficient_quantize')
    package.__path__=[str(path)]
    sys.modules[package.__name__]=package
    quantizer=load_module('rq_efficient_quantize.quantizer',path/'quantizer.py')
    # These modules use absolute quantize imports. Redirect only their imports.
    def adapted(name,filename):
        source=(path/filename).read_text().replace('from quantize.', 'from rq_efficient_quantize.')
        module=types.ModuleType(name)
        module.__file__=str(path/filename)
        sys.modules[name]=module
        exec(compile(source,str(path/filename),'exec'),module.__dict__)
        return module
    fake=adapted('rq_efficient_quantize.int_linear_fake','int_linear_fake.py')
    adapted('rq_efficient_quantize.utils','utils.py')
    real=adapted('rq_efficient_quantize.int_linear_real','int_linear_real.py')
    return fake.QuantLinear,real.QuantLinear,quantizer

def unpack_dim0(packed,bits,rows):
    shifts=torch.arange(0,32,bits,device=packed.device,dtype=torch.int64)
    codes=((packed.to(torch.int64).unsqueeze(1)>>shifts.view(1,-1,1))&((1<<bits)-1))
    return codes.reshape(-1,packed.shape[1])[:rows]

def unpack_dim1(packed,bits,columns):
    shifts=torch.arange(0,32,bits,device=packed.device,dtype=torch.int64)
    codes=((packed.to(torch.int64).unsqueeze(-1)>>shifts)&((1<<bits)-1))
    return codes.reshape(packed.shape[0],-1)[:,:columns]

def packed_weight(module):
    codes=unpack_dim0(module.qweight,module.bits,module.infeatures)
    zeros=unpack_dim1(module.qzeros,module.bits,module.outfeatures)
    return ((codes.reshape(-1,module.group_size,module.outfeatures).to(module.scales.dtype)
             -zeros.unsqueeze(1))*module.scales.unsqueeze(1)).reshape(module.infeatures,module.outfeatures)

def portable_packed_class():
    _,original,_=efficient_ops()
    class PortablePackedLinear(original):
        """Upstream packing/state layout, differentiable PyTorch dequantization."""
        def forward(self,x):
            weight=packed_weight(self)
            # Canonical F.linear layout matches the materialized HF module.
            # x@W and F.linear(x,W.T) select different BF16 GEMM layouts on
            # CUDA, which can accumulate visible errors across 28 blocks.
            return F.linear(x,weight.t().contiguous().to(x.dtype),
                            None if self.bias is None else self.bias.to(x.dtype))
    return PortablePackedLinear

def bitdistiller_ops():
    path=ROOT/'qat_baselines/BitDistiller'
    quant=load_module('rq_bit_quantizer',path/'quantization/quantizer.py')
    import types
    linear=types.ModuleType('rq_bit_linear')
    source=(path/'quantization/qlinear.py').read_text().replace('from quantizer import *','from rq_bit_quantizer import *')
    exec(compile(source,str(path/'quantization/qlinear.py'),'exec'),linear.__dict__)
    tree=ast.parse((path/'train/mytrainer.py').read_text())
    trainer=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='KDTrainer')
    method=next(node for node in trainer.body if isinstance(node,ast.FunctionDef) and node.name=='cakld_loss')
    namespace={'F':F}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'upstream_cakld','exec'),namespace)
    # Extract the original independent positive/negative clipping grid. Only
    # guard the sample stride, which is zero on short diagnostic sequences.
    source=(path/'quantization/autoclip.py').read_text()
    source=source.replace('input_feat.shape[1] // n_sample_token','max(1, input_feat.shape[1] // n_sample_token)')
    tree=ast.parse(source)
    clip=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='auto_2clip_layer')
    clip_ns={'torch':torch,'gc':gc,'pseudo_quantize_tensor':quant.pseudo_quantize_tensor,
             'pseudo_quantize_n2f3_tensor':quant.pseudo_quantize_n2f3_tensor}
    exec(compile(ast.Module(body=[clip],type_ignores=[]),'upstream_asymmetric_clip','exec'),clip_ns)
    return linear,quant,namespace['cakld_loss'],clip_ns['auto_2clip_layer']

def set_module(model,name,module):
    parent,_,leaf=name.rpartition('.')
    target=model.get_submodule(parent) if parent else model
    setattr(target,leaf,module)
