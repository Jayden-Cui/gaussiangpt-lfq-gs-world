"""Portable bindings: all runtime writes are below /data/CJH/1 on server 75."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,types
import torch

ROOT=Path(__file__).resolve().parent
ASSETS=Path(os.environ.get('GS_CAPACITY_ASSETS','/data/CJH/1/assets'))
REFERENCE=ASSETS/'reference/outputs'
FULL=REFERENCE/'gaussiangpt_neoverse_qformer_full177_20261101_v1'
sys.path.insert(0,str(FULL))
from support_decoder import Condition,state_loss,select_state,support_metrics
from core import image_losses,image_metrics,object_proxy

def require(x,message):
    if not x:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(data);temp.replace(path)
    else:
        with path.open('x') as f:f.write(data)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        t=t.detach().cpu().contiguous();h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def tree(v,device):
    if isinstance(v,torch.Tensor):return v.detach().to(device).contiguous()
    if isinstance(v,dict):return {k:tree(x,device) for k,x in v.items()}
    if isinstance(v,list):return [tree(x,device) for x in v]
    if isinstance(v,tuple):return tuple(tree(x,device) for x in v)
    return v
def verify_assets(full=False):
    require(sha(ASSETS/'MANIFEST.json')=='3702a70b2f13dc9f214700f5f6601d873829bbc6ba4ee289933200c87fc14b5e','Portable asset manifest changed')
    m=read(ASSETS/'MANIFEST.json')
    require(m['status']=='complete' and m['starts']==list(range(168)) and m['frames']==177,'Incomplete assets')
    for row in m['files']+(m['windows'] if full else []):
        p=ASSETS/row['file'];require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Asset changed: '+row['file'])
    return m
def reference_model():
    formal=REFERENCE/'gaussiangpt_neoverse_z_paired_formal_20261031_v1';sys.path.insert(0,str(formal))
    from model_adapter import load_pinned,attach_adapters
    api=load_pinned(REFERENCE)
    base=api.AnchoredLDM(blocks=4,checkpoint_blocks=True,residual_scale=3.,inv_temperature=1.,code_mode='posterior')
    api.attach_spatial_action(base,True);attach_adapters(base,mode='continuous16',s_ref=read(ASSETS/'original_protocol.json')['s_ref'])
    from support_decoder import SupportDecoder
    reference=SupportDecoder(base,'free16',.1)
    require(sha(ASSETS/'initial.pt')=='3893e488174f5c44fc8b696526af71d6ba718ba073bbda936d1960644fbc2699','Initial checkpoint changed')
    payload=torch.load(ASSETS/'initial.pt',map_location='cpu',weights_only=True)
    reference.load_state_dict(payload['decoder'],strict=True)
    require(digest(reference)==payload['decoder_digest'],'Reference decoder differs')
    return reference.eval().requires_grad_(False),payload

class Store:
    def __init__(self):
        self.manifest=read(ASSETS/'MANIFEST.json');self.cached={}
        self.frames=torch.load(ASSETS/'data/frames.pt',map_location='cpu',weights_only=True)
        self.p=read(ASSETS/'original_protocol.json');self.roi=torch.zeros(252,448,dtype=torch.bool)
        for x0,y0,x1,y1 in self.p['hand_roi_xyxy']:self.roi[y0:y1,x0:x1]=True
    def get(self,start,device='cuda'):
        require(type(start)==int and 0<=start<168,'Window out of range')
        if start not in self.cached:
            row=self.manifest['windows'][start];p=ASSETS/row['file']
            require(sha(p)==row['sha256'],'Window changed')
            self.cached[start]=torch.load(p,map_location='cpu',weights_only=True)
        w=tree(self.cached[start],device);w['condition']=Condition(**w['condition']);w['roi']=self.roi.to(device);return w
    def sample(self,start,h,device='cuda'):
        s=tree(self.frames['samples'][start+h],device)
        initial=self.frames['samples'][start]['record']['real'].to(device)
        s['motion']=((s['record']['real']-initial).abs().mean(-1)/.1).clamp(0,1);return s

def native_modules():
    root=ASSETS/'codec'
    common=module('common',root/'common.py')
    common.VENDOR=root/'vendor';common.CHECKPOINT=root/'official.ckpt'
    # Import the hash-bound existing extension before gsplat selects a JIT cache.
    for p in (ASSETS/'extensions').glob('*.so'):
        name=p.name.split('.')[0];ext=module(name,p)
        import gsplat
        gsplat.csrc=ext;sys.modules['gsplat.csrc']=ext
    return common,root

class NativeBridge(torch.nn.Module):
    def __init__(self,device='cuda'):
        super().__init__();common,root=native_modules()
        codec=module('codec',root/'codec.py');precision=module('_capacity_native_precision',root/'precision.py')
        import MinkowskiEngine as ME
        self.ME=ME;self.codec=codec.load_codec(root,root/'codec_final.pt',device=device).eval().requires_grad_(False)
        self.precision=precision.DecoderPrecision(self.codec,'float64')
        self.prediction_gs,self.render_fn=common.prediction_gs,common.render
        self.frozen_digest=digest(self.codec)
        require(isinstance(self.codec.ae.vq,torch.nn.Identity) and self.codec.ae.stride==4 and self.codec.grid==.008,'Native codec interface changed')
    def train(self,mode=True):
        super().train(mode);self.codec.eval();return self
    def assert_frozen(self):
        require(digest(self.codec)==self.frozen_digest and not any(p.requires_grad or p.grad is not None for p in self.codec.parameters()),'Native codec changed')
        require(not self.codec.training and self.ME.global_coordinate_manager() is None and not self.ME.is_deterministic(),'Native mode changed')
    def decode_points(self,coordinates,features):
        require(features.shape==(len(coordinates),64) and features.dtype==torch.float32,'Wrong state latent interface')
        sparse=self.codec._sparse(coordinates,features,4)
        prepared=self.precision.prepare_sparse(sparse,self.ME)
        points,_=self.codec.ae.decode(prepared,target_key=None)
        return self.precision.to_render_points(self.codec._world(points))
    def render(self,state,record,views=(0,)):
        gs,sh=self.prediction_gs(self.decode_points(state['coordinates'],state['features']))
        require(0<len(gs)<=2400000 and sh.shape==(len(gs),4,3),'Invalid native output')
        images=torch.stack([self.render_fn(gs,record['cameras'][v],record['intrinsics'],record['near'],record['far'],sh)[0].clamp(0,1) for v in views])
        require(bool(torch.isfinite(images).all()),'Nonfinite renderer');return images,gs,sh
