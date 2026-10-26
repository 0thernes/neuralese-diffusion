# Retrain the bottleneck SAE (replaces the weak first-pass SAE in train.py) and rewrite the export in place.
import sys, json, base64, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import UNet; from data import SPECS, load
mod=sys.argv[1]; torch.set_num_threads(2); torch.manual_seed(1)
meta=json.load(open(f'../site/models/{mod}.json')); buf=np.frombuffer(base64.b64decode(open(f'../site/models/{mod}.b64').read()),'<f4').copy()
get=lambda n: next(m for m in meta['manifest'] if m['name']==n)
net=UNet(meta['C'],meta['ncls']); sd={}
for k in net.state_dict(): m=get('net.'+k); sd[k]=torch.tensor(buf[m['offset']:m['offset']+m['size']]).view(m['shape'])
net.load_state_dict(sd); net.eval(); K=meta['ncls']; T=200; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T),0)
X,Y=load(mod); idx=torch.randperm(len(X))[:4000]; x0=X[idx]; t=torch.randint(0,T,(4000,)); n=torch.randn_like(x0); a=ab[t].view(-1,1,1,1)
with torch.no_grad(): net(a.sqrt()*x0+(1-a).sqrt()*n,t,torch.full((4000,),K))
P=net.mid.permute(0,2,3,1).reshape(-1,64); s=float(P.std()); Pn=P/s
enc=nn.Linear(64,256); dec=nn.Linear(256,64,bias=True); opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),1e-3); lam=float(sys.argv[2]) if len(sys.argv)>2 else 3e-2
for it in range(4000):
    b=Pn[torch.randint(0,len(Pn),(1024,))]; z=F.relu(enc(b)); r=dec(z); l=((r-b)**2).sum(1).mean()+lam*(z*dec.weight.norm(dim=0)).sum(1).mean()
    opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad():
    b=Pn[:8192]; z=F.relu(enc(b)); r=dec(z); fvu=float(((r-b)**2).sum()/((b-b.mean(0))**2).sum()); l0=float((z>0).float().sum(1).mean()); dead=int(((z>0).sum(0)==0).sum())
# fold the input scale into the encoder so JS can feed raw activations: z = relu(W (x/s) + b)
W=enc.weight.detach()/s
for name,val in [('sae.enc.weight',W),('sae.enc.bias',enc.bias.detach()),('sae.dec.weight',dec.weight.detach()*s),('sae.dec.bias',dec.bias.detach()*s)]:
    m=get(name); buf[m['offset']:m['offset']+m['size']]=val.numpy().astype('<f4').ravel()
by=buf.astype('<f4').tobytes(); meta['sha256']=hashlib.sha256(by).hexdigest(); meta['sae']=dict(width=256,lam=lam,fvu=fvu,l0=l0,dead=dead,note='L1 weighted by decoder norms, inputs scaled by global std (folded into weights)')
open(f'../site/models/{mod}.b64','w').write(base64.b64encode(by).decode()); json.dump(meta,open(f'../site/models/{mod}.json','w')); print(mod,meta['sae'])
