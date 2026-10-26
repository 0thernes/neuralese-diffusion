# Train one tiny class-conditional DDPM + a tiny classifier + probe + SAE for one modality, export for the JS engine.
import sys, json, time, base64, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import UNet, Clf; from data import SPECS, load
mod = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv)>2 else 3000; torch.set_num_threads(int(sys.argv[3]) if len(sys.argv)>3 else 2)
torch.manual_seed(0); sp = SPECS[mod]; C,H,W,K = sp['C'],sp['H'],sp['W'],sp['ncls']; T=200
X,Y = load(mod); Xte,Yte = (load(mod, 2000, seed=1) if mod!='image' else (X[-2000:],Y[-2000:]))
if mod=='image': X,Y = X[:-2000],Y[:-2000]
beta = torch.linspace(1e-4,0.02,T); ab = torch.cumprod(1-beta,0)
net = UNet(C,K); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.05)
log=[]; t0=time.time()
for it in range(steps):
    i = torch.randint(0,len(X),(128,)); x0=X[i]; y=Y[i].clone(); y[torch.rand(128)<0.1]=K
    t=torch.randint(0,T,(128,)); n=torch.randn_like(x0); a=ab[t].view(-1,1,1,1); xt=a.sqrt()*x0+(1-a).sqrt()*n
    loss=F.mse_loss(net(xt,t,y),n); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
    if it%200==0 or it==steps-1: log.append([it,float(loss)]); print(mod,it,float(loss),f'{time.time()-t0:.0f}s',flush=True)
train_s=time.time()-t0; net.eval()
# held-out denoising loss at fixed noise
g=torch.Generator().manual_seed(123)
with torch.no_grad():
    xs=Xte[:512]; t=torch.randint(0,T,(512,),generator=g); n=torch.randn(xs.shape,generator=g); a=ab[t].view(-1,1,1,1)
    test_loss=float(F.mse_loss(net((a.sqrt()*xs+(1-a).sqrt()*n),t,Yte[:512]),n))
# classifier (on clean data) — the "judge" for generated samples
clf=Clf(C,K); o2=torch.optim.Adam(clf.parameters(),2e-3)
for it in range(1500):
    i=torch.randint(0,len(X),(128,)); x=X[i]+0.05*torch.randn_like(X[i]); l=F.cross_entropy(clf(x),Y[i]); o2.zero_grad(); l.backward(); o2.step()
with torch.no_grad(): clf_acc=float((clf(Xte).argmax(1)==Yte).float().mean())
# mid-block features under the NULL class on noised held-in data (no label leakage into the net)
def midfeat(x0,t):
    n=torch.randn_like(x0); a=ab[t].view(-1,1,1,1)
    with torch.no_grad(): net(a.sqrt()*x0+(1-a).sqrt()*n,t,torch.full((len(x0),),K)); return net.mid
idx=torch.randperm(len(X))[:4000]; xm=X[idx]; ym=Y[idx]
tmid=torch.full((4000,),T//2); M=midfeat(xm,tmid); v=M.mean((2,3))
means=torch.stack([v[ym==k].mean(0) for k in range(K)]); mall=v.mean(0)
# linear probe (multinomial logistic regression) at t=T/2, evaluated on held-out
pr=nn.Linear(64,K); o3=torch.optim.Adam(pr.parameters(),1e-2)
mu,sd=v.mean(0),v.std(0)+1e-6
for it in range(800): l=F.cross_entropy(pr((v-mu)/sd),ym); o3.zero_grad(); l.backward(); o3.step()
Mte=midfeat(Xte[:1000],torch.full((1000,),T//2)).mean((2,3))
with torch.no_grad(): probe_acc=float((pr((Mte-mu)/sd).argmax(1)==Yte[:1000]).float().mean())
# SAE on per-position mid vectors at mixed t
tt=torch.randint(0,T,(4000,)); P=midfeat(xm,tt).permute(0,2,3,1).reshape(-1,64)
sae_W=nn.Linear(64,256); sae_D=nn.Linear(256,64); o4=torch.optim.Adam(list(sae_W.parameters())+list(sae_D.parameters()),1e-3)
for it in range(2000):
    b=P[torch.randint(0,len(P),(1024,))]; z=F.relu(sae_W(b)); r=sae_D(z); l=F.mse_loss(r,b)+5e-2*z.abs().mean()
    o4.zero_grad(); l.backward(); o4.step()
with torch.no_grad():
    b=P[:4096]; z=F.relu(sae_W(b)); r=sae_D(z); sae_fvu=float(((r-b)**2).sum()/((b-b.mean(0))**2).sum()); sae_l0=float((z>0).float().sum(1).mean())
# export
tensors={}
for k,w in net.state_dict().items(): tensors['net.'+k]=w
for k,w in clf.state_dict().items(): tensors['clf.'+k]=w
for k,w in pr.state_dict().items(): tensors['probe.'+k]=w
tensors['probe.mu']=mu; tensors['probe.sd']=sd
for k,w in sae_W.state_dict().items(): tensors['sae.enc.'+k]=w
for k,w in sae_D.state_dict().items(): tensors['sae.dec.'+k]=w
tensors['steer.means']=means; tensors['steer.mall']=mall
man=[]; buf=bytearray()
for k,w in tensors.items():
    a=w.detach().float().contiguous().numpy().astype('<f4'); man.append(dict(name=k,shape=list(a.shape),offset=len(buf)//4,size=a.size)); buf+=a.tobytes()
sha=hashlib.sha256(bytes(buf)).hexdigest()
# parity reference: fixed input through the trained net
g=torch.Generator().manual_seed(7); xr=torch.randn(1,C,H,W,generator=g)
with torch.no_grad(): er=net(xr,torch.tensor([57]),torch.tensor([1])); en=net(xr,torch.tensor([57]),torch.tensor([K])); cr=clf(xr)
nparam=sum(p.numel() for p in net.parameters())
meta=dict(modality=mod, **sp, T=T, beta=[1e-4,0.02], params_unet=nparam, params_clf=sum(p.numel() for p in clf.parameters()), bytes=len(buf), sha256=sha,
  train_steps=steps, batch=128, train_seconds=round(train_s,1), final_train_loss=log[-1][1], heldout_loss=test_loss, loss_curve=log, clf_heldout_acc=clf_acc,
  probe_heldout_acc_tHalf=probe_acc, sae=dict(width=256,l1=5e-2,fvu=sae_fvu,l0=sae_l0), torch=torch.__version__,
  ref=dict(x=xr.flatten().tolist(), t=57, y=1, eps_cond=er.flatten().tolist(), eps_null=en.flatten().tolist(), clf=cr.flatten().tolist()), manifest=man)
import os; os.makedirs('../site/models',exist_ok=True)
open(f'../site/models/{mod}.b64','w').write(base64.b64encode(bytes(buf)).decode())
json.dump(meta,open(f'../site/models/{mod}.json','w'))
print('DONE',mod,json.dumps({k:meta[k] for k in ['params_unet','bytes','train_seconds','heldout_loss','clf_heldout_acc','probe_heldout_acc_tHalf','sae']}))
