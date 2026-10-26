# v2: train a ~2.5M-param class-conditional DDPM (eps-pred, T=200 linear beta, CFG null-class dropout 10%, EMA 0.999) on a REAL dataset
# under a wall-clock budget, then judge/probe/SAE/FID-lite, and export fp16 weights for the JS engine.
# usage: python train2.py <mod> <budget_seconds> <threads>
import sys, json, time, base64, hashlib, math, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model2 import UNet, Clf; from data2 import SPECS, load
mod=sys.argv[1]; budget=float(sys.argv[2]); torch.set_num_threads(int(sys.argv[3])); torch.manual_seed(0)
(X,Y),(Xte,Yte)=load(mod); sp=SPECS[mod]; C,H,W,K=sp['C'],sp['H'],sp['W'],sp['ncls']; T=200; BS=64
beta=torch.linspace(1e-4,0.02,T); ab=torch.cumprod(1-beta,0)
net=UNet(C,K); ema=copy.deepcopy(net).eval(); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=0)
def aug(x):
    if mod=='pc3d':  # random rotation about the vertical (z) axis would break the azimuth ordering; use small jitter + scale only
        return x*torch.empty(len(x),1,1,1).uniform_(0.9,1.1)+0.01*torch.randn_like(x)
    if mod=='image': return torch.where(torch.rand(len(x),1,1,1)<0.5, x.flip(-1), x)
    return x
log=[]; t0=time.time(); it=0
CK=f'/workspace/data/ckpt_{mod}_{int(budget)}.pt'  # checkpoint/resume so an interrupted run loses <=500 iterations
import os
if os.path.exists(CK):
    c=torch.load(CK); net.load_state_dict(c['net']); ema.load_state_dict(c['ema']); opt.load_state_dict(c['opt']); it=c['it']; log=c['log']; t0=time.time()-c['el']; torch.set_rng_state(c['rng']); print('resumed',mod,it,flush=True)
while True:
    el=time.time()-t0
    if el>budget: break
    fr=el/budget; lr=2e-3*min(1,(it+1)/300)*(0.5*(1+math.cos(math.pi*fr))*0.95+0.05)
    for g in opt.param_groups: g['lr']=lr
    i=torch.randint(0,len(X),(BS,)); x0=aug(X[i]); y=Y[i].clone(); y[torch.rand(BS)<0.1]=K
    t=torch.randint(0,T,(BS,)); n=torch.randn_like(x0); a=ab[t].view(-1,1,1,1)
    loss=F.mse_loss(net(a.sqrt()*x0+(1-a).sqrt()*n,t,y),n); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    with torch.no_grad():
        d=min(0.999,(1+it)/(10+it))
        for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(d).add_(pn,alpha=1-d)
    if it%200==0: log.append([it,round(el),float(loss)]); print(mod,it,f'{float(loss):.4f}',f'{el:.0f}s',flush=True)
    it+=1
    if it%500==0: torch.save(dict(net=net.state_dict(),ema=ema.state_dict(),opt=opt.state_dict(),it=it,log=log,el=time.time()-t0,rng=torch.get_rng_state()),CK)
train_s=time.time()-t0; steps=it; net=ema
# fp16 round-trip of the weights so PyTorch reference == what JS loads
with torch.no_grad():
    for p in net.parameters(): p.copy_(p.half().float())
torch.set_num_threads(int(sys.argv[3]))
g=torch.Generator().manual_seed(123)
with torch.no_grad():
    xs=Xte[:512]; t=torch.randint(0,T,(len(xs),),generator=g); n=torch.randn(xs.shape,generator=g); a=ab[t].view(-1,1,1,1)
    test_loss=float(F.mse_loss(net(a.sqrt()*xs+(1-a).sqrt()*n,t,Yte[:512]),n))
clf=Clf(C,K); o2=torch.optim.Adam(clf.parameters(),2e-3)
for k in range(3000):
    i=torch.randint(0,len(X),(128,)); x=aug(X[i])+0.05*torch.randn_like(X[i]); l=F.cross_entropy(clf(x),Y[i]); o2.zero_grad(); l.backward(); o2.step()
clf.eval()
with torch.no_grad(): clf_acc=float((clf(Xte).argmax(1)==Yte).float().mean())
with torch.no_grad():
    for p in clf.parameters(): p.copy_(p.half().float())
@torch.no_grad()
def ddim(y,steps=20,gs=3.0,seed=0):
    gg=torch.Generator().manual_seed(seed); x=torch.randn(len(y),C,H,W,generator=gg); ts=torch.linspace(T-1,0,steps).round().long()
    for j,tt in enumerate(ts):
        tb=torch.full((len(y),),int(tt)); ec=net(x,tb,y); eu=net(x,tb,torch.full_like(y,K)); e=eu+gs*(ec-eu)
        a=ab[tt]; x0=(x-(1-a).sqrt()*e)/a.sqrt(); x0=x0.clamp(-1,1)
        ap=ab[ts[j+1]] if j+1<steps else torch.tensor(1.0); x=ap.sqrt()*x0+(1-ap).sqrt()*e
    return x
def fid(A,B):
    from scipy import linalg
    A=A.double().numpy(); B=B.double().numpy(); m1,m2=A.mean(0),B.mean(0); s1,s2=np.cov(A,rowvar=False),np.cov(B,rowvar=False)
    cs=linalg.sqrtm(s1@s2); return float(((m1-m2)**2).sum()+np.trace(s1+s2-2*cs.real))
NG=500; yg=torch.arange(NG)%K; G=torch.cat([ddim(yg[i:i+100],seed=i) for i in range(0,NG,100)])
with torch.no_grad():
    gen_acc=float((clf(G).argmax(1)==yg).float().mean()); fg=clf.feat(G); fr_te=clf.feat(Xte[:NG]); fr_tr=clf.feat(X[torch.randperm(len(X))[:NG]])
    G0=torch.cat([ddim(yg[i:i+100],gs=0.0,seed=i) for i in range(0,NG,100)]); gen_acc_g0=float((clf(G0).argmax(1)==yg).float().mean()); fg0=clf.feat(G0)
fid_gen=fid(fg,fr_te); fid_real=fid(fr_tr,fr_te); fid_g0=fid(fg0,fr_te)
M=net.M
def midfeat(x0,t):
    n=torch.randn_like(x0); a=ab[t].view(-1,1,1,1)
    with torch.no_grad(): net(a.sqrt()*x0+(1-a).sqrt()*n,t,torch.full((len(x0),),K)); return net.mid
nm=min(3000,len(X)); idx=torch.randperm(len(X))[:nm]; xm=X[idx]; ym=Y[idx]
v=torch.cat([midfeat(xm[i:i+250],torch.full((len(xm[i:i+250]),),T//2)).mean((2,3)) for i in range(0,nm,250)])
means=torch.stack([v[ym==k].mean(0) for k in range(K)]); mall=v.mean(0)
pr=nn.Linear(M,K); o3=torch.optim.Adam(pr.parameters(),1e-2); mu,sd=v.mean(0),v.std(0)+1e-6
for k in range(800): l=F.cross_entropy(pr((v-mu)/sd),ym); o3.zero_grad(); l.backward(); o3.step()
nt=min(1000,len(Xte)); Mte=torch.cat([midfeat(Xte[i:i+250],torch.full((len(Xte[i:i+250]),),T//2)).mean((2,3)) for i in range(0,nt,250)])
with torch.no_grad(): probe_acc=float((pr((Mte-mu)/sd).argmax(1)==Yte[:nt]).float().mean())
tt=torch.randint(0,T,(nm,)); P=torch.cat([midfeat(xm[i:i+250],tt[i:i+250]).permute(0,2,3,1).reshape(-1,M) for i in range(0,nm,250)])
s=float(P.std()); Pn=P/s; SW=4*M; enc=nn.Linear(M,SW); dec=nn.Linear(SW,M); o4=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),1e-3); lam=1.0
for k in range(4000):
    b=Pn[torch.randint(0,len(Pn),(1024,))]; z=F.relu(enc(b)); r=dec(z); l=((r-b)**2).sum(1).mean()+lam*(z*dec.weight.norm(dim=0)).sum(1).mean(); o4.zero_grad(); l.backward(); o4.step()
with torch.no_grad():
    b=Pn[:8192]; z=F.relu(enc(b)); r=dec(z); sae_fvu=float(((r-b)**2).sum()/((b-b.mean(0))**2).sum()); sae_l0=float((z>0).float().sum(1).mean()); dead=int(((z>0).sum(0)==0).sum())
tensors={}
for k,w in net.state_dict().items(): tensors['net.'+k]=w
for k,w in clf.state_dict().items(): tensors['clf.'+k]=w
for k,w in pr.state_dict().items(): tensors['probe.'+k]=w
tensors['probe.mu']=mu; tensors['probe.sd']=sd
tensors['sae.enc.weight']=enc.weight.detach()/s; tensors['sae.enc.bias']=enc.bias.detach(); tensors['sae.dec.weight']=dec.weight.detach()*s; tensors['sae.dec.bias']=dec.bias.detach()*s
tensors['steer.means']=means; tensors['steer.mall']=mall
man=[]; buf=bytearray()
for k,w in tensors.items():
    a=w.detach().float().contiguous().numpy().astype('<f2'); man.append(dict(name=k,shape=list(a.shape),offset=len(buf)//2,size=a.size)); buf+=a.tobytes()
sha=hashlib.sha256(bytes(buf)).hexdigest()
# parity refs: reload the fp16-rounded probe/SAE too (net+clf already rounded)
g=torch.Generator().manual_seed(7); xr=torch.randn(1,C,H,W,generator=g)
with torch.no_grad(): er=net(xr,torch.tensor([57]),torch.tensor([1])); en=net(xr,torch.tensor([57]),torch.tensor([K])); cr=clf(xr)
nparam=sum(p.numel() for p in net.parameters())
meta=dict(version=2, modality=mod, **sp, T=T, beta=[1e-4,0.02], B=64, M=M, E=256, heads=4, ctx=4, dtype='f16', params_unet=nparam, params_clf=sum(p.numel() for p in clf.parameters()),
  bytes=len(buf), sha256=sha, train_steps=steps, batch=BS, train_seconds=round(train_s,1), threads=int(sys.argv[3]), ema=0.999, final_train_loss=log[-1][2], heldout_loss=test_loss, loss_curve=log,
  clf_heldout_acc=clf_acc, gen_judge_acc_g3=gen_acc, gen_judge_acc_g0=gen_acc_g0, fid_lite=dict(gen_g3=fid_gen, gen_g0=fid_g0, real_train_vs_test=fid_real, n=NG, features='judge penultimate 64-d', sampler='DDIM-20'),
  probe_heldout_acc_tHalf=probe_acc, sae=dict(width=SW,lam=lam,fvu=sae_fvu,l0=sae_l0,dead=dead), torch=torch.__version__,
  ref=dict(x=xr.flatten().tolist(), t=57, y=1, eps_cond=er.flatten().tolist(), eps_null=en.flatten().tolist(), clf=cr.flatten().tolist()), manifest=man)
import os; os.makedirs('../site/models2',exist_ok=True)
b64=base64.b64encode(bytes(buf)).decode(); P=1900000; parts=[b64[i:i+P] for i in range(0,len(b64),P)]
for i,pp in enumerate(parts): open(f'../site/models2/{mod}.{i}.b64','w').write(pp)
meta['b64parts']=len(parts); json.dump(meta,open(f'../site/models2/{mod}.json','w'))
torch.save(dict(G=G[:40],yg=yg[:40]),f'/workspace/data/samples_{mod}.pt')
print('DONE',mod,json.dumps({k:meta[k] for k in ['params_unet','train_steps','train_seconds','heldout_loss','clf_heldout_acc','gen_judge_acc_g3','gen_judge_acc_g0','fid_lite','probe_heldout_acc_tHalf','sae']}))
