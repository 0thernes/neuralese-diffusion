# v2 UNet: same topology as v1 (so the JS engine stays generic), widths configurable. B=base ch, M=mid ch, E=embed dim.
import torch, torch.nn as nn, torch.nn.functional as F, math
CTX = 4
class Res(nn.Module):
    def __init__(s, ci, co, E):
        super().__init__(); s.n1=nn.GroupNorm(8,ci); s.c1=nn.Conv2d(ci,co,3,padding=1); s.t=nn.Linear(E,co)
        s.n2=nn.GroupNorm(8,co); s.c2=nn.Conv2d(co,co,3,padding=1); s.sk=nn.Conv2d(ci,co,1) if ci!=co else None
    def forward(s,x,e):
        h=s.c1(F.silu(s.n1(x))); h=h+s.t(e)[:,:,None,None]; h=s.c2(F.silu(s.n2(h)))
        return h+(s.sk(x) if s.sk is not None else x)
class Attn(nn.Module):
    def __init__(s, c, heads=4, ctx_dim=None):
        super().__init__(); s.h=heads; s.n=nn.GroupNorm(8,c); s.q=nn.Linear(c,c); kv=ctx_dim or c
        s.k=nn.Linear(kv,c); s.v=nn.Linear(kv,c); s.o=nn.Linear(c,c)
    def forward(s,x,ctx=None):
        B,C,H,W=x.shape; h=s.n(x).flatten(2).transpose(1,2); src = h if ctx is None else ctx
        q=s.q(h).view(B,-1,s.h,C//s.h).transpose(1,2); k=s.k(src).view(B,-1,s.h,C//s.h).transpose(1,2); v=s.v(src).view(B,-1,s.h,C//s.h).transpose(1,2)
        a=(q@k.transpose(-1,-2)/math.sqrt(C//s.h)).softmax(-1)
        o=(a@v).transpose(1,2).reshape(B,-1,C); return x+s.o(o).transpose(1,2).view(B,C,H,W)
def tembed(t, d=32):
    half=d//2; f=torch.exp(-math.log(10000)*torch.arange(half)/half); a=t.float()[:,None]*f[None]
    return torch.cat([torch.sin(a),torch.cos(a)],1)
class UNet(nn.Module):
    def __init__(s, C, ncls, B=64, M=128, E=256, heads=4):
        super().__init__(); s.ncls=ncls; s.M=M
        s.te1=nn.Linear(32,E); s.te2=nn.Linear(E,E); s.cls=nn.Embedding(ncls+1,E); s.ctx=nn.Embedding(ncls+1,CTX*M)
        s.cin=nn.Conv2d(C,B,3,padding=1); s.d1=Res(B,B,E); s.ds1=nn.Conv2d(B,M,3,stride=2,padding=1)
        s.d2=Res(M,M,E); s.ds2=nn.Conv2d(M,M,3,stride=2,padding=1)
        s.m1=Res(M,M,E); s.sa=Attn(M,heads); s.ca=Attn(M,heads,ctx_dim=M); s.m2=Res(M,M,E)
        s.us2=nn.Conv2d(M,M,3,padding=1); s.u2=Res(2*M,M,E); s.us1=nn.Conv2d(M,M,3,padding=1); s.u1=Res(M+B,B,E)
        s.no=nn.GroupNorm(8,B); s.cout=nn.Conv2d(B,C,3,padding=1)
    def forward(s,x,t,y,hook=None):
        e=s.te2(F.silu(s.te1(tembed(t))))+s.cls(y); e=F.silu(e); ctx=s.ctx(y).view(-1,CTX,s.M)
        h0=s.cin(x); h1=s.d1(h0,e); h=s.ds1(h1); h2=s.d2(h,e); h=s.ds2(h2)
        h=s.m1(h,e); h=s.sa(h); h=s.ca(h,ctx); h=s.m2(h,e)
        if hook: h=hook(h)
        s.mid=h
        h=s.us2(F.interpolate(h,size=h2.shape[-2:],mode='nearest')); h=s.u2(torch.cat([h,h2],1),e)
        h=s.us1(F.interpolate(h,size=h1.shape[-2:],mode='nearest')); h=s.u1(torch.cat([h,h1],1),e)
        return s.cout(F.silu(s.no(h)))
class Clf(nn.Module):
    """Judge: small CNN; penultimate 64-d features are used for FID-lite."""
    def __init__(s,C,ncls):
        super().__init__(); s.c1=nn.Conv2d(C,32,3,padding=1); s.c2=nn.Conv2d(32,64,3,stride=2,padding=1); s.c3=nn.Conv2d(64,64,3,stride=2,padding=1); s.fc=nn.Linear(64,ncls)
    def feat(s,x): h=F.relu(s.c1(x)); h=F.relu(s.c2(h)); h=F.relu(s.c3(h)); return h.mean((2,3))
    def forward(s,x): return s.fc(s.feat(x))
if __name__=='__main__':
    for C,H,W in [(1,28,28),(1,64,32),(8,32,32),(3,16,16)]:
        n=UNet(C,10); print(C,H,W,sum(p.numel() for p in n.parameters())); n(torch.randn(2,C,H,W),torch.tensor([3,4]),torch.tensor([1,10]))
