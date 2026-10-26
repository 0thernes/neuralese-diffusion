import torch, torch.nn as nn, torch.nn.functional as F, math
CTX = 4  # cross-attention context tokens per class
class Res(nn.Module):
    def __init__(s, ci, co, E=128):
        super().__init__(); s.n1=nn.GroupNorm(8,ci); s.c1=nn.Conv2d(ci,co,3,padding=1); s.t=nn.Linear(E,co)
        s.n2=nn.GroupNorm(8,co); s.c2=nn.Conv2d(co,co,3,padding=1); s.sk=nn.Conv2d(ci,co,1) if ci!=co else None
    def forward(s,x,e):
        h=s.c1(F.silu(s.n1(x))); h=h+s.t(e)[:,:,None,None]; h=s.c2(F.silu(s.n2(h)))
        return h+(s.sk(x) if s.sk is not None else x)
class Attn(nn.Module):
    def __init__(s, c, heads=2, ctx_dim=None):
        super().__init__(); s.h=heads; s.n=nn.GroupNorm(8,c); s.q=nn.Linear(c,c); kv=ctx_dim or c
        s.k=nn.Linear(kv,c); s.v=nn.Linear(kv,c); s.o=nn.Linear(c,c); s.last=None
    def forward(s,x,ctx=None):
        B,C,H,W=x.shape; h=s.n(x).flatten(2).transpose(1,2)  # B,N,C
        src = h if ctx is None else ctx
        q=s.q(h).view(B,-1,s.h,C//s.h).transpose(1,2); k=s.k(src).view(B,-1,s.h,C//s.h).transpose(1,2); v=s.v(src).view(B,-1,s.h,C//s.h).transpose(1,2)
        a=(q@k.transpose(-1,-2)/math.sqrt(C//s.h)).softmax(-1); s.last=a.detach()
        o=(a@v).transpose(1,2).reshape(B,-1,C); return x+s.o(o).transpose(1,2).view(B,C,H,W)
def tembed(t, d=32):
    half=d//2; f=torch.exp(-math.log(10000)*torch.arange(half)/half); a=t.float()[:,None]*f[None]
    return torch.cat([torch.sin(a),torch.cos(a)],1)
class UNet(nn.Module):
    def __init__(s, C, ncls):
        super().__init__(); s.ncls=ncls
        s.te1=nn.Linear(32,128); s.te2=nn.Linear(128,128); s.cls=nn.Embedding(ncls+1,128); s.ctx=nn.Embedding(ncls+1,CTX*64)
        s.cin=nn.Conv2d(C,32,3,padding=1); s.d1=Res(32,32); s.ds1=nn.Conv2d(32,64,3,stride=2,padding=1)
        s.d2=Res(64,64); s.ds2=nn.Conv2d(64,64,3,stride=2,padding=1)
        s.m1=Res(64,64); s.sa=Attn(64); s.ca=Attn(64,ctx_dim=64); s.m2=Res(64,64)
        s.us2=nn.Conv2d(64,64,3,padding=1); s.u2=Res(128,64); s.us1=nn.Conv2d(64,64,3,padding=1); s.u1=Res(96,32)
        s.no=nn.GroupNorm(8,32); s.cout=nn.Conv2d(32,C,3,padding=1)
    def forward(s,x,t,y,hook=None):
        e=s.te2(F.silu(s.te1(tembed(t))))+s.cls(y); e=F.silu(e); ctx=s.ctx(y).view(-1,CTX,64)
        h0=s.cin(x); h1=s.d1(h0,e); h=s.ds1(h1); h2=s.d2(h,e); h=s.ds2(h2)
        h=s.m1(h,e); h=s.sa(h); h=s.ca(h,ctx); h=s.m2(h,e)
        if hook: h=hook(h)
        mid=h
        h=s.us2(F.interpolate(h,scale_factor=2,mode='nearest')); h=s.u2(torch.cat([h,h2],1),e)
        h=s.us1(F.interpolate(h,scale_factor=2,mode='nearest')); h=s.u1(torch.cat([h,h1],1),e)
        s.mid=mid; return s.cout(F.silu(s.no(h)))
class Clf(nn.Module):
    def __init__(s,C,ncls):
        super().__init__(); s.c1=nn.Conv2d(C,16,3,padding=1); s.c2=nn.Conv2d(16,32,3,stride=2,padding=1); s.c3=nn.Conv2d(32,32,3,stride=2,padding=1); s.fc=nn.Linear(32,ncls)
    def forward(s,x): h=F.relu(s.c1(x)); h=F.relu(s.c2(h)); h=F.relu(s.c3(h)); return s.fc(h.mean((2,3)))
