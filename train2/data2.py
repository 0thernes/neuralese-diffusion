# v2 REAL datasets. All cached as tensors in /workspace/data/cache_<mod>.pt.
#  image : Fashion-MNIST 28x28 (Zalando), 10 classes, official train/test split.
#  audio : Free Spoken Digit Dataset (Jakobovski, 3000 wavs, 8 kHz, 6 speakers). Log-magnitude STFT, n_fft=126 (64 bins), hop=125,
#          4000 samples (0.5 s) -> 1x64x32. Split by utterance index (0-4 test, 5-49 train) as in the FSDD README.
#  video : Moving-MNIST procedure (Srivastava et al. 2015) on real MNIST digits: one 16x16 digit bouncing in a 32x32 frame, 8 frames
#          as channels. Class = digit. Train digits from MNIST train, test digits from MNIST test.
#  pc3d  : ModelNet10 classes taken from ModelNet40-2048 (HF jxie/modelnet40-2048; ModelNet40 alphabetical label ids). 256 points by
#          farthest-point sampling, centred, unit-sphere scaled, ordered on a 16x16 grid (z-rank rows x azimuth) -> 3x16x16.
import os, numpy as np, torch, glob, wave
D='/workspace/data'
SPECS={'image':dict(C=1,H=28,W=28,ncls=10,labels=['T-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','boot'],dataset='Fashion-MNIST'),
 'audio':dict(C=1,H=64,W=32,ncls=10,labels=[str(i) for i in range(10)],dataset='FSDD spoken digits',sr=8000,n_fft=126,hop=125,n=4000),
 'video':dict(C=8,H=32,W=32,ncls=10,labels=[str(i) for i in range(10)],dataset='Moving-MNIST procedure on MNIST'),
 'pc3d':dict(C=3,H=16,W=16,ncls=10,labels=['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet'],dataset='ModelNet10 (from ModelNet40-2048)')}
MN10=[1,2,8,12,14,22,23,30,33,35]
def fashion():
    import torchvision as tv
    def f(train): d=tv.datasets.FashionMNIST(D,train=train); return (d.data.float()/127.5-1)[:,None], d.targets.clone()
    return f(True), f(False)
def stft_mag(x, sp):
    w=torch.hann_window(sp['n_fft'],periodic=True); S=torch.stft(torch.tensor(x),sp['n_fft'],sp['hop'],window=w,center=True,return_complex=True)
    return S.abs()[:, :32]
def audio():
    sp=SPECS['audio']; tr,te=[],[]
    for f in sorted(glob.glob(D+'/free-spoken-digit-dataset-master/recordings/*.wav')):
        d,spk,i=os.path.basename(f)[:-4].split('_'); w=wave.open(f); a=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.float32)/32768
        a=a/(np.abs(a).max()+1e-6)*0.9; n=sp['n']
        if len(a)>n:  # keep the loudest 0.5 s window
            e=np.convolve(a*a,np.ones(n),'valid'); s=int(e.argmax()); a=a[s:s+n]
        else: p=(n-len(a))//2; a=np.pad(a,(p,n-len(a)-p))
        L=np.log(stft_mag(a,sp).numpy()+1e-2)[:64]
        (te if int(i)<5 else tr).append((L,int(d)))
    allL=np.stack([l for l,_ in tr]); lo,hi=float(allL.min()),float(allL.max()); sp['logmin'],sp['logmax']=lo,hi
    f=lambda S: ((torch.tensor(np.stack([l for l,_ in S]))[:,None]-lo)/(hi-lo)*2-1, torch.tensor([y for _,y in S]))
    return f(tr), f(te)
def mm(digits, labels, n, rng, F=8, S=32, d=16):
    X=np.zeros((n,F,S,S),np.float32); Y=np.zeros(n,np.int64)
    for k in range(n):
        j=rng.integers(len(digits)); im=torch.nn.functional.interpolate(digits[j][None,None].float(),size=(d,d),mode='area')[0,0].numpy()/255
        p=rng.uniform(0,S-d,2); th=rng.uniform(0,2*np.pi); v=np.array([np.cos(th),np.sin(th)])*rng.uniform(1.5,3)
        for f in range(F):
            for a in range(2):
                if p[a]<0: p[a]=-p[a]; v[a]=-v[a]
                if p[a]>S-d: p[a]=2*(S-d)-p[a]; v[a]=-v[a]
            r,c=int(round(p[0])),int(round(p[1])); X[k,f,r:r+d,c:c+d]=np.maximum(X[k,f,r:r+d,c:c+d],im); p=p+v
        Y[k]=labels[j]
    return torch.tensor(X*2-1), torch.tensor(Y)
def video():
    import torchvision as tv; rng=np.random.default_rng(0)
    a=tv.datasets.MNIST(D,train=True); b=tv.datasets.MNIST(D,train=False)
    return mm(a.data,a.targets.numpy(),30000,rng), mm(b.data,b.targets.numpy(),2000,np.random.default_rng(1))
def fps(P,m,rng):
    idx=[int(rng.integers(len(P)))]; d=np.full(len(P),np.inf)
    for _ in range(m-1): d=np.minimum(d,((P-P[idx[-1]])**2).sum(1)); idx.append(int(d.argmax()))
    return P[idx]
def order_grid(P,R=16):
    P=P[np.argsort(P[:,2],kind='stable')]; G=np.zeros((3,R,R),np.float32)
    for r in range(R):
        row=P[r*R:(r+1)*R]; row=row[np.argsort(np.arctan2(row[:,1],row[:,0]))]; G[:,r,:]=row.T
    return G
def pc3d():
    import pyarrow.parquet as pq; rng=np.random.default_rng(0); out=[]
    for f in ['mn_train','mn_test']:
        t=pq.read_table(f'{D}/{f}.parquet').to_pydict(); X=[];Y=[]
        for P,l in zip(t['inputs'],t['label']):
            if l not in MN10: continue
            P=np.array(P,np.float32); P=P-P.mean(0); P=P/np.linalg.norm(P,axis=1).max(); P=P[:,[0,2,1]] if False else P
            X.append(order_grid(fps(P,256,rng))); Y.append(MN10.index(l))
        out.append((torch.tensor(np.stack(X)),torch.tensor(Y)))
    return out[0], out[1]
def load(mod):
    c=f'{D}/cache_{mod}.pt'
    if os.path.exists(c): r=torch.load(c); SPECS[mod].update(r.get('spec',{})); return r['tr'],r['te']
    tr,te={'image':fashion,'audio':audio,'video':video,'pc3d':pc3d}[mod](); torch.save(dict(tr=tr,te=te,spec={k:v for k,v in SPECS[mod].items() if k in('logmin','logmax')}),c); return tr,te
if __name__=='__main__':
    import sys
    for m in sys.argv[1:]: (X,Y),(Xt,Yt)=load(m); print(m,X.shape,Xt.shape,float(X.min()),float(X.max()),np.bincount(Y.numpy()).tolist())
