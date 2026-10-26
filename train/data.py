# Data generators for the four modalities. All tiny, all reproducible from a seed.
import numpy as np, torch, math
SPECS = {
  'image': dict(C=1, H=16, W=16, ncls=10, labels=[str(i) for i in range(10)], desc='MNIST digits, 16x16 grayscale (bilinear downsample)'),
  'music': dict(C=1, H=16, W=32, ncls=4, labels=['arp-up','arp-down','chord','bass-alt'], desc='procedural piano-roll: 16 pitches (C-major scale, 2 octaves) x 32 steps'),
  'video': dict(C=4, H=16, W=16, ncls=4, labels=['right','left','down','up'], desc='procedural 4-frame clips: 4x4 blob moving 2 px/frame'),
  'shape3d': dict(C=3, H=8, W=8, ncls=4, labels=['sphere','cube','torus','cone'], desc='procedural 64-point clouds, xyz in channels, points ordered into an 8x8 grid by (z-rank row, azimuth column)'),
}
def mnist(n_max=None):
    import torchvision, torch.nn.functional as F
    ds = torchvision.datasets.MNIST('/workspace/mnist', train=True, download=True)
    x = ds.data.float().div(255).unsqueeze(1)
    x = F.interpolate(x, size=(16,16), mode='bilinear', antialias=True, align_corners=False)
    return (x*2-1), ds.targets.clone()
def music(n, rng):
    X = np.full((n,1,16,32), -1, np.float32); Y = rng.integers(0,4,n)
    for i in range(n):
        y = Y[i]; root = rng.integers(0,7); L = rng.choice([2,4])
        for s in range(0,32,L):
            k = s//L
            if y==0: p = root + [0,2,4,7][k%4]
            elif y==1: p = root + [7,4,2,0][k%4]
            elif y==2: ps = [root, root+2, root+4]
            else: p = root + (0 if k%2==0 else 4)
            for q in (ps if y==2 else [p]):
                q = int(q)
                if 0<=q<16: X[i,0,q,s:s+L-1 if L>1 else s+1] = 1
    return torch.tensor(X), torch.tensor(Y)
def video(n, rng):
    X = np.full((n,4,16,16), -1, np.float32); Y = rng.integers(0,4,n)
    d = {0:(0,2),1:(0,-2),2:(2,0),3:(-2,0)}
    for i in range(n):
        dy,dx = d[int(Y[i])]
        y0 = rng.integers(0 if dy>=0 else 6, 7 if dy<=0 else 1) if dy else rng.integers(0,13)
        x0 = rng.integers(0 if dx>=0 else 6, 7 if dx<=0 else 1) if dx else rng.integers(0,13)
        if dy>0: y0 = rng.integers(0,7)
        if dy<0: y0 = rng.integers(6,13)
        if dx>0: x0 = rng.integers(0,7)
        if dx<0: x0 = rng.integers(6,13)
        for f in range(4):
            yy, xx = y0+dy*f, x0+dx*f
            X[i,f,yy:yy+4,xx:xx+4] = 1
    return torch.tensor(X), torch.tensor(Y)
def shape_points(y, rng, m=64):
    if y==0:
        v = rng.normal(size=(m,3)); v /= np.linalg.norm(v,axis=1,keepdims=True); P = v*0.8
    elif y==1:
        P = rng.uniform(-1,1,(m,3)); a = rng.integers(0,3,m); P[np.arange(m),a] = rng.choice([-1,1],m); P*=0.6
    elif y==2:
        u = rng.uniform(0,2*np.pi,m); w = rng.uniform(0,2*np.pi,m); R,r=0.6,0.25
        P = np.stack([(R+r*np.cos(w))*np.cos(u),(R+r*np.cos(w))*np.sin(u),r*np.sin(w)],1)
    else:
        h = rng.uniform(0,1,m); u = rng.uniform(0,2*np.pi,m); rad = 0.7*(1-h)
        P = np.stack([rad*np.cos(u), rad*np.sin(u), h*1.4-0.7],1)
    th = rng.uniform(0,2*np.pi); c,s = np.cos(th),np.sin(th)
    P = P @ np.array([[c,-s,0],[s,c,0],[0,0,1]]).T  # random rotation about z
    P = P*rng.uniform(0.85,1.15) + rng.normal(0,0.01,(m,3))
    return P
def order_grid(P):
    idx = np.argsort(P[:,2], kind='stable'); P = P[idx]; G = np.zeros((3,8,8),np.float32)
    for r in range(8):
        row = P[r*8:(r+1)*8]; row = row[np.argsort(np.arctan2(row[:,1],row[:,0]))]; G[:,r,:] = row.T
    return G
def shape3d(n, rng):
    Y = rng.integers(0,4,n); X = np.stack([order_grid(shape_points(int(y),rng)) for y in Y])
    return torch.tensor(X), torch.tensor(Y)
def load(mod, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    if mod=='image': return mnist()
    return {'music':music,'video':video,'shape3d':shape3d}[mod](n, rng)
