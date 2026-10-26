import numpy as np, wave, json, torch
def rd(f):
    w=wave.open(f); a=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.float32).reshape(-1,2).mean(1)/32768; return torch.tensor(a)
def mel(sr=44100,n_fft=2048,n=64):
    hz=lambda m:700*(10**(m/2595)-1); mm=np.linspace(0,2595*np.log10(1+sr/2/700),n+2); f=np.floor((n_fft+1)*hz(mm)/sr).astype(int); M=np.zeros((n,n_fft//2+1))
    for i in range(n):
        for k in range(f[i],f[i+1]): M[i,k]=(k-f[i])/max(1,f[i+1]-f[i])
        for k in range(f[i+1],f[i+2]): M[i,k]=(f[i+2]-k)/max(1,f[i+2]-f[i+1])
    return torch.tensor(M,dtype=torch.float32)
M=mel()
def lm(a): S=torch.stft(a,2048,512,window=torch.hann_window(2048),return_complex=True).abs()**2; return torch.log(M@S+1e-6).mean(1)  # time-averaged log-mel
A,B,A1=[lm(rd(f'out/sa3_{n}.wav')) for n in ['upbeat_0','slow_0','upbeat_1']]
d=lambda x,y: float(((x-y)**2).mean().sqrt())
r=dict(prompt_distance=d(A,B),seed_distance=d(A,A1),feature='RMS diff of time-averaged 64-band log-mel (44.1 kHz, n_fft 2048)')
R=json.load(open('/workspace/neuralese-diffusion/site/results2/real_sa3.json')); R['R8']=dict(**r,max_total_ms=max(x['total_ms'] for x in R['runs']),min_rms=min(x['rms'] for x in R['runs']))
R['R8']['pass']=R['R8']['max_total_ms']<600000 and R['R8']['min_rms']>0.01 and r['prompt_distance']>r['seed_distance']
json.dump(R,open('/workspace/neuralese-diffusion/site/results2/real_sa3.json','w'),indent=1); print(R['R8'])
