# Post-training export: held-out batches (first 32 test examples, stratified by class order of the test split) for the LAB tab / tests,
# and the audio log-spectrogram normalisation (needed to invert to a waveform). Does not touch weights or their sha256.
import sys, json; sys.path.insert(0,'train2'); import data2, torch
for mod in ['image','audio','video','pc3d']:
    tr,te=data2.load(mod); X,Y=te; g=torch.Generator().manual_seed(0); idx=torch.randperm(len(Y),generator=g)[:32]
    json.dump(dict(mod=mod,source='test split, randperm(seed 0)[:32]',X=[[round(float(v),4) for v in X[i].flatten()] for i in idx],Y=[int(Y[i]) for i in idx]),open(f'site/results2/heldout_{mod}.json','w'))
    if mod=='audio':
        p='site/models2/audio.json'
        try: m=json.load(open(p)); m['spec']={k:data2.SPECS['audio'][k] for k in ['sr','n_fft','hop','n','logmin','logmax']}; json.dump(m,open(p,'w'))
        except FileNotFoundError: print('audio.json not yet written')
    print(mod,len(idx))
