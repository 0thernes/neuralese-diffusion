# Split each model's base64 into <=1.9 MB parts (deploy tooling has a ~2 MB request limit). Loader concatenates.
import json, os, glob
for m in ['image','music','video','shape3d']:
    s=open(f'site/models/{m}.b64').read().strip(); n=(len(s)+1899999)//1900000
    for f in glob.glob(f'site/models/{m}.*.b64'): os.remove(f)
    for i in range(n): open(f'site/models/{m}.{i}.b64','w').write(s[i*1900000:(i+1)*1900000])
    j=json.load(open(f'site/models/{m}.json')); j['b64parts']=n; json.dump(j,open(f'site/models/{m}.json','w')); os.remove(f'site/models/{m}.b64'); print(m,n)
