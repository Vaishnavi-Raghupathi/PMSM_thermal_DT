import json
import os
import glob

all_params = {}

for path in glob.glob("results/params/params_*.json"):

    pid = os.path.basename(path) \
            .replace("params_","") \
            .replace(".json","")

    with open(path) as f:
        all_params[pid] = json.load(f)

with open("data/calibrated_params.json","w") as f:
    json.dump(all_params,f,indent=2)

print(f"Saved {len(all_params)} profiles → data/calibrated_params.json")