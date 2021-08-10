import json
import os

f = open('test_config.json')
test_conf = json.load(f)
test_conf['model_folder'] = 1
snaps = [1000, 15000, 20000]
for snp in snaps:
    test_conf['model_snapshot'] = 15000
    with open('test_config.json', 'w') as outfile:
        json.dump(test_conf, outfile)
    os.system('python3 validation.py with test_config.json')
