import numpy as np
import os
import os.path as osp
import json

with open('thumos14_v2.json', encoding='utf-8') as f:
    meta = json.load(f)
feature_path = 'feature'
label_path = "label"
LABEL_NUM = 20

if not osp.exists(label_path):
    os.mkdir(label_path)


np.set_printoptions(threshold=np.inf)

for name in meta['database']:
    felen = len(np.load(osp.join(feature_path, f'{name}.npy')))
    duration = meta['database'][name]['duration']
    fps = felen/duration
    label = np.zeros((felen, LABEL_NUM+1), dtype=np.int64)
    label[:, 0] = 1
    for anno in meta['database'][name]['annotations']:
        st, ed = anno['segment']
        st, ed = round(st*fps), round(ed*fps)
        #set class label
        label[st:ed, anno['labelIndex'] + 1] = 1
        #reset background class
        label[st:ed, 0] = 0
    np.save(osp.join(label_path, f'{name}.npy'), label)

