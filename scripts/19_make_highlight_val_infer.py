#!/usr/bin/env python3
"""把高光分类数据集的验证节（默认 3-3v3_*.jsonl）转成只带问题的推理输入。"""
import json
import os
import sys
import glob

src_glob = sys.argv[1] if len(sys.argv) > 1 else 'data/highlight_cls_v2/3-3v3_*.jsonl'
dst = sys.argv[2] if len(sys.argv) > 2 else 'results/highlight_cls_v2/samples/infer_input.jsonl'
os.makedirs(os.path.dirname(dst), exist_ok=True)
n = 0
with open(dst, 'w', encoding='utf-8') as fo:
    for f in sorted(glob.glob(src_glob)):
        for line in open(f, encoding='utf-8'):
            d = json.loads(line)
            sol = json.loads(d['solution'])
            fo.write(json.dumps({
                'messages': [m for m in d['messages'] if m['role'] == 'user'],
                'images': d['images'],
                'solution': d['solution'],
                'labels': d['solution'],
                'meta': {'file': os.path.basename(f), 'clip': os.path.basename(os.path.dirname(d['images'][0])),
                         'is_highlight': sol['is_highlight'], 'description': sol.get('description', '')},
            }, ensure_ascii=False) + '\n')
            n += 1
print(f'{n} 条 -> {dst}')
