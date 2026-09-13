#!/usr/bin/env python3
"""把 count 数据集的 val.jsonl 转成"只带问题"的推理输入（供 swift infer 用）。"""
import json
import os
import sys

src = sys.argv[1] if len(sys.argv) > 1 else 'data/clevr_count_hard/val.jsonl'
dst = sys.argv[2] if len(sys.argv) > 2 else 'results/clevr_count_hard/samples/infer_input.jsonl'
os.makedirs(os.path.dirname(dst), exist_ok=True)
n = 0
with open(src, encoding='utf-8') as fi, open(dst, 'w', encoding='utf-8') as fo:
    for line in fi:
        d = json.loads(line)
        fo.write(json.dumps({
            'messages': [m for m in d['messages'] if m['role'] == 'user'],
            'images': d['images'],
            'solution': d['solution'],
            'labels': d['labels'],
            'meta': d['meta'],
        }, ensure_ascii=False) + '\n')
        n += 1
print(f'{n} 条 -> {dst}')
