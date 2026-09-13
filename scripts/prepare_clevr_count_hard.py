#!/usr/bin/env python3
"""离线从本地缓存的 CLEVR 数据里造「高计数 + 结构化输出」数据集（demo 用任务 B）。

* 数据来源：`.cache/modelscope/datasets/AI-ModelScope___clevr_cogen_a_train/.../*.arrow`
  （用 datasets.Dataset.from_file 直接读，不需要联网）
* 只保留物体数 >= --min-count 的"难题"（base 模型在这类图上错得最多）
* 输出格式改成严格的 `<think>…</think><answer>{"count": N}</answer>`

用法：
    python3 scripts/prepare_clevr_count_hard.py --min-count 8 --train 3000 --val 300
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
from typing import List

from datasets import Dataset

DEFAULT_CACHE = ('.cache/modelscope/datasets/AI-ModelScope___clevr_cogen_a_train/'
                 'default-d1b9c1e6e17ade03/0.0.0/master')
PROMPT = ('Look at the image and count how many distinct objects there are. '
          'First think step by step inside <think> </think>, then output ONLY a JSON object '
          'inside <answer> </answer>, e.g. <answer>{"count": 3}</answer>.')


def gt_count(solution: str) -> int:
    digits = re.findall(r'\d+', str(solution))
    return int(digits[0]) if digits else -1


def make_answer(n: int) -> str:
    listing = ' '.join(f'{i + 1}.' for i in range(n))
    return (f'<think>I count the distinct objects one by one: {listing} '
            f'That gives {n} objects in total.</think><answer>{{"count": {n}}}</answer>')


def pick_indices(cache_dir: str, min_count: int, need: int, max_shards: int = 0):
    """先用便宜的 solution 列筛选，拿到 (分片路径, 行号) 列表。

    关键点：**不要**在这里解码图片（第一版对每张图做 md5 去重，等于把上万张 PNG 全解一遍，
    于是看起来"卡住"）。图片只在真正要落盘时按行号取。
    """
    picked, dist = [], {}
    files = sorted(glob.glob(os.path.join(cache_dir, '*.arrow')))
    if max_shards:
        files = files[:max_shards]
    for f in files:
        ds = Dataset.from_file(f)
        if 'image' not in ds.column_names:
            print(f'  [skip] {os.path.basename(f)} 列={ds.column_names}')
            continue
        sols = ds['solution']            # 只读字符串列，不解码图片
        for i, s in enumerate(sols):
            n = gt_count(s)
            dist[n] = dist.get(n, 0) + 1
            if n >= min_count:
                picked.append((f, i, n))
                if len(picked) >= need:
                    return picked, dist
    return picked, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default=DEFAULT_CACHE)
    ap.add_argument('--out-dir', default='data/clevr_count_hard')
    ap.add_argument('--min-count', type=int, default=8)
    ap.add_argument('--train', type=int, default=3000)
    ap.add_argument('--val', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--stats-only', action='store_true')
    args = ap.parse_args()

    need = args.train + args.val
    picked, dist = pick_indices(args.cache_dir, args.min_count, need,
                                max_shards=2 if args.stats_only else 0)
    print(f'扫描到的数量分布: {dict(sorted(dist.items()))}')
    print(f'count>={args.min_count} 的候选: {len(picked)} 条（需要 {need}）')
    if args.stats_only:
        return
    if len(picked) < need:
        print(f'[warn] 只凑到 {len(picked)} 条，可降低 --min-count')

    rows = []
    cache_of = {}
    for f, i, n in picked:
        if f not in cache_of:
            cache_of[f] = Dataset.from_file(f)
        rows.append({'img': cache_of[f][i]['image'], 'n': n})

    random.Random(args.seed).shuffle(rows)
    splits = {'train': rows[:args.train], 'val': rows[args.train:args.train + args.val]}
    for name, rs in splits.items():
        img_dir = os.path.join(args.out_dir, 'images', name)
        os.makedirs(img_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f'{name}.jsonl')
        with open(out, 'w', encoding='utf-8') as f:
            for i, r in enumerate(rs):
                p = os.path.abspath(os.path.join(img_dir, f'{name}_{i:05d}_{r["n"]}.png'))
                r['img'].convert('RGB').save(p)
                sol = json.dumps({'count': r['n']}, ensure_ascii=False)
                f.write(json.dumps({
                    'messages': [{'role': 'user', 'content': PROMPT},
                                 {'role': 'assistant', 'content': make_answer(r['n'])}],
                    'images': [p],
                    'solution': sol,
                    'labels': sol,
                    'meta': {'count': r['n']},
                }, ensure_ascii=False) + '\n')
        print(f'{name}: {len(rs)} 条 -> {out}  (图 {img_dir})')


if __name__ == '__main__':
    main()
