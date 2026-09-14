#!/usr/bin/env python3
"""第 2 轮优化：把负例的 reason 缩短成固定短句（内容仍是固定的，不随机）。

动机：v3 里负例那句固定文案有 ~40 token，模型背下来就能白拿 2/3 样本的 token 准确率。
缩短后模板几乎不占 token，"答题"的收益几乎只来自 <answer>yes/no</answer> 这一处。

正例保持不变：<answer>yes</answer><reason>三分命中</reason>
负例改为：    <answer>no</answer><reason>无</reason>

用法：python3 scripts/22_rewrite_short_negative.py --src data/highlight_cls_v3 --dst data/highlight_cls_v4
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/highlight_cls_v3')
    ap.add_argument('--dst', default='data/highlight_cls_v4')
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    n_neg = 0
    for f in sorted(glob.glob(os.path.join(args.src, '*.jsonl'))):
        out = os.path.join(args.dst, os.path.basename(f))
        with open(f, encoding='utf-8') as fi, open(out, 'w', encoding='utf-8') as fo:
            for line in fi:
                d = json.loads(line)
                sol = json.loads(d['solution'])
                if not sol['is_highlight']:
                    d['messages'][-1]['content'] = '<answer>no</answer><reason>无</reason>'
                    n_neg += 1
                fo.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'负例目标已缩短为 <answer>no</answer><reason>无</reason>，共 {n_neg} 条 -> {args.dst}')


if __name__ == '__main__':
    main()
