#!/usr/bin/env python3
"""把高光分类数据改成「答案前置」格式（不改比例、不改负例固定句、保留 reason）。

旧：片段包含“三分命中”这类关键攻防回合，值得剪进集锦。<answer>yes</answer>
新：<answer>yes</answer><reason>三分命中</reason>

负例（保持那句固定文案，只是挪到 reason 里，内容不变、不随机）：
旧：画面只是常规运球/站位，没有得分或关键防守回合，不值得剪进集锦。<answer>no</answer>
新：<answer>no</answer><reason>画面只是常规运球/站位，没有得分或关键防守回合，不值得剪进集锦。</reason>

图片路径不动（仍指向 data/highlight_cls_v2/frames 下的帧），只重写文本。

用法：python3 scripts/21_rewrite_answer_first.py --src data/highlight_cls_v2 --dst data/highlight_cls_v3
"""
import argparse
import glob
import json
import os

PROMPT = ('这是篮球 3v3 比赛中的一段 {dur} 秒片段（比赛时间 {start}）。'
          '请先给出结论：用 <answer>yes</answer> 或 <answer>no</answer> 表示这段是否值得剪进高光集锦；'
          '然后在 <reason> </reason> 里用一句话说明理由（看了画面里的什么）。')


def sec2hms(s: float) -> str:
    s = int(round(s))
    return f'{s // 60:02d}:{s % 60:02d}'


def make_target(pos: bool, desc: str, start: float, end: float) -> str:
    if pos:
        return f'<answer>yes</answer><reason>{desc}</reason>'
    return ('<answer>no</answer><reason>画面只是常规运球/站位，没有得分或关键防守回合，'
            '不值得剪进集锦。</reason>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/highlight_cls_v2')
    ap.add_argument('--dst', default='data/highlight_cls_v3')
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    n_pos = n_neg = 0
    for f in sorted(glob.glob(os.path.join(args.src, '*.jsonl'))):
        out = os.path.join(args.dst, os.path.basename(f))
        with open(f, encoding='utf-8') as fi, open(out, 'w', encoding='utf-8') as fo:
            for line in fi:
                d = json.loads(line)
                sol = json.loads(d['solution'])
                pos = bool(sol['is_highlight'])
                start, end = sol['interval']
                dur = end - start
                user = PROMPT.format(dur=f'{dur:.1f}', start=sec2hms(start))
                target = make_target(pos, sol.get('description', ''), start, end)
                fo.write(json.dumps({
                    'messages': [{'role': 'user', 'content': user},
                                 {'role': 'assistant', 'content': target}],
                    'images': d['images'],
                    'solution': d['solution'],
                    'labels': d['solution'],
                    'meta': d['meta'],
                }, ensure_ascii=False) + '\n')
                n_pos += pos
                n_neg += (not pos)
        print(f'{os.path.basename(f)} -> {out}')
    print(f'\n合计 正例 {n_pos} / 负例 {n_neg}（比例不变），负例仍为 1 条固定句')


if __name__ == '__main__':
    main()
