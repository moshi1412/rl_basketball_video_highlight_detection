#!/usr/bin/env python3
"""GEOQA 任务的抽样 / 评测 / demo 视频生成。

和 07_make_clevr_demo.py 的区别：GEOQA 的答案不是整数而是数学表达式（"145°"、"25\\pi"），
所以判定用训练时同一套 reward（swift 的 accuracy ORM / math_verify），保证视频里的
✅❌ 和训练指标口径一致。

用法:
  python3 scripts/09_make_geoqa_demo.py prepare --num 100 --out results/geoqa_grpo_3b
  # 用 base / GRPO checkpoint 各跑一次 swift infer（见 results/geoqa_grpo_3b/README.md）
  python3 scripts/09_make_geoqa_demo.py render --out results/geoqa_grpo_3b \
      --models base=xx.jsonl GRPO=yy.jsonl --video-num 20
"""
import argparse
import json
import os
import random
import re
import subprocess
from typing import Dict, List

DATASET = 'AI-ModelScope/GEOQA_R1V_Train_8K'
SYSTEM_PROMPT = ('You are a helpful assistant. You first think about the reasoning process in the mind '
                 'and then provide the user with the answer. The reasoning process is enclosed within '
                 '<think> </think> and the final answer is enclosed within <answer> </answer> tags. '
                 'Keep the reasoning concise (a few steps) and put only the final result inside <answer>.')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def _acc_orm():
    """直接用训练时的 accuracy ORM 判定对错（math_verify）。"""
    from swift.rewards import orms
    return orms['accuracy']()


def prepare(args):
    from modelscope.msdatasets import MsDataset

    os.makedirs(os.path.join(args.out, 'samples', 'images'), exist_ok=True)
    ds = MsDataset.load(DATASET, split='train', use_streaming=False)
    rng = random.Random(args.seed)
    if args.index_range:
        lo, hi = (int(x) for x in args.index_range.split(':'))
        pool = list(range(lo, min(hi, len(ds))))
        idxs = pool[:args.num]
    else:
        idxs = sorted(rng.sample(range(len(ds)), args.num))

    rows = []
    for i, idx in enumerate(idxs):
        d = ds[idx]
        name = f'geoqa_{i:03d}_{idx}.png'
        img_path = os.path.abspath(os.path.join(args.out, 'samples', 'images', name))
        d['image'].convert('RGB').save(img_path)
        rows.append({
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': d['problem'].strip()}],
            'images': [img_path],
            'solution': d['solution'],
            'labels': d['solution'],   # swift infer 用 labels 作为 GT 字段
            'meta': {'geoqa_index': idx, 'gt': d['solution']},
        })
    out_jsonl = os.path.join(args.out, 'samples', 'infer_input.jsonl')
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'抽样 {len(rows)} 条 -> {out_jsonl}')


def _read_results(path: str) -> Dict[str, dict]:
    out = {}
    for line in open(path, encoding='utf-8'):
        d = json.loads(line)
        imgs = d.get('images') or []
        key = os.path.basename(str(imgs[0].get('path', ''))) if imgs and isinstance(imgs[0], dict) else ''
        out[key] = d
    return out


def _wrap(draw, text, font, x, y, width, line_h, fill=(20, 20, 20), max_lines=12):
    lines, cur = [], ''
    for ch in text:
        if ch == '\n':
            lines.append(cur)
            cur = ''
            continue
        cur += ch
        if draw.textlength(cur, font=font) > width:
            lines.append(cur)
            cur = ''
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    for i, ln in enumerate(lines):
        draw.text((x, y + i * line_h), ln, font=font, fill=fill)
    return y + len(lines) * line_h


def evaluate(args, models, orm, rows):
    """不生成视频，只算每个模型在抽样集上的准确率。"""
    summary = []
    for r in rows:
        key = os.path.basename(r['images'][0])
        item = {'sample': key, 'solution': r['solution'], 'models': {}}
        for label, res in models:
            resp = res.get(key, {}).get('response', '') or ''
            item['models'][label] = {
                'response': resp,
                'correct': bool(orm(completions=[resp], solution=[r['solution']])[0] >= 0.5),
            }
        summary.append(item)
    return summary


def render(args):
    from PIL import Image, ImageDraw, ImageFont

    rows = [json.loads(l) for l in open(os.path.join(args.out, 'samples', 'infer_input.jsonl'),
                                        encoding='utf-8')]
    models = [(m.split('=', 1)[0], _read_results(m.split('=', 1)[1])) for m in args.models]
    orm = _acc_orm()

    summary = evaluate(args, models, orm, rows)
    os.makedirs(os.path.join(args.out, 'samples'), exist_ok=True)
    with open(os.path.join(args.out, 'samples', 'vlm_answers.jsonl'), 'w', encoding='utf-8') as f:
        for s in summary:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    with open(os.path.join(args.out, 'samples', 'vlm_answers_readable.md'), 'w', encoding='utf-8') as f:
        for s in summary:
            f.write(f"### {s['sample']}  GT={s['solution'].strip()}\n\n")
            for label, _ in models:
                m = s['models'][label]
                f.write(f"- {label} {'✅' if m['correct'] else '❌'}\n")
                f.write(f"  > {m['response'].replace(chr(10), ' ')[:500]}\n\n")

    accs = {l: sum(s['models'][l]['correct'] for s in summary) / len(summary) for l, _ in models}
    print(f"评测集 n={len(summary)}  accuracy: " + ', '.join(f'{k} {v:.1%}' for k, v in accs.items()))
    with open(os.path.join(args.out, 'metrics', 'eval_accuracy.json'), 'w', encoding='utf-8') as f:
        json.dump({'n': len(summary), 'accuracy': accs}, f, ensure_ascii=False, indent=2)

    # ---- 渲染 demo 视频（取前 video_num 条）----
    f_dir = os.path.join(args.out, 'demo', 'frames')
    os.makedirs(f_dir, exist_ok=True)
    f_title = ImageFont.truetype(FONT, 30)
    f_txt = ImageFont.truetype(FONT, 19)
    f_small = ImageFont.truetype(FONT, 17)
    W, H = 1280, 720
    frame_idx = 0
    for i, (r, s) in enumerate(zip(rows, summary)):
        if i >= args.video_num:
            break
        img = Image.open(r['images'][0]).convert('RGB')
        img.thumbnail((600, 420))
        canvas = Image.new('RGB', (W, H), (252, 252, 252))
        canvas.paste(img, (30, 150 + (420 - img.height) // 2))
        dr = ImageDraw.Draw(canvas)
        dr.text((30, 24), f'GEOQA (geometry QA) — multimodal GRPO   sample {i + 1}/{min(args.video_num, len(rows))}',
                font=f_title, fill=(10, 10, 10))
        dr.text((30, 66), f"Q: {r['messages'][-1]['content'][:95]}", font=f_txt, fill=(80, 80, 80))
        dr.text((30, 96), f"GT: {r['solution'].strip()}", font=f_txt, fill=(20, 100, 20))

        x, y = 660, 150
        dr.rectangle([x - 12, y - 14, W - 24, H - 30], outline=(215, 215, 215), width=2)
        for label, _ in models:
            m = s['models'][label]
            mark = 'O' if m['correct'] else 'X'
            dr.text((x, y), f'{label}  [{mark}]', font=f_txt,
                    fill=(20, 130, 20) if m['correct'] else (190, 40, 40))
            y += 26
            y = _wrap(dr, m['response'].replace('\n', ' '), f_small, x, y, 580, 22,
                      fill=(105, 105, 105), max_lines=6 if len(models) > 2 else 9)
            y += 18
        dr.text((30, H - 44), f"Accuracy on n={len(summary)}:  " +
                '   '.join(f'{k} {v:.0%}' for k, v in accs.items()), font=f_txt, fill=(60, 60, 60))

        for _ in range(int(args.seconds * 25)):
            canvas.save(os.path.join(f_dir, f'{frame_idx:04d}.jpg'), quality=92)
            frame_idx += 1

    mp4 = os.path.join(args.out, 'demo', 'geoqa_grpo_demo.mp4')
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-framerate', '25', '-i',
                    os.path.join(f_dir, '%04d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-r', '25', '-vf', 'scale=1280:720', mp4, '-y'], check=True)
    print('视频:', mp4)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--num', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--index-range', default=None)
    p.add_argument('--out', default='results/geoqa_grpo_3b')
    p = sub.add_parser('render')
    p.add_argument('--out', default='results/geoqa_grpo_3b')
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--video-num', type=int, default=20)
    p.add_argument('--seconds', type=float, default=3.0)
    args = ap.parse_args()
    (prepare if args.cmd == 'prepare' else render)(args)


if __name__ == '__main__':
    main()
