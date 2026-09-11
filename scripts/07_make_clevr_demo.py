#!/usr/bin/env python3
"""为 ClevrCount GRPO 任务生成 demo：抽样图片 + 两个模型回答 + 对比视频。

三步：
  1) prepare  : 从 CLEVR 数据集抽样、存图、生成 swift infer 的输入 jsonl
  2) （外部）  : 用 base 模型 / GRPO checkpoint 各跑一次 `swift infer`（命令见 results/README.md）
  3) render   : 把两次回答对齐渲染成帧并合成 mp4，同时输出对比 jsonl

示例：
  python3 scripts/07_make_clevr_demo.py prepare --num 12 --out results/clevr_grpo_3b
  python3 scripts/07_make_clevr_demo.py render  --out results/clevr_grpo_3b \
      --base base.jsonl --trained trained.jsonl
"""
import argparse
import json
import os
import random
import re
import subprocess
from typing import Dict, List

PROMPT_SUFFIX = (' Output the thinking process in <think> </think> and final answer (number) in '
                 '<answer> </answer> tags.')
ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
FONT_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_LATIN = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def _num(text):
    m = ANSWER_RE.search(text or '')
    if not m:
        return None
    digits = re.findall(r'-?\d+', m.group(1))
    return int(digits[0]) if digits else None


def prepare(args):
    from modelscope.msdatasets import MsDataset

    os.makedirs(os.path.join(args.out, 'samples', 'images'), exist_ok=True)
    ds = MsDataset.load('AI-ModelScope/clevr_cogen_a_train', subset_name='default',
                        split='train', use_streaming=False)
    rng = random.Random(args.seed)
    if args.index_range:
        lo, hi = (int(x) for x in args.index_range.split(':'))
        pool = list(range(lo, min(hi, len(ds))))
        idxs = pool[:args.num] if args.num <= len(pool) else pool
    else:
        idxs = rng.sample(range(len(ds)), args.num)
    idxs.sort()

    rows = []
    for i, idx in enumerate(idxs):
        d = ds[idx]
        name = f'clevr_{i:02d}_{os.path.basename(str(idx))}.png'
        img_path = os.path.abspath(os.path.join(args.out, 'samples', 'images', name))
        d['image'].convert('RGB').save(img_path)
        gt = _num(d['solution'])
        rows.append({
            'messages': [{'role': 'user', 'content': d['problem'].strip() + PROMPT_SUFFIX}],
            'images': [img_path],
            'solution': d['solution'],
            'meta': {'clevr_index': idx, 'gt': gt},
        })

    out_jsonl = os.path.join(args.out, 'samples', 'infer_input.jsonl')
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'抽样 {len(rows)} 条 -> {out_jsonl}')
    print('图片目录:', os.path.join(args.out, 'samples', 'images'))
    print('GT:', [r['meta']['gt'] for r in rows])


def _read_results(path: str) -> Dict[str, dict]:
    """读取 swift infer 的输出，按图片文件名索引。"""
    out = {}
    for line in open(path, encoding='utf-8'):
        d = json.loads(line)
        imgs = d.get('images') or []
        key = ''
        if imgs and isinstance(imgs[0], dict):
            key = os.path.basename(str(imgs[0].get('path', '')))
        out[key] = d
    return out


def _wrap(draw, text, font, x, y, width, line_h, fill=(20, 20, 20), max_lines=14):
    """按像素宽度粗暴折行（中文/英文混排够用）。"""
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


def render(args):
    from PIL import Image, ImageDraw, ImageFont

    rows = [json.loads(l) for l in open(os.path.join(args.out, 'samples', 'infer_input.jsonl'),
                                        encoding='utf-8')]
    models = [(m.split('=', 1)[0], _read_results(m.split('=', 1)[1])) for m in args.models]

    f_dir = os.path.join(args.out, 'demo', 'frames')
    os.makedirs(f_dir, exist_ok=True)
    f_title = ImageFont.truetype(FONT_LATIN, 34)
    f_txt = ImageFont.truetype(FONT_LATIN, 20)
    f_small = ImageFont.truetype(FONT_LATIN, 18)

    W, H = 1280, 720
    summary, frame_idx = [], 0
    for i, r in enumerate(rows):
        key = os.path.basename(r['images'][0])
        gt = r['meta']['gt']
        row = {'sample': key, 'image': r['images'][0], 'question': r['messages'][0]['content'],
               'ground_truth': gt, 'models': {}}
        for label, res in models:
            d = res.get(key, {})
            num = _num(d.get('response', ''))
            row['models'][label] = {'answer': num, 'correct': num == gt,
                                    'response': d.get('response', '')}
        summary.append(row)

        img = Image.open(r['images'][0]).convert('RGB')
        img.thumbnail((560, 560))
        canvas = Image.new('RGB', (W, H), (250, 250, 250))
        canvas.paste(img, (30, 110 + (560 - img.height) // 2))
        dr = ImageDraw.Draw(canvas)
        dr.text((30, 26), f'ClevrCount (multimodal GRPO)  sample {i + 1}/{len(rows)}',
                font=f_title, fill=(10, 10, 10))
        dr.text((30, 72), 'Q: How many items are there in the image?', font=f_txt, fill=(90, 90, 90))

        x = 620
        y = 110
        dr.rectangle([x - 10, y - 12, W - 24, H - 30], outline=(210, 210, 210), width=2)
        dr.text((x, y), f'Ground truth: {gt}', font=f_txt, fill=(20, 90, 20))
        y += 40
        max_lines = 4 if len(models) > 2 else 7
        for label, _ in models:
            m = row['models'][label]
            mark = 'O' if m['correct'] else 'X'
            dr.text((x, y), f'{label}: {m["answer"]}  [{mark}]', font=f_txt,
                    fill=(20, 130, 20) if m['correct'] else (190, 40, 40))
            y += 28
            y = _wrap(dr, (m['response'] or '').replace('\n', ' '), f_small, x, y, 620, 23,
                      fill=(110, 110, 110), max_lines=max_lines)
            y += 16
        accs = '   '.join(f'{l} {sum(s["models"][l]["correct"] for s in summary) / len(summary):.0%}'
                          for l, _ in models)
        dr.text((30, H - 46), f'Running accuracy (n={len(summary)}):   {accs}',
                font=f_txt, fill=(60, 60, 60))

        cur = os.path.join(f_dir, f'{frame_idx:04d}.jpg')
        canvas.save(cur, quality=92)
        # 每张图停留 args.seconds 秒（25fps）
        for _ in range(int(args.seconds * 25) - 1):
            frame_idx += 1
            canvas.save(os.path.join(f_dir, f'{frame_idx:04d}.jpg'), quality=92)
        frame_idx += 1

    os.makedirs(os.path.join(args.out, 'samples'), exist_ok=True)
    with open(os.path.join(args.out, 'samples', 'vlm_answers.jsonl'), 'w', encoding='utf-8') as f:
        for s in summary:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    with open(os.path.join(args.out, 'samples', 'vlm_answers_readable.md'), 'w', encoding='utf-8') as f:
        for s in summary:
            f.write(f"### {s['sample']}  (GT={s['ground_truth']})\n\n")
            for label, _ in models:
                m = s['models'][label]
                f.write(f"- {label}: **{m['answer']}** {'✅' if m['correct'] else '❌'}\n")
                f.write(f"  > {(m['response'] or '').replace(chr(10), ' ')[:400]}\n\n")

    mp4 = os.path.join(args.out, 'demo', 'clevr_grpo_demo.mp4')
    subprocess.run([
        'ffmpeg', '-nostdin', '-loglevel', 'error', '-framerate', '25', '-i',
        os.path.join(f_dir, '%04d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '25',
        '-vf', 'scale=1280:720', mp4, '-y'
    ], check=True)
    print(f'视频: {mp4}')
    print(f"样本数 {len(summary)}  accuracy: " + ', '.join(
        f'{l} {sum(s["models"][l]["correct"] for s in summary) / len(summary):.1%}'
        for l, _ in models))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--num', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--index-range', default=None,
                   help='从数据集的哪一段取样本，如 1980:2000（训练 run 的 val 就是该区间的 20 条）')
    p.add_argument('--out', default='results/clevr_grpo_3b')
    p = sub.add_parser('render')
    p.add_argument('--out', default='results/clevr_grpo_3b')
    p.add_argument('--models', nargs='+', required=True,
                   help='标签=infer结果jsonl，可多个，如 base=xx.jsonl grpo100=yy.jsonl')
    p.add_argument('--seconds', type=float, default=3.0, help='每条样本停留秒数')
    args = ap.parse_args()
    (prepare if args.cmd == 'prepare' else render)(args)


if __name__ == '__main__':
    main()
