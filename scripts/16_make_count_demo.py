#!/usr/bin/env python3
"""任务 B（CLEVR 高计数）的评测 + demo 视频生成。

判定口径：
  * accuracy = `<answer>{"count": N}</answer>` 里的 N 与 GT 完全一致
  * format   = 形如 `<think>…</think><answer>…</answer>`（严格：必须以 <think> 开头、以 </answer> 结尾）

用法：
  python3 scripts/16_make_count_demo.py --out results/clevr_count_hard \
      --models "base=xx.jsonl" "SFT=yy.jsonl" ["GRPO=zz.jsonl"]
"""
import argparse
import json
import os
import re
import subprocess
from typing import Dict, List

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
ANS_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
FMT_RE = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>\s*$', re.DOTALL)


def parse_count(text: str):
    m = ANS_RE.search(text or '')
    if not m:
        return None
    payload = m.group(1).strip()
    try:
        d = json.loads(payload)
        if isinstance(d, dict) and 'count' in d:
            return int(d['count'])
    except Exception:
        pass
    nums = re.findall(r'\d+', payload)
    return int(nums[0]) if nums else None


def is_valid_format(text: str) -> bool:
    return bool(FMT_RE.match((text or '').strip()))


def read_results(path: str) -> Dict[str, dict]:
    out = {}
    for line in open(path, encoding='utf-8'):
        d = json.loads(line)
        imgs = d.get('images') or []
        key = os.path.basename(str(imgs[0].get('path', ''))) if imgs and isinstance(imgs[0], dict) else ''
        out[key] = d
    return out


def wrap(draw, text, font, x, y, width, line_h, fill=(20, 20, 20), max_lines=12):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/clevr_count_hard')
    ap.add_argument('--models', nargs='+', required=True, help='标签=infer结果jsonl')
    ap.add_argument('--video-num', type=int, default=20)
    ap.add_argument('--seconds', type=float, default=3.0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    inp = [json.loads(l) for l in open(os.path.join(args.out, 'samples', 'infer_input.jsonl'),
                                       encoding='utf-8')]
    gt = {os.path.basename(r['images'][0]): json.loads(r['solution'])['count'] for r in inp}
    models = [(m.split('=', 1)[0], read_results(m.split('=', 1)[1])) for m in args.models]

    summary = []
    for r in inp:
        key = os.path.basename(r['images'][0])
        item = {'sample': key, 'ground_truth': gt[key], 'models': {}}
        for label, res in models:
            resp = res.get(key, {}).get('response', '') or ''
            pred = parse_count(resp)
            item['models'][label] = {
                'answer': pred,
                'correct': pred == gt[key],
                'format_ok': is_valid_format(resp),
                'chars': len(resp),
                'response': resp,
            }
        summary.append(item)

    stats = {label: {'accuracy': round(sum(s['models'][label]['correct'] for s in summary) / len(summary), 4),
                     'format': round(sum(s['models'][label]['format_ok'] for s in summary) / len(summary), 4),
                     'mean_chars': round(sum(s['models'][label]['chars'] for s in summary) / len(summary))}
             for label, _ in models}
    os.makedirs(os.path.join(args.out, 'metrics'), exist_ok=True)
    with open(os.path.join(args.out, 'metrics', 'eval.json'), 'w', encoding='utf-8') as f:
        json.dump({'n': len(summary), 'stats': stats}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, 'samples', 'vlm_answers.jsonl'), 'w', encoding='utf-8') as f:
        for s in summary:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    os.makedirs(os.path.join(args.out, 'samples'), exist_ok=True)
    with open(os.path.join(args.out, 'samples', 'vlm_answers_readable.md'), 'w', encoding='utf-8') as f:
        for s in summary:
            f.write(f"### {s['sample']}  GT count = {s['ground_truth']}\n\n")
            for label, _ in models:
                m = s['models'][label]
                f.write(f"- {label}: count={m['answer']} {'✅' if m['correct'] else '❌'}"
                        f"（格式{'OK' if m['format_ok'] else '不合格'}）\n")
                f.write(f"  > {m['response'].replace(chr(10), ' ')[:300]}\n\n")

    print(f"评测集 n={len(summary)}")
    for label, st in stats.items():
        print(f"  {label:>14}: accuracy={st['accuracy']:.1%}  格式合规={st['format']:.1%}  平均 {st['mean_chars']} 字符")

    # ---- 渲染 demo 视频 ----
    f_dir = os.path.join(args.out, 'demo', 'frames')
    os.makedirs(f_dir, exist_ok=True)
    f_title, f_txt, f_small = (ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 19),
                               ImageFont.truetype(FONT, 17))
    W, H = 1280, 720
    idx = 0
    for i, (r, s) in enumerate(zip(inp, summary)):
        if i >= args.video_num:
            break
        img = Image.open(r['images'][0]).convert('RGB')
        img.thumbnail((600, 420))
        canvas = Image.new('RGB', (W, H), (252, 252, 252))
        canvas.paste(img, (30, 150 + (420 - img.height) // 2))
        dr = ImageDraw.Draw(canvas)
        dr.text((30, 24), f'CLEVR hard counting (count>=8) — SFT / GRPO demo   sample {i + 1}/{min(args.video_num, len(inp))}',
                font=f_title, fill=(10, 10, 10))
        dr.text((30, 66), 'Q: How many distinct objects are there?', font=f_txt, fill=(80, 80, 80))
        dr.text((30, 96), f'Ground truth: {s["ground_truth"]}', font=f_txt, fill=(20, 100, 20))
        x, y = 660, 150
        dr.rectangle([x - 12, y - 14, W - 24, H - 30], outline=(215, 215, 215), width=2)
        for label, _ in models:
            m = s['models'][label]
            mark = 'O' if m['correct'] else 'X'
            dr.text((x, y), f'{label}: count = {m["answer"]}   [{mark}]', font=f_txt,
                    fill=(20, 130, 20) if m['correct'] else (190, 40, 40))
            y += 26
            y = wrap(dr, m['response'].replace('\n', ' '), f_small, x, y, 580, 22,
                     fill=(105, 105, 105), max_lines=5 if len(models) > 2 else 8)
            y += 16
        dr.text((30, H - 44), f"n={len(summary)}   " + '   '.join(
            f"{label}: {st['accuracy']:.0%} (格式 {st['format']:.0%})" for label, st in stats.items()),
            font=f_txt, fill=(60, 60, 60))
        for _ in range(int(args.seconds * 25)):
            canvas.save(os.path.join(f_dir, f'{idx:04d}.jpg'), quality=92)
            idx += 1
    mp4 = os.path.join(args.out, 'demo', 'clevr_count_demo.mp4')
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-framerate', '25', '-i',
                    os.path.join(f_dir, '%04d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-r', '25', '-vf', 'scale=1280:720', mp4, '-y'], check=True)
    print('视频:', mp4)


if __name__ == '__main__':
    main()
