#!/usr/bin/env python3
"""高光片段二分类的评测 + demo 视频（playbook 要求的 balanced accuracy / Precision / Recall）。

用法：
  python3 scripts/20_make_highlight_demo.py --out results/highlight_cls_v2 \
      --models "base=xx.jsonl" "SFT=yy.jsonl" "GRPO=zz.jsonl" [--video-num 24]
"""
import argparse
import json
import os
import re
import subprocess

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
ANS_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
# v3 数据是「答案前置」：<answer>yes/no</answer><reason>…</reason>
# 所以"格式合规"定义为：以 <answer>yes|no</answer> **开头**（理由可有可无）。
FMT_RE = re.compile(r'^\s*<answer>\s*(yes|no)\s*</answer>', re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def parse_label(text: str):
    m = ANS_RE.search(text or '')
    if not m:
        return None
    payload = m.group(1).strip().lower()
    if payload.startswith('yes'):
        return True
    if payload.startswith('no'):
        return False
    return None


def read_results(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        d = json.loads(line)
        imgs = d.get('images') or []
        if imgs and isinstance(imgs[0], dict):
            out[os.path.basename(os.path.dirname(str(imgs[0]['path'])))] = d
    return out


def metrics(rows, label):
    tp = fp = tn = fn = fm = 0
    for r in rows:
        p = r['models'][label]['pred']
        g = r['is_highlight']
        if p is None:
            fn += g
            fp += (not g)
            continue
        fm += r['models'][label]['format_ok']
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and g:
            fn += 1
        else:
            tn += 1
    n = len(rows)
    acc = (tp + tn) / n
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    bal = 0.5 * (rec + (tn / (tn + fp) if (tn + fp) else 0.0))
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {'n': n, 'accuracy': round(acc, 4), 'balanced_accuracy': round(bal, 4),
            'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
            'format': round(fm / n, 4), 'pred_positive': tp + fp}


def wrap(draw, text, font, x, y, width, line_h, fill=(20, 20, 20), max_lines=10):
    lines, cur = [], ''
    for ch in text:
        if ch == '\n':
            lines.append(cur); cur = ''; continue
        cur += ch
        if draw.textlength(cur, font=font) > width:
            lines.append(cur); cur = ''
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    for i, ln in enumerate(lines):
        draw.text((x, y + i * line_h), ln, font=font, fill=fill)
    return y + len(lines) * line_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/highlight_cls_v2')
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--video-num', type=int, default=24)
    ap.add_argument('--seconds', type=float, default=3.0)
    args = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont

    inp = [json.loads(l) for l in open(os.path.join(args.out, 'samples', 'infer_input.jsonl'), encoding='utf-8')]
    models = [(m.split('=', 1)[0], read_results(m.split('=', 1)[1])) for m in args.models]

    rows = []
    for r in inp:
        clip = os.path.basename(os.path.dirname(r['images'][0]))
        item = {'clip': clip, 'images': r['images'], 'is_highlight': r['meta']['is_highlight'],
                'description': r['meta']['description'], 'question': r['messages'][0]['content'],
                'models': {}}
        for label, res in models:
            d = res.get(clip, {})
            resp = d.get('response', '') or ''
            item['models'][label] = {'pred': parse_label(resp), 'response': resp,
                                     'format_ok': bool(FMT_RE.match(resp.strip())),
                                     'has_think': bool(THINK_RE.search(resp))}
        rows.append(item)

    stats = {label: metrics(rows, label) for label, _ in models}
    os.makedirs(os.path.join(args.out, 'metrics'), exist_ok=True)
    with open(os.path.join(args.out, 'metrics', 'eval.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.join(args.out, 'samples'), exist_ok=True)
    with open(os.path.join(args.out, 'samples', 'vlm_answers.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(args.out, 'samples', 'vlm_answers_readable.md'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(f"### {r['clip']}  GT={'高光' if r['is_highlight'] else '非高光'}"
                    f"{('  ' + r['description']) if r['description'] else ''}\n\n")
            for label, _ in models:
                m = r['models'][label]
                ok = (m['pred'] == r['is_highlight'])
                f.write(f"- {label}: {'yes' if m['pred'] else ('no' if m['pred'] is not None else '解析失败')}"
                        f" {'✅' if ok else '❌'}\n  > {m['response'].replace(chr(10), ' ')[:260]}\n\n")
    print(f"验证集 n={len(rows)}（正例 {sum(r['is_highlight'] for r in rows)}）")
    for label, st in stats.items():
        print(f"  {label:>10}: balanced_acc={st['balanced_accuracy']:.1%} acc={st['accuracy']:.1%} "
              f"P={st['precision']:.1%} R={st['recall']:.1%} F1={st['f1']:.2f} 格式={st['format']:.0%} "
              f"预测正例={st['pred_positive']}")

    f_dir = os.path.join(args.out, 'demo', 'frames')
    os.makedirs(f_dir, exist_ok=True)
    f_title, f_txt, f_small = (ImageFont.truetype(FONT, 28), ImageFont.truetype(FONT, 19),
                               ImageFont.truetype(FONT, 16))
    W, H = 1280, 720
    idx = 0
    for i, r in enumerate(rows):
        if i >= args.video_num:
            break
        strip = Image.new('RGB', (620, 320), (255, 255, 255))
        for j, p in enumerate(r['images'][:8]):
            im = Image.open(p).convert('RGB')
            im.thumbnail((150, 150))
            strip.paste(im, (5 + (j % 4) * 155, 5 + (j // 4) * 155))
        canvas = Image.new('RGB', (W, H), (252, 252, 252))
        canvas.paste(strip, (25, 170))
        dr = ImageDraw.Draw(canvas)
        dr.text((25, 22), f'Basketball highlight clip classification   样本 {i + 1}/{min(args.video_num, len(rows))}',
                font=f_title, fill=(10, 10, 10))
        dr.text((25, 62), f"问题：这段 5 秒片段值得剪进高光吗？（4 机位同一场比赛，第 3 节留出验证）",
                font=f_txt, fill=(80, 80, 80))
        gt_txt = '高光' if r['is_highlight'] else '非高光'
        dr.text((25, 96), f"GT：{gt_txt}   标注描述：{r['description'] or '（无）'}", font=f_txt, fill=(20, 100, 20))
        x, y = 665, 170
        dr.rectangle([x - 12, y - 14, W - 22, H - 30], outline=(215, 215, 215), width=2)
        for label, _ in models:
            m = r['models'][label]
            ok = (m['pred'] == r['is_highlight'])
            ans = 'yes' if m['pred'] else ('no' if m['pred'] is not None else '?')
            dr.text((x, y), f"{label}: {ans}  [{'O' if ok else 'X'}]", font=f_txt,
                    fill=(20, 130, 20) if ok else (190, 40, 40))
            y += 25
            y = wrap(dr, m['response'].replace('\n', ' '), f_small, x, y, 570, 21,
                     fill=(105, 105, 105), max_lines=6 if len(models) > 2 else 9)
            y += 14
        dr.text((25, H - 46), f"n={len(rows)}   " + '   '.join(
            f"{label}: balAcc {st['balanced_accuracy']:.0%} P {st['precision']:.0%} R {st['recall']:.0%}"
            for label, st in stats.items()), font=f_txt, fill=(60, 60, 60))
        for _ in range(int(args.seconds * 25)):
            canvas.save(os.path.join(f_dir, f'{idx:04d}.jpg'), quality=92)
            idx += 1
    mp4 = os.path.join(args.out, 'demo', 'highlight_cls_demo.mp4')
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-framerate', '25', '-i',
                    os.path.join(f_dir, '%04d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-r', '25', '-vf', 'scale=1280:720', mp4, '-y'], check=True)
    print('视频:', mp4)


if __name__ == '__main__':
    main()
