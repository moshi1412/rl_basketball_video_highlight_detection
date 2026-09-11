#!/usr/bin/env python3
"""把 11.19 的整场 3v3 视频 + highlight/*.json 标注做成可训练的片段数据。

用法示例
--------
# 1) 只统计，不落盘（先看数据量够不够）
python3 scripts/prepare_highlight_dataset.py --mode stats

# 2) 切片段 + 生成 ms-swift 数据集（默认切正例 + 等量负例）
python3 scripts/prepare_highlight_dataset.py --mode cut --task classify \
    --out-dir data/highlight_cls --limit 40

# 3) 列表式（给模型一段视频帧，让它输出该保留的时间区间）
python3 scripts/prepare_highlight_dataset.py --mode cut --task clip_list \
    --out-dir data/highlight_list --window 30 --stride 30

数据组织（2026-09-12 确认）
--------------------------
A1/A2/B3/B4 是**同一场比赛的四个机位**，-1/-2/-3 是三节（10:58 / 8:10 / 6:29）。
所以 `highlight/1-3v3.json` 标的是第 1 节，**四个机位的 -1 视频共用这条时间轴**；
2-3v3.json / 3-3v3.json 同理对应第 2、3 节。默认映射即为「一个 json -> 四个机位视频」，
可用 --map / --views 覆盖。

⚠️ 划 train/val 时要**按节或按事件划分，绝不能按机位划分**：同一事件的四个视角若被拆到
两个集合里，就是标准的泄漏（同一时刻的不同画面）。
"""
import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

DATA_ROOT = '/data/ljy23/data/videodata/11.19'

# json -> 机位视频列表（同一节的四个视角共用同一条标注时间轴）
DEFAULT_MAP = [
    ('1-3v3.json', [
        'A1/A1-1_camera1_undistorted.mp4',
        'A2/A2-1_camera1_undistorted.mp4',
        'B3/B3-1_camera1_undistorted.mp4',
        'B4/B4-1_camera1_undistorted.mp4',
    ]),
    ('2-3v3.json', [
        'A1/A1-2_camera1_undistorted.mp4',
        'A2/A2-2_camera1_undistorted.mp4',
        'B3/B3-2_camera1_undistorted.mp4',
        'B4/B4-2_camera1_undistorted.mp4',
    ]),
    ('3-3v3.json', [
        'A1/A1-3_camera1_undistorted.mp4',
        'A2/A2-3_camera1_undistorted.mp4',
        'B3/B3-3_camera1_undistorted.mp4',
        'B4/B4-3_camera1_undistorted.mp4',
    ]),
]


def hms2sec(x: str) -> float:
    parts = [int(p) for p in str(x).replace('：', ':').split(':') if p != '']
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0]) if parts else 0.0


def sec2hms(s: float) -> str:
    s = int(round(s))
    return f'{s // 60:02d}:{s % 60:02d}'


def load_segments(json_path: str) -> List[dict]:
    segs = []
    for k, v in json.load(open(json_path, encoding='utf-8')).items():
        try:
            st, et = hms2sec(v['StartTime']), hms2sec(v['EndTime'])
        except Exception:
            print(f'  [warn] {os.path.basename(json_path)} #{k} 时间字段异常，跳过: {v}')
            continue
        if et <= st:
            print(f'  [warn] {os.path.basename(json_path)} #{k} 结束<=开始，跳过: {v}')
            continue
        segs.append({
            'idx': k,
            'start': st,
            'end': et,
            'description': str(v.get('Description', '')).strip(),
            'ids': v.get('ID', []),
        })
    segs.sort(key=lambda x: x['start'])
    return segs


def iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclass
class Candidate:
    view: str
    video: str
    start: float
    end: float
    max_iou: float
    seg: dict

    @property
    def positive(self) -> bool:
        return self.max_iou >= 0.5


def build_candidates(view: str, video_rel: str, segs: List[dict], win: float, stride: float,
                     duration: float) -> List[Candidate]:
    out, t = [], 0.0
    while t + win <= duration + 1e-6:
        best, best_seg = 0.0, None
        for s in segs:
            v = iou((t, t + win), (s['start'], s['end']))
            if v > best:
                best, best_seg = v, s
        out.append(Candidate(view, video_rel, t, t + win, best, best_seg or {'description': '', 'ids': []}))
        t += stride
    return out


def probe_duration(path: str) -> float:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path]
    return float(subprocess.check_output(cmd).decode().strip())


def cut_frames(video: str, start: float, end: float, out_dir: str, n_frames: int, width: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    dur = max(0.2, end - start)
    fps = max(0.2, n_frames / dur)
    cmd = [
        'ffmpeg', '-nostdin', '-loglevel', 'error', '-ss',
        f'{start:.3f}', '-i', video, '-t', f'{dur:.3f}', '-vf',
        f'fps={fps:.4f},scale={width}:-2', '-q:v', '4', '-frames:v', str(n_frames),
        os.path.join(out_dir, '%03d.jpg'), '-y'
    ]
    subprocess.run(cmd, check=True)
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg'))


def make_question(cand: Candidate, task: str) -> str:
    if task == 'classify':
        return (f'这是篮球 3v3 比赛中的一段 {cand.end - cand.start:.1f} 秒片段'
                f'（比赛时间 {sec2hms(cand.start)}~{sec2hms(cand.end)}）。'
                '请判断它是否值得剪进高光集锦：先给出简短理由（攻防类型、是否有得分/助攻/抢断/盖帽等），'
                '最后用 <answer>yes</answer> 或 <answer>no</answer> 结尾。')
    return ('下面是篮球 3v3 比赛的一段连续画面。请找出其中值得剪进高光集锦的时间段，'
            '每个时间段给出起止秒数（相对于本段开头）和理由。'
            '最后用 <answer>[{"start": 3.0, "end": 8.5, "reason": "..."}]</answer> 的 JSON 列表结尾。')


def make_answer(cand: Candidate, task: str) -> str:
    """造 SFT 的 assistant 目标（--with-answer 时写入 messages，可直接用于 swift sft）。"""
    if task == 'classify':
        desc = cand.seg.get('description') or ''
        if cand.positive:
            reason = f'片段包含“{desc}”这类关键攻防回合，值得剪进集锦。' if desc else '片段包含关键得分回合。'
            return f'{reason}<answer>yes</answer>'
        return '画面只是常规运球/站位，没有得分或关键防守回合，不值得剪进集锦。<answer>no</answer>'
    ivs = [[round(cand.start, 1), round(cand.end, 1)]] if cand.positive else []
    return f'<answer>{json.dumps(ivs, ensure_ascii=False)}</answer>'


def write_rows(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def stats(args, mapping):
    total_segs, total_pos, total_cand = 0, 0, 0
    for jf, views in mapping:
        segs = load_segments(os.path.join(DATA_ROOT, 'highlight', jf))
        hl_time = sum(s['end'] - s['start'] for s in segs)
        total_segs += len(segs)
        print(f'\n{jf}: 高光 {len(segs)} 段 / {hl_time:.1f}s（该节 4 个机位共用此标注）')
        for vf in select_views(views, args):
            dur = probe_duration(os.path.join(DATA_ROOT, vf))
            cands = build_candidates(view_name(vf), vf, segs, args.window, args.stride, dur)
            pos = [c for c in cands if c.positive]
            total_pos += len(pos)
            total_cand += len(cands)
            print(f'   {vf:<42} 视频 {dur:6.1f}s | 候选 {len(cands):4d}'
                  f' (win={args.window}s stride={args.stride}s)'
                  f' | 正例 {len(pos):4d} ({len(pos) / max(1, len(cands)) * 100:.1f}%)')
    print(f'\n合计: 标注高光 {total_segs} 段, 候选 {total_cand} 个, 正例 {total_pos} 个, '
          f'正例率 {total_pos / max(1, total_cand) * 100:.1f}%')
    print('说明: 每个事件的 4 个视角画面不同、时间轴相同，可直接做多视角增广；'
          '但划 train/val 必须按“节/事件”划分（见 docs/highlight_clip_rl_design.md）。')


def view_name(video_rel: str) -> str:
    return video_rel.split('/')[0]


def select_views(views, args):
    if not args.views:
        return views
    want = {v.strip().upper() for v in args.views.split(',')}
    return [v for v in views if view_name(v).upper() in want]


def cut(args, mapping):
    random.seed(args.seed)
    neg_per_pos = max(0, args.neg_ratio)
    for jf, views in mapping:
        segs = load_segments(os.path.join(DATA_ROOT, 'highlight', jf))
        tag = os.path.splitext(jf)[0]
        for vf in select_views(views, args):
            video = os.path.join(DATA_ROOT, vf)
            dur = probe_duration(video)
            cands = build_candidates(view_name(vf), vf, segs, args.window, args.stride, dur)
            pos = [c for c in cands if c.positive]
            neg = [c for c in cands if not c.positive]
            if args.limit:
                pos = pos[:args.limit]
            n_neg = min(len(neg), len(pos) * neg_per_pos)
            neg = random.sample(neg, n_neg) if n_neg else []

            rows = []
            for label, group in (('pos', pos), ('neg', neg)):
                for c in group:
                    stem = f'{tag}_{c.view}_{c.start:07.1f}_{c.end:07.1f}'
                    frames = cut_frames(video, c.start, c.end,
                                        os.path.join(args.out_dir, 'frames', stem),
                                        args.frames, args.width)
                    messages = [{'role': 'user', 'content': make_question(c, args.task)}]
                    if args.with_answer:
                        messages.append({'role': 'assistant', 'content': make_answer(c, args.task)})
                    rows.append({
                        'messages': messages,
                        'images': frames,
                        'solution': json.dumps({
                            'is_highlight': bool(c.positive),
                            'interval': [c.start, c.end],
                            'max_iou': round(c.max_iou, 3),
                            'description': c.seg.get('description', ''),
                            'ids': c.seg.get('ids', []),
                        }, ensure_ascii=False),
                        'meta': {'video': vf, 'view': c.view, 'start': c.start, 'end': c.end, 'label': label},
                    })
            out = os.path.join(args.out_dir, f'{tag}_{view_name(vf)}.jsonl')
            write_rows(rows, out)
            print(f'{tag} @ {view_name(vf)}: 写入 {len(rows)} 条 (正 {len(pos)} / 负 {len(neg)}) -> {out}')

    print('\n提示: 一个事件有 4 个视角(画面不同、时间轴相同)，划 train/val 请按“节/事件”切，'
          '不要把同一事件的视角拆到两个集合。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['stats', 'cut'], default='stats')
    ap.add_argument('--task', choices=['classify', 'clip_list'], default='classify')
    ap.add_argument('--map', nargs='*', default=None,
                    help='覆盖映射，格式 json=视频1,视频2,...（同一节的多个机位）')
    ap.add_argument('--views', default=None, help='只使用指定机位，如 A1 或 A1,B3')
    ap.add_argument('--window', type=float, default=5.0)
    ap.add_argument('--stride', type=float, default=1.0)
    ap.add_argument('--frames', type=int, default=8, help='每个片段抽几帧给 VLM')
    ap.add_argument('--width', type=int, default=448)
    ap.add_argument('--neg-ratio', type=int, default=1, help='每个正例配几个负例')
    ap.add_argument('--limit', type=int, default=0, help='每个视频最多取多少正例(0=全部)')
    ap.add_argument('--out-dir', default='data/highlight')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--with-answer', action='store_true',
                    help='额外写入 assistant 目标回答，生成的 jsonl 可直接用于 swift sft')
    args = ap.parse_args()

    mapping = DEFAULT_MAP
    if args.map:
        mapping = [(x.split('=', 1)[0], x.split('=', 1)[1].split(',')) for x in args.map]
    print('json -> 机位视频 映射:')
    for jf, views in mapping:
        print(f'  {jf:>14} -> {", ".join(views)}')
    print()
    if args.mode == 'stats':
        stats(args, mapping)
    else:
        cut(args, mapping)


if __name__ == '__main__':
    main()
