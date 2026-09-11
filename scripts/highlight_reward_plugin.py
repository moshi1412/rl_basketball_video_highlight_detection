#!/usr/bin/env python3
"""高光剪辑 RLHF 的 reward 插件（ms-swift ORM）。

用法（配合 prepare_highlight_dataset.py 生成的 jsonl）:

    swift rlhf --rlhf_type grpo \
      --external_plugins scripts/highlight_reward_plugin.py \
      --reward_funcs highlight_format highlight_label highlight_reason \
      --reward_weights 0.2 1.0 0.3

列表式剪辑任务（--task clip_list 的数据）:

    swift rlhf --rlhf_type grpo \
      --external_plugins scripts/highlight_reward_plugin.py \
      --reward_funcs highlight_format highlight_temporal highlight_length \
      --reward_weights 0.2 1.0 0.3

实现说明：
1. 所有 reward 都从 completion 里解析 <answer>...</answer>；
2. 数据集里的 solution 是 JSON 字符串（见 prepare_highlight_dataset.py），GRPO 会原样透传给 ORM；
3. reward 都做了平滑（0~1 连续），避免 GRPO 组内全 0 导致无梯度。
"""
import json
import re
from typing import Any, Dict, List

from swift.rewards import ORM, orms

ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

# 高光事件的“事实关键词”，用于和标注 Description 做密集匹配
EVENT_KEYWORDS = {
    '得分': ['得分', '命中', '打进', '得手', '轻取', '拿下', '入网', 'score'],
    '投篮': ['投篮', '跳投', '中投', '三分', '远投', '后撤', '干拔', 'jumper', 'shot'],
    '上篮': ['上篮', '打板', '拉杆', '对抗', '突破', 'layup', 'drive'],
    '扣篮': ['扣篮', '暴扣', 'dunk'],
    '助攻': ['助攻', '传球', '分球', '妙传', 'assist', 'pass'],
    '抢断': ['抢断', '断球', 'steal'],
    '盖帽': ['盖帽', '封盖', '帽掉', 'block'],
    '篮板': ['篮板', '二次进攻', '补篮', 'rebound'],
    '不中/失误': ['不中', '失误', '打铁', '出界', '被断', 'miss', 'turnover'],
}


def _parse_solution(solution: Any) -> Dict[str, Any]:
    if isinstance(solution, dict):
        return solution
    text = str(solution)
    m = ANSWER_RE.search(text)
    if m:
        text = m.group(1)
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {'intervals': d}
    except Exception:
        return {'raw': text}


def _parse_answer(text: str):
    m = ANSWER_RE.search(text or '')
    return m.group(1).strip() if m else None


def _keywords(text: str) -> set:
    text = (text or '').lower()
    return {name for name, words in EVENT_KEYWORDS.items() if any(w.lower() in text for w in words)}


def _temporal_iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _parse_intervals(payload: str) -> List[List[float]]:
    if payload is None:
        return []
    try:
        data = json.loads(payload)
    except Exception:
        pairs = re.findall(r'(\d+(?:\.\d+)?)\s*(?:-|~|到|至)\s*(\d+(?:\.\d+)?)', payload)
        return [[float(a), float(b)] for a, b in pairs if float(b) > float(a)]
    out = []
    if isinstance(data, dict):
        data = data.get('highlights') or data.get('intervals') or [data]
    for d in (data if isinstance(data, list) else []):
        if isinstance(d, dict):
            s, e = d.get('start', d.get('start_time')), d.get('end', d.get('end_time'))
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            s, e = d[0], d[1]
        else:
            continue
        try:
            s, e = float(s), float(e)
        except (TypeError, ValueError):
            continue
        if e > s:
            out.append([s, e])
    return out


class _Base(ORM):

    def _pairs(self, completions, solution):
        if solution is None:
            solution = [''] * len(completions)
        if not isinstance(solution, (list, tuple)):
            solution = [solution] * len(completions)
        return zip(completions, solution)


class HighlightFormat(_Base):
    """格式 reward：必须有 <answer>，且内容可解析（0 / 0.5 / 1）。"""

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards = []
        for content, sol in self._pairs(completions, solution):
            payload = _parse_answer(content)
            if payload is None:
                rewards.append(0.0)
                continue
            if 'is_highlight' in _parse_solution(sol):
                rewards.append(1.0 if payload.lower().startswith(('yes', 'no')) else 0.5)
            else:
                rewards.append(1.0 if _parse_intervals(payload) else 0.5)
        return rewards


class HighlightLabel(_Base):
    """分类式主 reward：yes/no 判对得 1；理由里事件关键词也命中的再 +0.2。"""

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards = []
        for content, sol in self._pairs(completions, solution):
            gt = _parse_solution(sol)
            payload = (_parse_answer(content) or '').lower()
            if 'is_highlight' not in gt:
                rewards.append(0.0)
                continue
            pred = payload.startswith('yes')
            r = 1.0 if pred == bool(gt.get('is_highlight')) else 0.0
            if r > 0 and gt.get('description') and (_keywords(content) & _keywords(gt['description'])):
                r += 0.2
            rewards.append(min(r, 1.2) / 1.2)
        return rewards


class HighlightReason(_Base):
    """理由 reward：completion 与标注 Description 的事件关键词重合度（0~1）。"""

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards = []
        for content, sol in self._pairs(completions, solution):
            gt_kw = _keywords(str(_parse_solution(sol).get('description', '')))
            rewards.append(len(gt_kw & _keywords(content)) / len(gt_kw) if gt_kw else 0.0)
        return rewards


class HighlightTemporal(_Base):
    """列表式主 reward：与标注区间的匹配 F1，tIoU 0.3 起给分、0.5 满分。"""

    def __init__(self, args=None, iou_threshold: float = 0.5, **kwargs):
        super().__init__(args, **kwargs)
        self.iou_threshold = iou_threshold

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards = []
        for content, sol in self._pairs(completions, solution):
            gt = _parse_solution(sol)
            gt_iv = [list(map(float, iv)) for iv in gt.get('intervals', [])]
            if not gt_iv and 'interval' in gt:
                gt_iv = [list(map(float, gt['interval']))]
            pred = _parse_intervals(_parse_answer(content))
            if not pred or not gt_iv:
                rewards.append(0.0)
                continue
            used, matched, score = set(), 0, 0.0
            for p in pred:
                best_i, best = -1, 0.0
                for i, g in enumerate(gt_iv):
                    if i in used:
                        continue
                    v = _temporal_iou(p, g)
                    if v > best:
                        best_i, best = i, v
                if best_i >= 0 and best >= 0.3:
                    used.add(best_i)
                    matched += 1
                    score += 1.0 if best >= self.iou_threshold else (best - 0.3) / (self.iou_threshold - 0.3)
            precision = matched / len(pred)
            recall = len(used) / len(gt_iv)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            rewards.append(0.5 * f1 + 0.5 * (score / len(pred)))
        return rewards


class HighlightLength(_Base):
    """时长/数量先验：总时长占比接近标注、片段数别太多、互相不重叠。"""

    def __init__(self, args=None, ideal_ratio=(0.08, 0.3), max_clips: int = 12, **kwargs):
        super().__init__(args, **kwargs)
        self.ideal_ratio = ideal_ratio
        self.max_clips = max_clips

    def __call__(self, completions, solution=None, video_seconds: float = 600.0, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            pred = _parse_intervals(_parse_answer(content))
            if not pred:
                rewards.append(0.0)
                continue
            ratio = sum(e - s for s, e in pred) / max(1.0, float(video_seconds))
            lo, hi = self.ideal_ratio
            r_len = 1.0 if lo <= ratio <= hi else max(0.0, 1 - abs(ratio - lo) / max(lo, 1e-6))
            r_num = 1.0 if len(pred) <= self.max_clips else max(0.0, self.max_clips / len(pred))
            ov = 0.0
            for i in range(len(pred)):
                for j in range(i + 1, len(pred)):
                    ov = max(ov, _temporal_iou(pred[i], pred[j]))
            rewards.append(0.4 * r_len + 0.3 * r_num + 0.3 * (1.0 - min(1.0, ov / 0.7)))
        return rewards


orms['highlight_format'] = HighlightFormat
orms['highlight_label'] = HighlightLabel
orms['highlight_reason'] = HighlightReason
orms['highlight_temporal'] = HighlightTemporal
orms['highlight_length'] = HighlightLength
