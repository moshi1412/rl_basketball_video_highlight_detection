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
import os
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


class HighlightEvidence(_Base):
    """理由里是否提到“可观测事实”（投篮/上篮/防守/篮板/抢断…）。

    与 HighlightReason 的区别：不看标注措辞，只看模型有没有说出画面里能看到的攻防要素，
    因此对模型是**可学**的（上一次实验里 HighlightReason 一直卡在 0.1~0.2 就是因为它要求
    复述标注原文）。负例里说“这球被防住了/没进/只是运球”同样算证据。
    """

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            kw = _keywords(content)
            if not kw:
                rewards.append(0.0)
                continue
            rewards.append(min(1.0, 0.4 + 0.2 * len(kw)))
        return rewards


class HighlightLabelBalanced(_Base):
    """分类主奖励（playbook 推荐的"平衡 + 反退化"版本）。

    * 每题判对得 1.0；判对且理由里出现了标注事件词再 +0.2（归一化到 1.0）
    * **反退化惩罚**：如果整批 completion 的预测几乎全是 yes（>90%）或全是 no（<10%），
      说明策略退化成"恒答同一个标签"（前面实验里 base 恒 yes、SFT 恒 no），
      整批奖励统一扣 0.3，让"会区分的策略"严格优于"恒答策略"。
    """

    def __init__(self, args=None, degenerate_penalty: float = 0.3, **kwargs):
        super().__init__(args, **kwargs)
        self.degenerate_penalty = degenerate_penalty

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards, preds = [], []
        for content, sol in self._pairs(completions, solution):
            gt = _parse_solution(sol)
            payload = (_parse_answer(content) or '').lower()
            pred = payload.startswith('yes')
            preds.append(pred)
            if 'is_highlight' not in gt:
                rewards.append(0.0)
                continue
            r = 1.0 if pred == bool(gt['is_highlight']) else 0.0
            if r > 0 and gt.get('description') and (_keywords(content) & _keywords(gt['description'])):
                r += 0.2
            rewards.append(min(r, 1.2) / 1.2)
        if preds:
            pos_rate = sum(preds) / len(preds)
            if pos_rate > 0.9 or pos_rate < 0.1:
                rewards = [max(0.0, r - self.degenerate_penalty) for r in rewards]
        return rewards


def _group_labels(preds, group_size):
    """把整批预测按 group_size 切组（GRPO 里同一 prompt 的 num_generations 条是连续的）。"""
    return [preds[i:i + group_size] for i in range(0, len(preds), group_size)]


class HighlightLabelFocused(_Base):
    """只看判断的奖励（配合「答案前置」数据）：

    * 判对 = 1.0（负例）/ yes_weight（正例，默认 2.0），判错 = 0
      —— yes_weight 取"正负样本比例的反比"（数据是 1:2 就取 2.0），
      这样"恒答 no"的期望奖励不再占优：E = (1/3)·w·p + (2/3)·(1-p)，
      w=2 时与 p 无关，先验导致的坍缩压力被消掉，梯度只来自"能不能区分"。
    * 解析不出答案 = 0
    * **探索奖励**：如果同组里你是少数派（跟多数回答不同），额外 +bonus；
      上限 = max(yes_weight, 1) + bonus（**不封在 1.0**，否则"敢于不同"的加分会被上限吃掉）。
    """

    def __init__(self, args=None, bonus: float = 0.3, yes_weight: float = None, **kwargs):
        super().__init__(args, **kwargs)
        self.bonus = float(os.getenv('EXPLORE_BONUS', bonus))
        self.yes_weight = float(yes_weight if yes_weight is not None else os.getenv('YES_WEIGHT', '2.0'))
        self.cap = max(self.yes_weight, 1.0) + self.bonus
        self.group_size = int(os.getenv('GRPO_NUM_GENERATIONS', '4'))

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards, preds = [], []
        for content, sol in self._pairs(completions, solution):
            gt = _parse_solution(sol)
            payload = (_parse_answer(content) or '').lower()
            if payload.startswith('yes'):
                pred = True
            elif payload.startswith('no'):
                pred = False
            else:
                pred = None
            preds.append(pred)
            if pred is None:
                rewards.append(0.0)
            elif pred == bool(gt.get('is_highlight', False)):
                rewards.append(self.yes_weight if pred else 1.0)
            else:
                rewards.append(0.0)

        # 探索奖励：组内"少数派"加分（逐样本，能给退化策略制造组内方差）
        if self.bonus > 0 and preds:
            for gi, g in enumerate(_group_labels(preds, self.group_size)):
                known = [p for p in g if p is not None]
                if not known:
                    continue
                majority_yes = sum(known) * 2 > len(known)
                for j, p in enumerate(g):
                    if p is not None and p != majority_yes:
                        k = gi * self.group_size + j
                        rewards[k] = min(self.cap, rewards[k] + self.bonus)
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
orms['highlight_label_balanced'] = HighlightLabelBalanced
orms['highlight_label_focused'] = HighlightLabelFocused
orms['highlight_reason'] = HighlightReason
orms['highlight_evidence'] = HighlightEvidence
orms['highlight_temporal'] = HighlightTemporal
orms['highlight_length'] = HighlightLength
