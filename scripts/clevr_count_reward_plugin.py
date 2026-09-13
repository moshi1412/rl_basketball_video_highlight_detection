#!/usr/bin/env python3
"""任务 B（CLEVR 高计数）GRPO 用的 reward：

  * count_accuracy : <answer>{"count": N}</answer> 里的 N 与 GT 的差距（完全对=1，差 1=0.5，其余 0）
  * count_format   : 严格 `<think>…</think><answer>…</answer>` 且 JSON 可解析（1 / 0.5 / 0）
  * count_brevity  : 输出过长要扣分（避免 GEOQA 那次"越训越长、越说越错"）

用法（配合训练脚本 17）：
    --external_plugins scripts/clevr_count_reward_plugin.py \
    --reward_funcs count_accuracy count_format count_brevity \
    --reward_weights 1.0 0.2 0.2
"""
import json
import re
from typing import Any, Dict, List

from swift.rewards import ORM, orms

ANS_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
THINK_RE = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>\s*$', re.DOTALL)


def _gt(solution: Any) -> int:
    if isinstance(solution, dict):
        return int(solution.get('count', -1))
    text = str(solution)
    m = ANS_RE.search(text)
    if m:
        text = m.group(1)
    try:
        d = json.loads(text)
        if isinstance(d, dict) and 'count' in d:
            return int(d['count'])
    except Exception:
        pass
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else -1


def _pred(text: str):
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


def _pairs(completions, solution):
    if not isinstance(solution, (list, tuple)):
        solution = [solution] * len(completions)
    return zip(completions, solution)


class CountAccuracy(ORM):
    """主奖励：数对了给满分，差 1 给半分（GRPO 组内有梯度）。"""

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        out = []
        for c, s in _pairs(completions, solution):
            g, p = _gt(s), _pred(c)
            if p is None or g < 0:
                out.append(0.0)
            elif p == g:
                out.append(1.0)
            elif abs(p - g) == 1:
                out.append(0.5)
            else:
                out.append(0.0)
        return out


class CountFormat(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        out = []
        for c in completions:
            c = (c or '').strip()
            if THINK_RE.match(c) and _pred(c) is not None:
                out.append(1.0)
            elif ANS_RE.search(c) and _pred(c) is not None:
                out.append(0.5)
            else:
                out.append(0.0)
        return out


class CountBrevity(ORM):
    """200 字符以内满分，800 字符以上 0 分（线性）。"""

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        out = []
        for c in completions:
            n = len(c or '')
            if n <= 200:
                out.append(1.0)
            elif n >= 800:
                out.append(0.0)
            else:
                out.append(1.0 - (n - 200) / 600)
        return out


orms['count_accuracy'] = CountAccuracy
orms['count_format'] = CountFormat
orms['count_brevity'] = CountBrevity
