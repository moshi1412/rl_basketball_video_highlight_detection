#!/usr/bin/env python3
"""自定义 SFT loss：给"答案 token"加权，让 loss 由判断决定，而不是被 reason 模板稀释。

背景（实测数据）：
  * 一条回答约 52 字，其中真正决定分类对错的只有 <answer>yes/no</answer>（约 4~5 个 token）；
  * 负例目标是一条固定句（1568 条完全相同），模型背下它就白拿 2/3 样本的 token 准确率；
  * 逐 token 平均的交叉熵里，"判断"只占 ~2% 权重 → SFT 直接坍缩成"恒答 no"。

做法：
  训练数据用「答案前置」格式：`<answer>yes</answer><reason>三分命中</reason>`
  本 loss 把**每行回答的前 K 个 token（即 <answer>…</answer>）权重放大 ANSWER_WEIGHT 倍**，
  其余 token 权重 1。这样"答对/答错"成为 loss 的主导项，模型无法靠背 reason 模板降低损失。

用法：
    swift sft ... --external_plugins scripts/highlight_sft_loss_plugin.py --loss_type answer_weighted
环境变量：ANSWER_TOKENS（默认 8）、ANSWER_WEIGHT（默认 10）、POS_WEIGHT（正例样本的额外权重，默认 1.0）
"""
import os

import torch

from swift.loss import BaseLoss, loss_map


class AnswerWeightedLoss(BaseLoss):

    def __init__(self, args, trainer, k_tokens: int = None, weight: float = None, pos_weight: float = None):
        super().__init__(args, trainer)
        self.k_tokens = int(k_tokens if k_tokens is not None else os.getenv('ANSWER_TOKENS', '8'))
        self.weight = float(weight if weight is not None else os.getenv('ANSWER_WEIGHT', '10'))
        self.pos_weight = float(pos_weight if pos_weight is not None else os.getenv('POS_WEIGHT', '1.0'))
        self._yes_ids = None

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        from swift.trainers import per_token_loss_func
        # per_token_loss_func 返回的是按 (batch*seq) 展平的逐 token loss，忽略位为 0
        token_loss = per_token_loss_func(outputs, labels)                    # [B*L]
        flat_labels = torch.roll(labels, shifts=-1, dims=-1).reshape(-1)      # 与上面同样的 shift
        valid = flat_labels != -100
        batch_size, seq_len = labels.shape

        w = torch.ones_like(token_loss)
        if self.weight != 1.0:
            valid_2d = valid.view(batch_size, seq_len)
            flat_2d = flat_labels.view(batch_size, seq_len)
            tokenizer = getattr(self.trainer, 'processing_class', None)
            if self.pos_weight != 1.0 and tokenizer is not None and self._yes_ids is None:
                self._yes_ids = set(tokenizer.encode('yes', add_special_tokens=False))
            for b in range(batch_size):
                idx = torch.nonzero(valid_2d[b], as_tuple=False).flatten()
                if idx.numel():
                    abs_idx = idx[:self.k_tokens] + b * seq_len
                    scale = self.weight
                    # 正例（回答以 <answer>yes 开头）再乘 pos_weight，抵消数据 1:2 的先验
                    if self.pos_weight != 1.0 and self._yes_ids:
                        head_ids = flat_2d[b, idx[:self.k_tokens]].tolist()
                        if any(i in self._yes_ids for i in head_ids):
                            scale *= self.pos_weight
                    w[abs_idx] = scale

        weighted = token_loss * w * valid
        denom = (w * valid).sum().clamp_min(1.0)
        return weighted.sum() / denom


loss_map['answer_weighted'] = AnswerWeightedLoss
