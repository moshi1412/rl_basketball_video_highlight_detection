#!/usr/bin/env python3
"""把训练日志 (logging.jsonl) 画成曲线图，方便离线查看/写报告。

用法:
    python3 scripts/06_plot_curves.py output/grpo_clevr_3b/v2-*/logging.jsonl \
        -o docs/grpo_clevr_reward_curve.png
"""
import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if 'reward' not in d and 'eval_reward' not in d:
            continue  # 跳过 trainer 末尾的状态汇总行
        gs = d.get('global_step/max_steps') or d.get('global_step')
        if gs is None:
            continue
        d['_step'] = int(str(gs).split('/')[0])
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('-o', '--output', default='docs/training_curve.png')
    args = ap.parse_args()

    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for path in args.logs:
        rows = load(path)
        tr = [r for r in rows if 'reward' in r and 'eval_reward' not in r]
        ev = [r for r in rows if 'eval_reward' in r]
        name = path.split('/')[-2][:28]
        x = [r['_step'] for r in tr]
        ax = axes[0][0]
        ax.plot(x, [r['reward'] for r in tr], label=f'{name} reward')
        ax.plot(x, [r.get('rewards/MathAccuracy/mean', 0) for r in tr], '--', label=f'{name} accuracy')
        ax.plot(x, [r.get('rewards/Format/mean', 0) for r in tr], ':', label=f'{name} format')
        axes[0][1].plot(x, [r.get('kl', 0) for r in tr], label=f'{name} kl')
        axes[1][0].plot(x, [r.get('completions/mean_length', 0) for r in tr], label=f'{name} len')
        axes[1][1].plot(x, [r.get('grad_norm', 0) for r in tr], label=f'{name} grad_norm')
        for r in ev:
            axes[0][0].scatter([r['_step']], [r['eval_reward']], marker='*', s=120,
                               label=f'{name} eval_reward')
    axes[0][0].set_title('reward / accuracy / format'); axes[0][0].set_ylim(0, 2)
    axes[0][1].set_title('KL')
    axes[1][0].set_title('completion length (token)')
    axes[1][1].set_title('grad norm')
    for a in axes.ravel():
        a.set_xlabel('step'); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print('saved', args.output)


if __name__ == '__main__':
    main()
