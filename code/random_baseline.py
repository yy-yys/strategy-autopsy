# -*- coding: utf-8 -*-
"""
random_baseline.py · 随机对照基线(本框架的护城河)

问题:策略赚到的钱,是不是"随便什么时候进场"也能赚到?
做法:扔掉信号,换成「同数量、同成本、同出场机制」的随机入场,跑 N 次,
     得到"随机入场平均每笔净盈亏"的分布,再看策略落在分布第几百分位。

机制匹配(诚实口径):
- 入场:均匀随机取 t,次根开盘成交(与策略同)
- 出场:持有 K 根后开盘出场,K = 策略中位持有根数(可变持有的近似)
- 止损:与策略同一比例,持有窗口内触发按止损价成交
- 成本:与策略完全同一套 Cost

依赖:numpy。可独立复用:
    from random_baseline import random_baseline, percentile_of
"""
import numpy as np


def random_baseline(o, h, l, c, *, n_trades: int, hold: int, stop_pct: float,
                    cost, n_sims: int = 300, seed: int = 7) -> dict:
    """返回随机入场「平均每笔净盈亏」的分布与分位数。"""
    rng = np.random.default_rng(seed)
    T = len(c)
    if T < hold + 10:
        raise ValueError("序列太短,放不下一个持有窗")
    starts = rng.integers(5, T - hold - 2, size=(n_sims, n_trades))
    entry_i = starts + 1
    exit_i = entry_i + hold
    ep = o[entry_i]
    xp = o[exit_i]
    stop_px = ep * (1.0 - stop_pct)
    lows = np.lib.stride_tricks.sliding_window_view(l, hold + 1)[starts].min(axis=2)
    hit = lows <= stop_px
    xp = np.where(hit, stop_px, xp)
    fee = (ep + xp) * cost.commission_bps * 1e-4 + 2 * cost.slippage_ticks * cost.tick_size
    net = (xp - ep) - fee
    per_trade_mean = net.mean(axis=1)
    return dict(
        n_sims=n_sims, n_trades=n_trades, hold=hold,
        mean=per_trade_mean,
        p50=float(np.percentile(per_trade_mean, 50)),
        p95=float(np.percentile(per_trade_mean, 95)),
    )


def percentile_of(value: float, dist: np.ndarray) -> float:
    """策略值在随机分布中的百分位(0–100)。"""
    return float((dist < value).mean() * 100.0)
