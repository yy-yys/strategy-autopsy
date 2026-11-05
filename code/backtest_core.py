# -*- coding: utf-8 -*-
"""
backtest_core.py · 精简演示版验证内核

为公开 repo 重写的最小可读版本,只保留「预注册 → 回测 → 随机对照 → 判决」
流程需要的骨架。与任何内部生产系统无关(独立实现,零拷贝)。

约定:
- 价格为 numpy 数组,单合约演示口径:open/high/low/close
- 成交假设:信号在 t 根收盘确认,t+1 根开盘价成交(次根开盘,可过夜)
- 成本:单边 = 手续费(bps) + 滑点(跳数 × 最小变动价位)
- 止损:触发按劣价成交,跳空穿越按开盘价(对策略更不利的那一个)

依赖:numpy
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Cost:
    """成本模型。全部按单边计,往返 = 2 × 单边。"""
    commission_bps: float = 2.0    # 单边手续费(万分之)
    slippage_ticks: float = 2.0    # 单边滑点(跳)
    tick_size: float = 0.01        # 最小变动价位
    multiplier: float = 1.0        # 合约乘数(演示取 1)

    def per_side(self, price: float) -> float:
        return price * self.commission_bps * 1e-4 + self.slippage_ticks * self.tick_size


@dataclass
class Spec:
    """预注册规格:跑前冻结,跑完不改。"""
    cci_n: int = 10
    enter_below: float = -100.0    # CCI 上穿该值 → 做多
    exit_above: float = 100.0      # CCI 下穿该值 → 平仓
    stop_pct: float = 0.02         # 固定比例止损


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 10) -> np.ndarray:
    """Commodity Channel Index。前 n-1 根为 NaN。"""
    tp = (high + low + close) / 3.0
    win = np.lib.stride_tricks.sliding_window_view(tp, n)
    sma = win.mean(axis=1)
    md = np.abs(win - sma[:, None]).mean(axis=1)
    out = np.full(tp.shape, np.nan)
    out[n - 1:] = (tp[n - 1:] - sma) / (0.015 * md)
    return out


def run_strategy(o, h, l, c, spec: Spec, cost: Cost):
    """CCI 均值回归演示策略(只做多)。

    返回逐笔列表,每笔含 entry_i/exit_i/entry_px/exit_px/gross/net
    (gross = 0 成本毛差;net = 扣双边成本后)。
    """
    ind = cci(h, l, c, spec.cci_n)
    trades = []
    T = len(c)
    i = spec.cci_n + 1
    pos = 0
    entry_i = exit_i = 0
    entry_px = fee_in = 0.0
    while i < T - 1:
        if pos == 0:
            if ind[i - 1] <= spec.enter_below and ind[i] > spec.enter_below:
                entry_i = i + 1
                entry_px = o[entry_i]
                fee_in = cost.per_side(entry_px)
                pos = 1
        else:
            stop_px = entry_px * (1.0 - spec.stop_pct)
            exit_px = None
            if o[i] <= stop_px:                      # 跳空穿越:按开盘劣价
                exit_px, exit_i = o[i], i
            elif l[i] <= stop_px:                    # 盘中触发:按止损价
                exit_px, exit_i = stop_px, i
            elif ind[i - 1] >= spec.exit_above and ind[i] < spec.exit_above:
                exit_px, exit_i = o[i + 1], i + 1    # 次根开盘
            if exit_px is not None:
                fee_out = cost.per_side(exit_px)
                trades.append(dict(
                    entry_i=entry_i, exit_i=exit_i,
                    entry_px=float(entry_px), exit_px=float(exit_px),
                    gross=float(exit_px - entry_px),
                    net=float(exit_px - entry_px - fee_in - fee_out),
                ))
                pos = 0
                i = exit_i
        i += 1
    return trades


def metrics(trades) -> dict:
    """交易级指标。gross_total 即「0 成本毛利」(死刑线判据)。"""
    n = len(trades)
    if n == 0:
        return dict(n=0)
    net = np.array([t["net"] for t in trades])
    gross = np.array([t["gross"] for t in trades])
    sd = net.std(ddof=1) if n > 1 else 0.0
    t_stat = float(net.mean() / sd * np.sqrt(n)) if sd > 0 else float("nan")
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))
    mdd = float((eq - peak[1:]).min()) if len(eq) else 0.0
    return dict(
        n=n,
        win_rate=float((net > 0).mean()),
        gross_total=float(gross.sum()),
        net_total=float(net.sum()),
        net_per_trade=float(net.mean()),
        t_stat=t_stat,
        mdd_per_trade=mdd,
    )
