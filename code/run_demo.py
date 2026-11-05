# -*- coding: utf-8 -*-
"""
run_demo.py · 一键跑通:冻结规格 → 回测 → 随机对照 → 判决

数据是合成的几何随机游走(无漂移)——市场里没有可赚的 edge,这是 demo 的设定。
本脚本只演示流程与判据口径,不构成任何市场结论。

运行:python run_demo.py      依赖:numpy
"""
import numpy as np

from backtest_core import Cost, Spec, metrics, run_strategy
from random_baseline import percentile_of, random_baseline

# ---- PREREG(跑前冻结,跑完不改)--------------------------------
SEED = 7
N_BARS = 6000
SPEC = Spec(cci_n=10, enter_below=-100.0, exit_above=100.0, stop_pct=0.02)
COST = Cost(commission_bps=2.0, slippage_ticks=2.0, tick_size=0.01, multiplier=1.0)
N_SIMS = 300
# ----------------------------------------------------------------


def make_data(seed: int = SEED, n: int = N_BARS):
    """几何随机游走,OHLC 白盒自检(open 落在 [low, high] 内)。"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.010, n)
    close = 100.0 * np.exp(np.cumsum(r))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.002, n - 1))])
    open_ = np.clip(open_, np.minimum(low, close), np.maximum(high, close))
    assert np.all(open_ >= low) and np.all(open_ <= high), "OHLC 自检失败"
    assert np.all(high >= low)
    return open_, high, low, close


def main():
    print("=" * 64)
    print("PREREG(跑前冻结)")
    print(f"  数据: 合成随机游走 {N_BARS} 根日线, seed={SEED}")
    print(f"  策略: CCI({SPEC.cci_n}) 上穿 {SPEC.enter_below:+.0f} 做多, "
          f"下穿 {SPEC.exit_above:+.0f} 平, 止损 {SPEC.stop_pct:.0%}")
    print(f"  成本: 手续费 {COST.commission_bps:.0f}bp/边 + 滑点 "
          f"{COST.slippage_ticks:.0f} 跳/边")
    print("=" * 64)

    o, h, l, c = make_data()
    trades = run_strategy(o, h, l, c, SPEC, COST)
    m = metrics(trades)
    if m["n"] == 0:
        print("[X] 零成交,demo 无法演示")
        return

    holds = [t["exit_i"] - t["entry_i"] for t in trades]
    hold = int(np.median(holds))
    rb = random_baseline(o, h, l, c, n_trades=m["n"], hold=hold,
                         stop_pct=SPEC.stop_pct, cost=COST,
                         n_sims=N_SIMS, seed=SEED)
    pct = percentile_of(m["net_per_trade"], rb["mean"])

    print(f"\n策略回测({m['n']} 笔,中位持有 {hold} 根):")
    print(f"  0 成本毛利(死刑线)  : {m['gross_total']:+,.0f}")
    print(f"  净利合计             : {m['net_total']:+,.0f}")
    print(f"  每笔净盈亏           : {m['net_per_trade']:+,.3f}")
    print(f"  净利 t 值            : {m['t_stat']:+.2f}")
    print(f"  胜率                 : {m['win_rate']:.1%}")

    print(f"\n随机对照({N_SIMS} 次模拟,每次 {m['n']} 笔,持有 {hold} 根,同成本同止损):")
    print(f"  随机每笔净盈亏 P50   : {rb['p50']:+,.3f}")
    print(f"  随机每笔净盈亏 P95   : {rb['p95']:+,.3f}")
    print(f"  策略在随机分布的百分位: {pct:.1f}%")

    print("\n判决(口径与公开案例一致):")
    g1 = m["gross_total"] > 0
    g2 = m["t_stat"] >= 2.0
    g5 = pct > 50
    print(f"  [考场1 毛信号]   0 成本毛利 {'>' if g1 else '≤'} 0      → {'过' if g1 else '不过(死刑线)'}")
    print(f"  [考场2 净利 t]   |t| = {abs(m['t_stat']):.2f} {'≥' if g2 else '<'} 2      → {'过' if g2 else '不过'}")
    print(f"  [考场5 随机超额] 百分位 {pct:.0f}% {'>' if g5 else '≤'} 50%  → {'过' if g5 else '不过'}")
    verdict = "判死" if not (g1 and g2 and g5) else "过门(仅演示口径,实判还需考场 0/3/4/6)"
    print(f"\n  ⇒ {verdict}")
    print("  注: 数据是合成随机游走,demo 只演示流程;真实判读按预注册规格在真实数据上跑。")


if __name__ == "__main__":
    main()
