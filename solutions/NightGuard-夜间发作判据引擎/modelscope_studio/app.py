# -*- coding: utf-8 -*-
"""
NightGuard 魔搭创空间 · 夜间发作判据引擎在线演示

真实调用 engine/ 判据引擎逐窗推理（不是播放预渲染结果）。
信号来自 sim/ 合成模拟器，固定种子可复现，不含任何患者数据。
"""

import os
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.events import group_windows
from engine.gate import NightGate
from engine.signals import wearable_signals
from sim.simulator import generate_night, windowize

# 中文字体：魔搭镜像不一定装了中文字体，图里统一用英文标签，避免方块
matplotlib.rcParams["axes.unicode_minus"] = False

LEVEL_COLOR = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "LOG": "#64748b"}


def run_night(seizure_type, n_seizures, hours, seed):
    """跑一整夜，返回 (图, 结论 markdown, 事件表)"""
    minutes = int(hours * 60)
    # 发作安排在夜间均匀分布，避开首尾 30 分钟（基线预热期）
    seizures = []
    n = int(n_seizures)
    for i in range(n):
        onset = int(30 + (minutes - 60) * (i + 1) / (n + 1))
        seizures.append({
            "onset_min": onset,
            "dur_s": 70 if i % 2 == 0 else 55,
            "intensity": 1.0 if i % 2 == 0 else 0.75,
            "type": seizure_type,
        })

    night = generate_night(seed=int(seed), minutes=minutes,
                           seizures=seizures, traps=True)
    wins = windowize(night)
    gate = NightGate(signal_fn=wearable_signals)

    ts, hrs, motions, risks = [], [], [], []
    rung_wins = []
    for w in wins:
        r = gate.step(w)
        ts.append(w["t"] / 3600.0)
        hrs.append(w["hr"])
        motions.append(w["motion"])
        risks.append(r.risk)
        if r.level in ("CRITICAL", "WARNING"):
            rung_wins.append({
                "t": w["t"], "risk": r.risk, "level": r.level,
                "driver": r.driver, "evidence": r.evidence,
            })

    # ---------- 事件级归并（两把尺子：事件级 + 响铃级）----------
    # gate 的 journal 是逐窗日志（start == end），一次发作会写成几十行。
    # 事件级必须先按 events.DEFAULT_MERGE_S=300s 归并，否则误报数会放大一个量级。
    evs = group_windows(rung_wins)

    true_spans = [(s["onset_min"] * 60, s["onset_min"] * 60 + s["dur_s"])
                  for s in seizures]
    TOL = 60.0        # 发作前后 60 秒内的响铃算命中

    caught = set()
    false_alarms = 0
    ev_rows = []
    for e in evs:
        matched = None
        for idx, (a, b) in enumerate(true_spans):
            if e.t1 >= a - TOL and e.t0 <= b + TOL:
                matched = idx
                break
        if matched is None:
            false_alarms += 1
            verdict = "误报"
        else:
            caught.add(matched)
            verdict = "真发作"
        w0 = e.windows[0] if e.windows else {}
        evid = w0.get("evidence")
        ev_rows.append([
            f"{int(e.t0) // 3600:d}:{int(e.t0) % 3600 // 60:02d}:{int(e.t0) % 60:02d}",
            f"{e.duration_s:.0f}s",
            e.top_level,
            verdict,
            w0.get("driver") or "",
            ", ".join(evid) if isinstance(evid, list) else (evid or ""),
        ])

    recall = f"{len(caught)}/{len(true_spans)}"

    # ---------- 画图 ----------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.4]})
    fig.patch.set_facecolor("#0b1220")
    for ax in axes:
        ax.set_facecolor("#111c2e")
        ax.tick_params(colors="#8ea2c4", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#22314d")
        ax.grid(color="#22314d", linewidth=0.5, alpha=0.6)

    # 真发作阴影
    for a, b in true_spans:
        for ax in axes:
            ax.axvspan(a / 3600.0, b / 3600.0, color="#f43f5e", alpha=0.28, lw=0)

    axes[0].plot(ts, hrs, color="#f87171", lw=1.0)
    axes[0].set_ylabel("Heart rate (bpm)", color="#e8eefb", fontsize=9)

    axes[1].plot(ts, motions, color="#60a5fa", lw=0.9)
    axes[1].set_ylabel("Motion", color="#e8eefb", fontsize=9)

    axes[2].plot(ts, risks, color="#fbbf24", lw=1.0)
    axes[2].set_ylabel("Risk", color="#e8eefb", fontsize=9)
    axes[2].set_xlabel("Hours into the night", color="#e8eefb", fontsize=9)
    axes[2].set_ylim(-0.05, 1.05)

    # 响铃事件起点
    for e in evs:
        axes[2].axvline(e.t0 / 3600.0,
                        color=LEVEL_COLOR.get(e.top_level, "#64748b"),
                        lw=1.0, alpha=0.9)

    axes[0].set_title(
        f"NightGuard engine on a synthetic night  ·  "
        f"{seizure_type} x{n_seizures}  ·  seed={int(seed)}  ·  "
        f"pink band = true seizure, vertical line = alarm",
        color="#e8eefb", fontsize=10)
    fig.tight_layout()

    # ---------- 结论 ----------
    note = ""
    if seizure_type == "NCSE":
        note = (
            "\n\n> ⚠️ **NCSE（隐匿发作）是我们在真实数据上抓不到的那一类。**\n"
            "> 这里是合成信号，血氧下降被显式写进了模拟器，所以看起来可检。\n"
            "> 真实临床数据（SeizeIT2）**没有血氧通道**，而心率线索不够用："
            "发作时心率升 1.0–1.83 倍，但背景有 **10.8% 的时间**也超过自身 1.5 倍。\n"
            "> **请不要把这一档的演示效果当成真实性能。**"
        )

    md = f"""### 本次运行结果（合成信号，引擎实时推理）

| 指标 | 值 |
|---|---|
| 发作检出 | **{recall}** |
| **误报（事件级）** | **{false_alarms}** 次 / {hours} 小时 |
| 响铃窗口数 | {len(rung_wins)} 个窗 |
| 事件总数 | {len(evs)}（按 300 秒间隔归并）|
| 推理窗口数 | {len(wins)}（8 秒窗 / 4 秒步进）|

> **两把尺子都要看**：上面 {false_alarms} 个误报**事件**对应 {len(rung_wins)} 个响铃**窗口**。
> 只报事件数会低估打扰程度，只报窗口数会高估误报次数 ——
> 真实数据上我们见过 9 个误报事件等于 2846 次响铃。

---

### 真实临床数据上的成绩（这个才是可引用的数字）

> **致命级发作（继发全身化 + 阵挛期，与 SUDEP 相关）23 次检出 23 次，
> 代价是误报 6.70 次/10 夜。** 两个数字必须一起看。

`6.70 次/10夜` = `0.105 次/小时`（致命级队列 242h / 38 夜 = 6.37 小时/夜）。
自定目标是 0.05 次/小时，**实际差 2.1 倍 —— 没达标**。

数据来源 SeizeIT2（佩戴式：胸贴 + 耳后贴片）与 CHB-MIT。
**上面这个 Demo 是合成信号，只用来看引擎怎么决策，不代表真实性能。**{note}

---

### 必须说清的局限

- **不是无穿戴**：全部结论基于佩戴式传感器，而赛题要求「完全无穿戴」
- **五路通道只实测 2 路**：心率、体动。呼吸由 ECG 反推（覆盖 72%），血氧和体温数据集无通道
- **抓不到隐匿发作**：信噪比问题，换传感器无用
- **样本量小**：致命级发作仅 23 次，认证口径 recall ≥99% 需 161 次
- **分级报警无静默档**：实测 29 次致命级发作有 26 次会落进静默档，所以 `SILENT_TIERS` 恒空
- 未经临床验证的算法原型，**不是医疗诊断设备**，不替代脑电图与医生诊疗

完整说明与 292 项测试：<https://github.com/qiqiiiii77/nightguard-seizure-monitor>
"""

    return fig, md, ev_rows


INTRO = """# NightGuard · 夜间发作判据引擎

「罕见·无界」黑客松 · 赛题5（Dravet 综合征夜间睡眠监护）· **算法层方案**

**这不是产品，是一个在真实临床数据上跑出来的算法验证结果，包含达不到的部分。**

下面的演示用**合成信号**驱动**真实的判据引擎**逐窗推理 —— 你调参数，引擎当场算，
不是播放预先渲染好的结果。合成信号不含任何患者数据（固定种子可复现）。

真实数据上的成绩和局限，在运行后的结论区里逐条写明。
"""


# theme 放 Blocks 构造里。gradio 5.x 的 launch() 不接受 theme（传进去 TypeError），
# 6.x 这里只是弃用警告。放构造里是对 5/6 两个大版本都能跑的写法，两版均已实测。
with gr.Blocks(title="NightGuard · 夜间发作判据引擎",
               theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=1):
            seizure_type = gr.Radio(
                ["GTC", "NCSE"], value="GTC", label="发作类型",
                info="GTC = 强直-阵挛（有抽搐）；NCSE = 非惊厥持续状态（隐匿，真实数据上抓不到）")
            n_seizures = gr.Slider(1, 5, value=2, step=1, label="发作次数")
            hours = gr.Slider(2, 10, value=6, step=1, label="夜长（小时）")
            seed = gr.Number(value=2026, label="随机种子", precision=0)
            btn = gr.Button("跑这一夜", variant="primary")
        with gr.Column(scale=3):
            plot = gr.Plot(label="整夜信号与引擎判定")

    result = gr.Markdown()
    table = gr.Dataframe(
        headers=["起始", "时长", "级别", "判定", "主驱动", "证据链"],
        label="响铃事件（事件级，按 300 秒归并）",
        wrap=True)

    gr.Markdown("""
---
### 静态仪表盘（夜视暗色主题，完整回放）

仓库 `demo/` 下有三个自包含 HTML，可本地双击打开：
`dashboard.html`（GTC）· `dashboard_ncse.html`（NCSE）· `validation.html`（检验报告）

### 许可
MIT。代码：<https://github.com/qiqiiiii77/nightguard-seizure-monitor>
""")

    btn.click(run_night,
              inputs=[seizure_type, n_seizures, hours, seed],
              outputs=[plot, result, table])
    demo.load(run_night,
              inputs=[seizure_type, n_seizures, hours, seed],
              outputs=[plot, result, table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))
