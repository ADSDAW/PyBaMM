# -*- coding: utf-8 -*-
"""
PyBaMM 参数灵敏度分析案例（扩展版）
====================================

场景 : 硅碳(Si-C) + 石墨复配负极 匹配 三元(NCM811)正极
模型 : DFN, 负极两相颗粒 (石墨 = primary, 硅 = secondary), 含 SEI 生长
参数 : Chen2020_composite

本脚本输出四部分：
  A. 多个变量的灵敏度随时间演化图        -> sensitivity_evolution.png
  B. 参数可辨识性排序表（含参数可区分性）
  C. 有量纲 vs 无量纲 灵敏度对比          -> sensitivity_matrix.png
  D. 参数标定建议

三个待研究参数（标成 [input]，求解时传入数值）：
  1. 硅体积分数  Secondary: Negative electrode active material volume fraction
  2. 硅扩散系数  Secondary: Negative particle diffusivity [m2.s-1]
  3. SEI 生长率  Secondary: SEI kinetic rate constant [m.s-1]

说明：图中标签用英文，避免无中文字体环境出现方框；控制台输出为中文。
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无显示环境，只保存图片不弹窗
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pybamm


# ==================================================================
# 1. 建模：DFN + 复配负极两相 + SEI
# ==================================================================
model = pybamm.lithium_ion.DFN(
    options={
        "particle phases": ("2", "1"),      # 负极 2 相（石墨 + 硅），正极 1 相
        "SEI": "ec reaction limited",       # 开启 SEI 生长模型
    }
)

# ==================================================================
# 2. 参数：Chen2020_composite，把 3 个待研究参数标成 [input]
# ==================================================================
# (中文简称, PyBaMM 参数名, 名义值)
PARAMS = [
    ("硅体积分数", "Secondary: Negative electrode active material volume fraction", 0.015),
    ("硅扩散系数", "Secondary: Negative particle diffusivity [m2.s-1]", 1.67e-14),
    ("SEI生长率", "Secondary: SEI kinetic rate constant [m.s-1]", 1e-12),
]
NOMINAL = {name: val for _, name, val in PARAMS}
N_PARAM = len(PARAMS)

# (中文简称, PyBaMM 变量名, 图中用的英文简称)
VARS = [
    ("电压", "Terminal voltage [V]", "Voltage"),
    ("放电容量", "Discharge capacity [A.h]", "Capacity"),
    ("硅表面浓度", "X-averaged negative secondary particle surface concentration [mol.m-3]", "Si surface c"),
    ("硅SEI厚度", "X-averaged negative secondary SEI thickness [m]", "SEI thickness"),
]

param = pybamm.ParameterValues("Chen2020_composite")
for name in NOMINAL:
    param[name] = "[input]"     # 求解时才传入实际数值


# ==================================================================
# 3. 求解：IDAKLU + 灵敏度计算
# ==================================================================
solver = pybamm.IDAKLUSolver()
sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
solution = sim.solve([0, 3600], inputs=NOMINAL, calculate_sensitivities=True)
t = solution["Time [s]"].entries


# ==================================================================
# 工具函数：提取绝对/相对灵敏度
# ==================================================================
def extract_sensitivities(solution, variable, inputs):
    """返回 (变量值, 绝对灵敏度, 相对灵敏度)，后两者形状均为 (n_t, n_p)。

    绝对灵敏度  d(变量)/d(参数)        —— 有量纲，不同参数之间不可比
    相对灵敏度  dln(变量)/dln(参数)    —— 无量纲，可跨参数、跨变量比较
    """
    var = np.asarray(solution[variable].entries).flatten()

    cols = []
    for p_name in inputs:
        sens = np.asarray(solution[variable].sensitivities[p_name])
        if sens.ndim > 1:
            sens = sens[:, 0]
        cols.append(sens.flatten())
    S_abs = np.column_stack(cols)                       # (n_t, n_p)

    with np.errstate(divide="ignore", invalid="ignore"):
        S_rel = S_abs * np.array(list(inputs.values())) / var[:, None]
    S_rel[~np.isfinite(S_rel)] = np.nan                 # 变量为 0 的时刻无定义
    return var, S_abs, S_rel


# 一次性算好，后面各部分复用
DATA = {}
for vlabel, vname, vshort in VARS:
    DATA[vname] = extract_sensitivities(solution, vname, NOMINAL)


# ==================================================================
# A. 灵敏度随时间演化图
# ==================================================================
print("=" * 72)
print(" A. 灵敏度随时间演化图  ->  sensitivity_evolution.png")
print("=" * 72)

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for ax, (vlabel, vname, vshort) in zip(axes.flat, VARS):
    S_rel = DATA[vname][2]
    for j, (plabel, _, _) in enumerate(PARAMS):
        ax.plot(t, S_rel[:, j], lw=2, label=plabel)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_title(f"{vlabel}  ({vshort})", fontsize=11)
    ax.set_ylabel("dln(y) / dln(p)", fontsize=10)
    ax.set_yscale("symlog", linthresh=1e-3)   # 数值跨多个量级且有正负，用对称对数轴
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

for ax in axes[-1, :]:
    ax.set_xlabel("Time [s]", fontsize=10)

fig.suptitle(
    "Dimensionless relative sensitivities  dln(y)/dln(p)  vs time",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("sensitivity_evolution.png", dpi=150, bbox_inches="tight")
print("   已保存: sensitivity_evolution.png")
print("   读图:  曲线越高(绝对值越大) = 该参数对该变量影响越大;")
print("          正值 = 参数变大变量变大, 负值 = 参数变大变量变小。\n")


# ==================================================================
# B. 参数可辨识性排序表
# ==================================================================
print("=" * 72)
print(" B. 参数可辨识性排序表")
print("    指标 = 相对灵敏度的 L2 范数(RMS, 对时间取均方根)，无量纲")
print("    越大 = 该参数变化时该变量变化越明显 = 越容易从数据中辨识")
print("=" * 72)

for vlabel, vname, vshort in VARS:
    S_rel = DATA[vname][2]
    rms = np.sqrt(np.nanmean(S_rel**2, axis=0))     # (n_p,)
    peak = np.nanmax(np.abs(S_rel), axis=0)
    order = np.argsort(rms)[::-1]

    print(f"\n【{vlabel}】  ({vname})")
    print(f"   {'排名':<4}{'参数':<12}{'L2范数(RMS)':<16}{'最大值':<14}{'方向':<8}")
    print("   " + "-" * 54)
    for rank, j in enumerate(order, start=1):
        sign = "+" if np.nansum(S_rel[:, j]) >= 0 else "-"
        print(f"   {rank:<6}{PARAMS[j][0]:<14}{rms[j]:<16.3e}{peak[j]:<14.3e}{sign:<8}")

# --- B2. 参数可区分性（对电压）-------------------------------------
# 两个参数的灵敏度曲线若高度相关(共线)，就无法从同一组数据里分别辨识
print("\n" + "-" * 72)
print(" B2. 参数可区分性（以电压为观测量）")
print("     = 各参数灵敏度曲线之间的相关系数 r")
print("     |r| > 0.95 表示两条曲线形状雷同, 需要增加不同工况才能分开")
print("-" * 72)

S_volt = np.nan_to_num(DATA["Terminal voltage [V]"][2], nan=0.0)
corr = np.corrcoef(S_volt.T)                        # (n_p, n_p)

print("\n   " + "".join(f"{p[0]:<14}" for p in PARAMS))
for i, (plabel, _, _) in enumerate(PARAMS):
    row = "".join(f"{corr[i, j]:<14.3f}" for j in range(N_PARAM))
    print(f"   {plabel:<6}{row}")

flagged = False
for i in range(N_PARAM):
    for j in range(i + 1, N_PARAM):
        if abs(corr[i, j]) > 0.95:
            flagged = True
            print(f"   ⚠ {PARAMS[i][0]} 与 {PARAMS[j][0]} 高度共线 (|r|={abs(corr[i, j]):.3f}), "
                  f"仅凭电压曲线无法分别辨识")
if not flagged:
    print("   ✓ 三个参数的灵敏度曲线线性独立，理论上可从电压数据中分别辨识")


# ==================================================================
# C. 有量纲 vs 无量纲 对比 + 无量纲对比矩阵
# ==================================================================
print("\n" + "=" * 72)
print(" C. 有量纲 vs 无量纲 灵敏度对比（放电结束时刻 t = %.0f s）" % t[-1])
print("=" * 72)

for vlabel, vname, vshort in VARS:
    _, S_abs, S_rel = DATA[vname]
    print(f"\n【{vlabel}】")
    print(f"   {'参数':<12}{'绝对灵敏度 d y/d p':<26}{'相对灵敏度 dlny/dlnp':<22}")
    print("   " + "-" * 60)
    for j, (plabel, _, _) in enumerate(PARAMS):
        print(f"   {plabel:<14}{S_abs[-1, j]:<26.4e}{S_rel[-1, j]:<22.4e}")
    print("   注: 绝对灵敏度带量纲、跨参数不可比; 相对灵敏度无量纲、可直接比较")

# --- C2. 无量纲灵敏度对比矩阵 -------------------------------------
print("\n" + "-" * 72)
print(" C2. 无量纲灵敏度对比矩阵（L2 范数）  ->  sensitivity_matrix.png")
print("-" * 72)

matrix = np.zeros((N_PARAM, len(VARS)))              # 行 = 参数, 列 = 变量
for j, (vlabel, vname, vshort) in enumerate(VARS):
    matrix[:, j] = np.sqrt(np.nanmean(DATA[vname][2] ** 2, axis=0))

print("\n   " + "".join(f"{v[0]:<16}" for v in VARS))
for i, (plabel, _, _) in enumerate(PARAMS):
    row = "".join(f"{matrix[i, j]:<16.3e}" for j in range(len(VARS)))
    print(f"   {plabel:<8}{row}")

# 绘制热图（数值跨量级大，颜色用对数标度）
floor = 1e-5
plot_data = np.where(matrix > 0, matrix, floor)

fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(plot_data, norm=LogNorm(vmin=floor, vmax=matrix.max()),
               cmap="YlOrRd", aspect="auto")

ax.set_xticks(range(len(VARS)))
ax.set_xticklabels([f"{v[0]}\n{v[2]}" for v in VARS], fontsize=9)
ax.set_yticks(range(N_PARAM))
ax.set_yticklabels([f"{p[0]}\n{p[1].split('[')[0].strip()}" for p in PARAMS], fontsize=9)

for i in range(N_PARAM):
    for j in range(len(VARS)):
        val = matrix[i, j]
        txt = "0\n(not sensitive)" if val == 0 else f"{val:.2e}"
        color = "black" if val < 1e-2 else "white"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

ax.set_title("Dimensionless sensitivity matrix   RMS |dln(y)/dln(p)|", fontsize=12)
fig.colorbar(im, ax=ax, label="dimensionless sensitivity (log scale)")
fig.tight_layout()
fig.savefig("sensitivity_matrix.png", dpi=150, bbox_inches="tight")
print("\n   已保存: sensitivity_matrix.png")
print("   读图:  颜色越深 = 该参数对该变量影响越大; 矩阵同一列内可直接比较不同参数。")


# ==================================================================
# D. 参数标定建议
# ==================================================================
print("\n" + "=" * 72)
print(" D. 参数标定建议")
print("=" * 72)
print(" 原则 1: 用对该参数最敏感的变量去标定该参数（观测信噪比最高）")
print(" 原则 2: 两个参数灵敏度曲线高度共线时，需增加不同工况（倍率/温度）才可分开")
print(" 原则 3: 灵敏度 ≈ 0 的参数无法用该变量标定，必须换观测量\n")

for i, (plabel, pname, val) in enumerate(PARAMS):
    col = matrix[i, :]
    best_j = int(np.argmax(col))
    best_val = col[best_j]
    if best_val < 1e-6:
        note = "所有考察变量都不敏感，需更换观测量或工况"
    else:
        note = f"建议用【{VARS[best_j][0]}】标定（L2 = {best_val:.2e}）"
    print(f"   {plabel:<10} (名义值 {val:.3e})  ->  {note}")

print("\n 注：以上结论基于 1C 恒流放电 1 小时工况；换用大倍率或低温工况，")
print("     扩散类参数的灵敏度通常会显著上升，可辨识性变好。")
