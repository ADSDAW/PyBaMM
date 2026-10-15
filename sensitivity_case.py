# -*- coding: utf-8 -*-
"""
PyBaMM 参数灵敏度分析案例
==========================

场景: 硅碳(Si-C) + 石墨复配负极 匹配 三元(NCM811)正极
模型: DFN, 负极两相颗粒 (石墨=primary, 硅=secondary), 含 SEI 生长
参数集: Chen2020_composite

研究三个参数对多个输出变量的影响:
  1. 硅体积分数   "Secondary: Negative electrode active material volume fraction"
  2. 硅扩散系数   "Secondary: Negative particle diffusivity [m2.s-1]"
  3. SEI 生长速率 "Secondary: SEI kinetic rate constant [m.s-1]"

回答: 灵敏度不只针对电压! 每个输出变量(容量/浓度/SEI厚度...)都有自己的灵敏度。
"""
import numpy as np
import pybamm

# ============================================================
# 1. 建模: DFN + 复配负极两相 + SEI
# ============================================================
model = pybamm.lithium_ion.DFN(
    options={
        "particle phases": ("2", "1"),          # 负极2相(石墨+硅), 正极1相
        "SEI": "ec reaction limited",           # 开启 SEI 生长模型
    }
)

# ============================================================
# 2. 参数: 用 Chen2020_composite, 把 3 个待研究参数标成 [input]
# ============================================================
param = pybamm.ParameterValues("Chen2020_composite")

# 输入参数的基准值(实际就是参数集里的原值, 传参时用它们)
p_nominal = {
    "Secondary: Negative electrode active material volume fraction": 0.015,
    "Secondary: Negative particle diffusivity [m2.s-1]": 1.67e-14,
    "Secondary: SEI kinetic rate constant [m.s-1]": 1e-12,
}
for name in p_nominal:
    param[name] = "[input]"   # 告诉 pybamm: 这个参数求解时才传入

# ============================================================
# 3. 求解: IDAKLU 求解器 + 开启灵敏度计算
# ============================================================
solver = pybamm.IDAKLUSolver()
sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

solution = sim.solve(
    [0, 3600],                 # 模拟 1 小时放电
    inputs=p_nominal,          # 传入参数值 (不是电压!)
    calculate_sensitivities=True,
)


# ============================================================
# 4. 后处理: 计算相对灵敏度  |dln(变量)/dln(参数)|
#    含义: 参数变化 1% 时, 该变量大约变化百分之几
# ============================================================
def relative_sensitivity(solution, variable, inputs):
    """返回 {参数名: 相对灵敏度数组(随时间)}, 仅限输出变量为正、不为0的时刻。"""
    var = np.asarray(solution[variable].entries).flatten()
    rel = {}
    for name, p_val in inputs.items():
        sens = np.asarray(solution[variable].sensitivities[name])
        if sens.ndim > 1:
            sens = sens[:, 0]
        sens = sens.flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = sens * p_val / var          # dvar/dp * p / var
        r[~np.isfinite(r)] = np.nan         # 起点 var=0 处无定义, 置为 NaN
        rel[name] = r
    return rel


# 要考察的多个输出变量(注意: 不只是电压!)
variables = [
    "Terminal voltage [V]",
    "Discharge capacity [A.h]",
    "X-averaged negative secondary particle surface concentration [mol.m-3]",
    "X-averaged negative secondary SEI thickness [m]",
]

print("=" * 78)
print("各输出变量的相对灵敏度 |dln(变量)/dln(参数)|  (参数变化1%, 变量变化的百分比)")
print("=" * 78)

end_time_index = -1   # 用放电结束时刻来比较
for var_name in variables:
    rel = relative_sensitivity(solution, var_name, p_nominal)
    print(f"\n【{var_name}】")
    values = {}
    for p_name in p_nominal:
        v = rel[p_name][end_time_index]
        values[p_name] = v
        print(f"   {p_name[:52]:<55} {v: .4e}")
    # 排序: 哪个参数对该变量影响最大
    ranked = sorted(values, key=lambda k: abs(values[k]) if np.isfinite(values[k]) else -1, reverse=True)
    print(f"   → 影响排序: {' > '.join(r.split(':')[1].split('[')[0].strip() for r in ranked)}")

# ============================================================
# 5. 补充: 看绝对灵敏度(带量纲)的形状, 理解它是随时间变化的曲线
# ============================================================
print("\n" + "=" * 78)
print("绝对灵敏度 d(变量)/d(参数) 示例: 硅扩散系数 对 电压 的灵敏度")
print("=" * 78)
t = solution["Time [s]"].entries
sens = solution["Terminal voltage [V]"].sensitivities[
    "Secondary: Negative particle diffusivity [m2.s-1]"
].flatten()
for i in [0, len(t) // 2, -1]:
    print(f"   t = {t[i]:7.1f} s :  dV/dD_Si = {sens[i]: .6e}  V/(m2.s-1)")
print("\n(说明: 绝对灵敏度带量纲, 不同参数间不能直接比大小; 跨参数比较用相对灵敏度)")
