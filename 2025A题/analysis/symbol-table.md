# 2025 年 A 题统一符号表

本表不用 LaTeX 公式语法；所有符号均按普通文本显示，便于直接在任意 Markdown 阅读器中查看。

## 1. 索引、集合与常数

| 中文含义 | 记号或名称 | 建议代码字段 | 单位/类型 | 取值或定义 |
|---|---|---|---|---|
| 无人机索引 | i | uav_id | 枚举 | FY1–FY5 |
| 导弹索引 | j | missile_id | 枚举 | M1–M3 |
| 同一无人机的烟幕弹索引 | k | smoke_id | 整数 | 1–3 |
| 真目标实体集合 | T | target_cylinder | 三维集合 | x² + (y - 200)² ≤ 49，且 0 ≤ z ≤ 10 |
| 烟幕有效半径 | Rc | cloud_radius_m | m | 10 |
| 烟幕有效寿命 | Tc | cloud_lifetime_s | s | 20 |
| 云团下沉速度 | vc | cloud_sink_speed_mps | m/s | 3，竖直向下 |
| 导弹速度 | vm | missile_speed_mps | m/s | 300 |
| 目标半径 | RT | target_radius_m | m | 7 |
| 目标高度 | HT | target_height_m | m | 10 |
| 重力加速度 | g | gravity_mps2 | m/s² | 后续模型统一设定并记录 |
| 时间长度 | measure | interval_measure | s | 时间区间并集的长度 |

## 2. 初始位置与状态变量

| 中文含义 | 记号或名称 | 建议代码字段 | 单位/类型 | 定义 |
|---|---|---|---|---|
| 导弹初始位置 | missile_pos0[j] | missile_pos0_m | m，三维向量 | 题面给定 |
| 导弹在 t 时刻的位置 | missile_pos(j,t) | missile_pos_m | m，三维向量 | 沿初始点到原点的匀速直线 |
| 导弹抵达假目标时刻 | missile_arrival(j) | missile_arrival_s | s | 初始位置到原点的距离除以 300 |
| 无人机初始位置 | uav_pos0[i] | uav_pos0_m | m，三维向量 | 题面给定 |
| 无人机在 t 时刻的位置 | uav_pos(i,t) | uav_pos_m | m，三维向量 | 等高匀速直线 |
| 真目标几何中心 | target_center | target_center_m | m，三维向量 | (0, 200, 5) |
| 目标上的任意点 | target_point | target_point_m | m，三维向量 | 属于真目标圆柱体 |
| 烟幕弹投放点 | drop_pos(i,k) | drop_pos_m | m，三维向量 | 无人机 i 在投放时刻的位置 |
| 烟幕弹起爆点 | burst_pos(i,k) | burst_pos_m | m，三维向量 | 由投放点、速度、航向、延迟和重力决定 |
| 云团中心位置 | cloud_center(i,k,t) | cloud_center_m | m，三维向量 | 起爆后以 3 m/s 下沉 |

## 3. 决策变量与派生时间

| 中文含义 | 记号或名称 | 建议代码字段 | 单位/类型 | 取值或定义 |
|---|---|---|---|---|
| 无人机航向角 | heading[i] | heading_deg | 度 | 0°–360°；正 x 轴为 0°，逆时针为正 |
| 无人机飞行速度 | speed[i] | uav_speed_mps | m/s | 70–140 |
| 烟幕弹投放时刻 | drop_time[i,k] | drop_time_s | s | 不小于 0 |
| 引信延迟 | fuse_delay[i,k] | fuse_delay_s | s | 不小于 0 |
| 烟幕弹起爆时刻 | burst_time[i,k] | burst_time_s | s | 投放时刻加引信延迟 |
| 烟幕弹启用变量 | is_used[i,k] | is_used | 0/1 | 问题 5 中是否投放 |
| 主要贡献导弹 | primary_missile[i,k] | primary_target_id | 枚举 | 求解后按单弹贡献标注为 M1、M2 或 M3，不限制烟雾的物理作用 |

## 4. 遮蔽与目标函数

| 中文含义 | 记号或名称 | 建议代码字段 | 单位/类型 | 定义 |
|---|---|---|---|---|
| 单云团有效指示 | is_effective(i,k,j,t) | is_effective | 0/1 | 满足寿命、导弹时间窗和几何判据时为 1 |
| 单弹对单导弹有效时长 | duration(i,k,j) | duration_per_smoke_s | s | 该烟幕弹有效时间段的长度 |
| 多弹对导弹 j 的并集时长 | duration_union(j) | duration_union_s | s | 同一导弹所有有效时间段的并集长度 |
| 三导弹同时遮蔽时长 | duration_simultaneous | simultaneous_full_target_duration_s | s | 三枚导弹完整遮蔽状态在统一时间轴上的交集长度 |
| 单导弹时长之和（诊断量） | duration_sum_diagnostic | sum_of_separate_missile_durations_s | 导弹·s | 仅用于诊断，不作为第五问目标函数 |
| 多弹重叠时长 | duration_overlap | duration_overlap_s | s | 逐枚时长之和与并集时长的差异 |

## 5. 模板字段映射

| 模板字段 | 对应值 | 建议代码字段 | 单位 |
|---|---|---|---|
| 烟幕干扰弹投放点的 x 坐标 | drop_pos(i,k) 的 x 分量 | drop_x_m | m |
| 烟幕干扰弹投放点的 y 坐标 | drop_pos(i,k) 的 y 分量 | drop_y_m | m |
| 烟幕干扰弹投放点的 z 坐标 | drop_pos(i,k) 的 z 分量 | drop_z_m | m |
| 烟幕干扰弹起爆点的 x 坐标 | burst_pos(i,k) 的 x 分量 | burst_x_m | m |
| 烟幕干扰弹起爆点的 y 坐标 | burst_pos(i,k) 的 y 分量 | burst_y_m | m |
| 烟幕干扰弹起爆点的 z 坐标 | burst_pos(i,k) 的 z 分量 | burst_z_m | m |
| 有效干扰时长 | duration(i,k,j) | duration_per_smoke_s | s |

## 6. 固定编号与模板行

- result1.xlsx：3 条数据行，对应 FY1 的烟幕弹 1–3；三行航向和速度必须一致。
- result2.xlsx：3 条数据行，对应 FY1、FY2、FY3 各 1 枚；每行有独立航向和速度。
- result3.xlsx：15 个候选弹位，对应 FY1–FY5 每机烟幕弹 1–3；同一无人机的所有启用行必须使用相同航向和速度，未启用行不得填写虚构结果。
