# 绿风润耕 · 智慧大棚总控 V4.2.1

> 面向设施农业的**环境与能源协同分布式边缘智能控制系统**
>  Transformer 温度预测 + SAC 上层策略 + 约束控制与安全裁决 + 光伏/风电/储能协同

单棚闭环演示系统。当前版本中，Transformer 预测与 SAC 策略**已进入实际运行链路**，不是离线展示。

---

## 一、系统做了什么

一条完整的控制闭环：

```text
虚拟大棚 → 统一环境/能源状态 → Transformer 预测 → SAC 上层策略
   ↑                                                    ↓
前端总控 ← 功率平衡与设备实际效果 ← 加热/通风口/保温帘/风机 ← 约束控制与安全裁决
```

**六个页面**

| 页面 | 内容 |
|---|---|
| 大棚实况 | 东/中/西棚温、湿度、CO₂、冠层温度、基质温度与含水率；总控曲线含实测棚温、10/30 分钟预测与目标上下限 |
| 控制中心 | 人工/AI 辅助/AI 自动三种模式；直接调节加热、通风口、保温帘、风机；内置保温升温、通风降温、排湿、设备归零预设 |
| AI 智能托管 | 把模型输出翻译为可读建议，展示建议动作与安全裁决结果，支持批准、修改、拒绝 |
| 能源协同 | 光伏/风电出力、市电功率、设备负荷、储能功率与 SOC；可注入正常供电、限功率、停电场景 |
| 实验与报告 | 控制周期、人工命令、AI 执行、安全裁决次数与温度趋势；导出交互追溯CSV |
| 系统与投运 | 系统连接关系、模型与状态文件位置、从虚拟大棚切换到真实大棚的接口边界 |

**三种控制模式**

| 模式 | 行为 |
|---|---|
| 人工控制 | 人工设定四类设备目标，系统仍执行安全裁决 |
| AI 建议 + 人工确认 | 模型形成建议，仿真暂停等待批准 / 修改 / 拒绝 |
| AI 自动托管 | AI 每周期自动执行，仍必须经过约束控制与安全裁决 |

人工按钮不是只改前端数字 —— 命令写入 `var/greenhouse_ui/commands`，统一运行层读取后作用到虚拟设备，虚拟大棚再按设备动作计算下一周期环境状态。

---

## 二、快速开始

### 环境要求

- Windows 10/11
- Python 3.10+
- 依赖：`torch`、`numpy`

```bash
conda create -n greenwind_ai_env python=3.10 -y
conda activate greenwind_ai_env
pip install torch numpy
```

### 启动

双击项目根目录的 `启动绿风润耕V4.bat`。

脚本会自动探测 Python 环境。若未找到，可手动指定：

```bat
set GREENWIND_PYTHON=D:\path\to\envs\greenwind_ai_env\pythonw.exe
```

进入软件后点击右上角 **「启动虚拟大棚」**。首次加载 Transformer 与 SAC 约需十几秒，状态变为 `RUNNING` 后即可操作。

启动失败时，双击 `验证绿风润耕整合版.bat`，保留错误窗口并按提示排查。

---

## 三、技术要点

### 3.1 热惯性：多热容网络

棚内热过程不是"设备开关直接改温度"。东、中、西三个空气分区、作物冠层、栽培基质与棚架/管路蓄热体各自保存独立温度，按能量平衡迭代：

```text
C_i × dT_i/dt = ΣQ_in - ΣQ_out
```

加热、太阳得热、覆盖散热、通风换热、相邻分区换热，以及冠层/基质/棚架的蓄放热，共同决定下一时刻温度。

`热惯性阶跃响应.png` 是加热开启—关闭的阶跃实验曲线，可直观看到热惯性确实参与了仿真计算。

### 3.2 能源协同：可解释规则 EMS

光伏和风电先供棚内设备，多余清洁能源用于充电；负荷高于阈值或市电限功率/停电时，储能在 SOC 安全边界内放电。若三者仍无法满足负荷，能源系统会**降低可供功率，并把有效设备功率反馈给热环境模型** —— 而不是假定设备仍满功率运行。

电池功率为正表示放电，为负表示充电。

当前采用可解释规则 EMS 负责充放电；SAC V3 的动作维度保持不变，尚未直接输出电池功率。若要把 SOC、能源价格与电池寿命纳入 SAC，需建立新状态/动作/奖励函数并重新训练。

### 3.3 安全裁决

所有人工命令与 AI 命令都先经过范围检查、互斥检查和数据有效性检查，再进入虚拟大棚。裁决次数与原因记录在 `state.json` 的 `safety` 字段，可追溯。

---

## 四、运行证据

```text
var/greenhouse_ui/
```

| 文件 | 内容 |
|---|---|
| `state.json` | 前端读取的最新统一状态 |
| `interaction_trace.csv` | 每个控制周期的温度、预测、设备动作与安全裁决 |
| `commands/` | 尚未处理的前端命令 |
| `command_archive/` | 已处理命令，便于追溯 |

`interaction_trace.csv` 字段：

```text
timestamp, control_step, mode, temperature_c, humidity_pct,
pred_10min_c, pred_30min_c,
heater_safe, vent_safe, curtain_safe, fan_safe,
battery_soc_pct, battery_power_kw, renewable_power_kw,
grid_power_kw, load_power_kw, grid_status,
unserved_power_kw, safety_overrides, command_type
```

---

## 五、回归测试与实验

```bash
# 回归测试
python tools/virtual_greenhouse/test_thermal_inertia.py
python tools/virtual_greenhouse/test_energy_system.py
python tools/ai_pipeline/test_interactive_v4_contract.py

# 对照实验（输出 CSV / JSON 指标 / PNG 曲线）
python tools/virtual_greenhouse/thermal_inertia_experiment.py
python tools/virtual_greenhouse/energy_storage_experiment.py
```

输出位置：

- `tools/virtual_greenhouse/thermal_inertia_results/`
- `tools/virtual_greenhouse/energy_results/`

### 已通过的验证

| 项目 | 结果 |
|---|---|
| legacy_virtual_greenhouse_tests | 47 项 |
| packaged_regression_tests | 19 项 |
| thermal_inertia_regression_tests | 4 项 |
| thermal_inertia_step_response | 通过 |
| energy_system_regression_tests | 5 项 |
| energy_storage_comparison | 通过 |
| v4_contract_tests | 5 项 |
| python_static_compile | 通过 |

---

## 六、常用配置

`config/ui_demo_config.json`

| 字段 | 说明 |
|---|---|
| `default_mode` | `manual` / `assist` / `auto` |
| `wall_interval_seconds` | 每个控制周期在演示中等待的秒数 |
| `assist_timeout_seconds` | AI 辅助模式等待人工决定的秒数 |
| `base_outdoor_temp_c` | 仿真室外基础温度 |
| `weather_scenes` | 天气场景 |
| `max_control_steps` | 单次演示最大控制周期数 |

模型路径已配置完成，普通演示不需要修改 Python 文件。

---

## 七、诚实的边界声明

这一节请认真读，它定义了系统**能证明什么、不能证明什么**。

**本系统能证明**：预测、决策、约束、执行、反馈与人工复核在仿真环境中形成了完整闭环。

**本系统不能证明**：真实现场效果。热容与换热系数、电池、光伏与风机参数均为工程仿真初值，**未经真实大棚标定**。阶跃实验能证明热惯性机制确实参与了仿真计算，但不代表系数已现场辨识。

**真实接入需要**：

1. 新增传感器适配器与执行器适配器，保持同一字段与命令契约
2. 完成设备限值标定
3. 断网/故障测试
4. Transformer 微调（使用相同训练管线重新校准）
5. 现场分阶段试运行

**能源侧**需分别接入智能电表、光伏/风机逆变器、BMS 与 PCS；当前参数不构成实际储能容量选型或 ROI 结论。

**GNN 多棚协同**属于扩展方向，未在单棚版本中虚构节点关系。

---

## 八、项目结构

```text
绿风润耕_V4.2.1_能源按钮修复版/
├── 启动绿风润耕V4.bat          # 启动器（自动探测 Python 环境）
├── 验证绿风润耕整合版.bat       # 排障脚本
├── README_绿风润耕整合版.md     # 内部技术文档
├── config/
│   └── ui_demo_config.json      # 演示配置
├── tools/
│   ├── GreenWindConsoleV4.ps1   # WinForms 前端（702 行）
│   ├── ai_pipeline/
│   │   ├── interactive_runtime_v2.py   # 统一运行层（666 行）
│   │   ├── greenhouse_sac_env.py       # SAC 环境
│   │   ├── strategy_to_control_v3.py   # 策略到控制映射
│   │   ├── train_transformer_v2.py     # Transformer 训练
│   │   ├── train_sac_thermal_v4.py     # SAC 训练
│   │   └── transformer_inference.py    # 推理
│   └── virtual_greenhouse/
│       ├── greenhouse.py               # 虚拟大棚（热惯性模型）
│       ├── controllers.py              # 约束控制与安全裁决
│       ├── energy_system.py            # 能源与储能 EMS
│       ├── actuators.py / sensors.py   # 执行器 / 传感器
│       └── *_experiment.py             # 对照实验
└── var/greenhouse_ui/          # 运行证据（自动生成）
```

---

## 九、模型资产

| 文件 | 说明 |
|---|---|
| `tools/ai_pipeline/outputs_thermal_v3/transformer_v2.pt` | Transformer 温度预测权重 |
| `tools/ai_pipeline/sac_runs/thermal_v4/sac_thermal_v4_best.zip` | SAC V4 策略权重 |
| `tools/ai_pipeline/sac_runs/thermal_v4/training_metrics.csv` | SAC 训练曲线数据 |
| `tools/ai_pipeline/sac_runs/thermal_v4/training_curve.png` | SAC 训练曲线 |

训练数据集（约 35 MB）未纳入仓库，可重新生成：

```bash
python tools/ai_pipeline/generate_data_v2.py
```

---

## 十、许可与边界

本仓库代码仅用于赛事评审与技术交流。仿真参数为工程初值，不构成现场部署结论。