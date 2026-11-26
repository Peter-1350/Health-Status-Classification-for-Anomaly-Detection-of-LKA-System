# LKA 健康状态分类 · Stage 1（仅 CAN 信号）

端到端流程：原始 CAN CSV → 特征表（缓存）→ 特征筛选 → 4 个模型 GroupKFold 交叉验证 → 记录级预测与评估。
本阶段不使用图像数据、不做超参数搜索，预测/分类单位是 **record**。

## 运行

```bash
cd "/mnt/d/SCHOOL/7.2026-Autumn/机器学习与大数据/Final project 2026"
PY=/mnt/d/Coding/anaconda3/envs/pytorch_env/python.exe

$PY src/build_dataset.py    # 解析 + 特征 + 筛选 + 缓存（~2.5 min，--force 强制重建）
$PY src/train.py            # GroupKFold(5) CV + 阈值 + 全量重训（~15 s）
$PY src/predict.py          # 测试集记录级预测 + 指标 + 图（~4 s）
```

Windows 下直接 `conda activate pytorch_env` 后用 `python src/xxx.py` 同样可跑。
正式评估只用官方 Test 划分（41 条，含 1 条无 CAN 数据）。

## 目录

```
src/common.py         常量、模型配置、记录级聚合与指标、阈值选择
src/build_dataset.py  Step1-3：解析 → 帧级特征 → 特征筛选 → 缓存
src/train.py          Step4：GroupKFold(5) 按 record 分组 → OOF 阈值 → 保存模型
src/predict.py        Step5：帧概率 → 记录级概率 → 指标/图/表

data/processed/frames_{train,test}.parquet   每行 = 一个 CAN CSV（帧），含特征+标签
data/processed/feature_meta.json             列清单、筛选理由、每记录 LKA 与标签、类别表
models/{logreg,tree,rf,xgb}.joblib           含缺失填充(+标准化)的完整 pipeline
models/selection.json                        选定特征、各模型阈值、best
results/cv_summary.csv, cv_folds.csv         交叉验证（记录级）
results/test_summary.csv, test_metrics.json  测试集（记录级为主，帧级为辅）
results/predictions_test.csv                 每条记录的概率与预测
results/feature_importance.csv, figures/     best 模型 top-20 特征、混淆矩阵+PR 曲线
```

## 数据口径与决策记录

| 决策 | 取值 | 说明 |
|---|---|---|
| 一帧的定义 | 一个 `can_XXXX.csv`（300 行，3 s 滑动窗，窗移 0.5 s，相邻窗重叠） | 共 2890 个文件，其中 8 个是只有表头的空文件 |
| 标签（本阶段） | **record 口径**：记录内各帧 `mean(LKA_error)` 的均值 ≥ 0.25 判 Anomaly，记录内全部帧继承该标签 | 与 Proposal §1 一致；改成"铺平去重后全部行"的均值只在 4/212 条记录上翻转（record109/154/173/178），故该口径稳健 |
| 空记录 | `Train/record100`、`Test/record177` 无可用 CAN 数据 | 训练中剔除；预测表保留并置 NaN，评估按 40 条记录计 |
| 标签泄漏 | 丢弃 `LKA_error`、`op_left_laneline`、`op_right_laneline` | 三者满足 `LKA_error=(left+right)/2`，误差 <3e-16；另丢 `Time/unix_time/can_mono_time/wallTimeCentiseconds/easternTime` |
| 维度与缺失 | 521 维 → 筛选后 **270** 维；270 维中 4 维存在缺失（≤46/473 帧，单行 warmup 窗口的差分统计） | 数值列 49×(mean/std/min/max/range/last−first/mean\|Δ\|/std(Δ))；bool 12×True 占比；字符串 5 个取窗内众数后 one-hot（类别表只在 Train 上确定）；8 个数组（17/33 维）先算逐行 mean/std/min/max/first/last 再对窗内取 mean/std |
| 筛选 | 常量 73、近常量 25、缺失率>20% 2、\|r\|>0.95 共线 151 | 规则只在 Train 上拟合；Test 独有的 `front_torque_*` 因 Train 全 NaN 被自然剔除 |
| 划分 | `GroupKFold(5)`，`groups=record_id` | 帧高度自相关（一阶自相关中位 0.979），不能按帧随机划分 |
| 缺失值 | 中位数填充（在 pipeline 内，随模型保存） | 差分统计在单行窗口按"无变化"填 0；数组统计逐行 nan-aware；剩余少量缺失由中位数填充 |
| 类别不平衡 | `class_weight='balanced'` / `scale_pos_weight` | 训练集异常帧 1073/2409，未用 SMOTE |
| 阈值 | OOF 记录概率上，约束"异常召回 ≥0.9 下最大化 Macro-F1"；同时另报 argmax-F1 阈值 | 阈值只在训练集 OOF 上确定，测试集仅做评估 |
| 模型选择 | OOF **记录级 PR-AUC**（与阈值无关）最高者 | 避免用测试集选模型；四个模型结果全部输出 |

## 结果快照（Test，40 条可用记录，7 条异常）

| 模型 | 召回优先 t | Acc | Macro-F1 | 异常 Recall | 异常 Prec | PR-AUC | ROC-AUC | F1 最优 t / Acc / Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| **xgb（best）** | 0.01 | 0.425 | 0.422 | 1.000 | 0.233 | **0.928** | 0.978 | 0.63 / 0.950 / **0.902** |
| rf | 0.26 | 0.425 | 0.422 | 1.000 | 0.233 | 0.850 | 0.905 | 0.55 / 0.950 / 0.913 |
| tree | 0.47 | 0.750 | 0.642 | 0.571 | 0.364 | 0.306 | 0.684 | 0.47 / 0.750 / 0.642 |
| logreg | 0.12 | 0.625 | 0.594 | 1.000 | 0.318 | 0.492 | 0.831 | 0.56 / 0.775 / 0.664 |

帧级 PR-AUC：xgb 0.868、rf 0.857、logreg 0.756、tree 0.282。

训练集 OOF（172 条记录、75 条异常，记录级 PR-AUC）：xgb 0.876、logreg 0.858、rf 0.855、tree 0.729。

关键特征（xgb 重要性）：`energy_total_charge__max`、`op_right_roadedge__rowfirst__mean`、`vLead2__min`、
`aEgo__range`、`lead1_spacing__max`、`plan_accel__rowlast__mean`、`op_ll_right_prob__range`。

## 已知限制

- 测试集只有 7 条异常记录（1 条记录 = 14% recall），记录级指标波动极大，不能据此改选模型。
- **阈值会退化**：概率未校准时，"召回 ≥0.9" 会把 xgb 的阈值压到网格下界 0.01，等价于"全部判异常"（68% 记录被判正）。`train.py` 现在会对这种情况打印 `shortcut:` 警告；报告建议用 `f1opt_*` 那组（阈值 0.63）作主结论，召回优先点作为对照。
- 概率未校准（等渗回归留到下一阶段）、无推理延迟统计、无图像分支。
- 缓存不含原始长表；要改特征定义直接 `build_dataset.py --force` 重跑（~2.5 min）。

## 修复记录：xgb 测试集 PR-AUC 曾为 1.0

原因是两处**静默 NaN 注入**，不是模型问题：

1. 数组列里有字面 `nan` 填充值（例如 `op_laneline_t` 的第 33 个网格槽），逐行 `mean/std/min/max` 会把整窗统计变成 NaN（995/2409 帧）；
2. 每条记录第一个 CSV 常只有 1 行（buffer warmup），`diff()` 后 `mean|Δ|`、`std(Δ)` 全 NaN（397/2409 帧）。

这些 NaN 在 pipeline 里被"训练集中位数"填充，等于把部分记录的概率钉在中位数附近；恰好把 7 条测试异常都排到了 33 条正常之前，于是记录级 PR-AUC / 精度 / F1 全成了 1.0。

修复：数组统计改为 nan-aware（`nanmean/nanstd/nanmin/nanmax`），单行窗口的差分统计填 0（无可观测变化），并新增"缺失率 >20% 直接删列"规则（删掉 2 个 `op_laneline_t` 垃圾特征）。
修复后 xgb 记录级 PR-AUC 1.000 → 0.928，帧级 PR-AUC 0.963 → 0.868。
