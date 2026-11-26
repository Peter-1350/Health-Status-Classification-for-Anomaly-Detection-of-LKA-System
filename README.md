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
| 维度与缺失 | 521 维 → 筛选后 **274** 维；274 维中 58 维存在缺失（最多 995/2409 帧，来自 short-window 数组统计与单行窗口的差分特征） | 数值列 49×(mean/std/min/max/range/last−first/mean\|Δ\|/std(Δ))；bool 12×True 占比；字符串 5 个取窗内众数后 one-hot（类别表只在 Train 上确定）；8 个数组（17/33 维）先算逐行 mean/std/min/max/first/last 再对窗内取 mean/std |
| 筛选 | 常量 73、近常量 21、\|r\|>0.95 共线 153 | 规则只在 Train 上拟合；Test 独有的 `front_torque_*` 因 Train 全 NaN 被自然剔除 |
| 划分 | `GroupKFold(5)`，`groups=record_id` | 帧高度自相关（一阶自相关中位 0.979），不能按帧随机划分 |
| 缺失值 | 中位数填充（在 pipeline 内，随模型保存） | 274 维中 58 维含缺失，最大 995/2409 帧 |
| 类别不平衡 | `class_weight='balanced'` / `scale_pos_weight` | 训练集异常帧 1073/2409，未用 SMOTE |
| 阈值 | OOF 记录概率上，约束"异常召回 ≥0.9 下最大化 Macro-F1"；同时另报 argmax-F1 阈值 | 阈值只在训练集 OOF 上确定，测试集仅做评估 |
| 模型选择 | OOF **记录级 PR-AUC**（与阈值无关）最高者 | 避免用测试集选模型；四个模型结果全部输出 |

## 结果快照（Test，40 条可用记录，7 条异常）

| 模型 | t | Acc | Macro-F1 | 异常 Recall | 异常 Prec | PR-AUC | ROC-AUC | F1最优t / Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| logreg （best） | 0.14 | 0.600 | 0.560 | 0.857 | 0.286 | 0.546 | 0.835 | 0.44 / 0.664 |
| xgb | 0.05 | 0.600 | 0.573 | 1.000 | 0.304 | 1.000 | 1.000 | 0.49 / 1.000 |
| rf | 0.24 | 0.400 | 0.398 | 1.000 | 0.226 | 0.851 | 0.909 | 0.55 / 0.842 |
| tree | 0.47 | 0.725 | 0.620 | 0.571 | 0.333 | 0.319 | 0.684 | 0.47 / 0.620 |

训练集 OOF（172 条记录、75 条异常）：logreg Acc 0.744 / Macro-F1 0.744 / PR-AUC 0.878；xgb 0.733/0.732/0.877；
rf 0.674/0.670/0.844；tree 0.791/0.787/0.748。

关键特征（logreg 系数、RF/XGB 重要性一致）：`op_right_roadedge__rowfirst__mean`、`acc_enable__frac`、
`e2e_position_y__rowfirst__mean`、`op_lane_left_prob__min/mean`、`op_state_steer_torque__stddiff`。

## 已知限制

- 测试集只有 7 条异常记录，记录级指标粒度极粗（1 条记录 = 14% recall），模型间排序不稳定：xgb 在测试集上 PR-AUC=1.0 属小样本偶然，不能据此改选模型。
- 召回约束阈值（≥0.9）会把概率阈值压到 0.05–0.24，精度明显下降，这是刻意选择的"宁可抓错"运行点；`f1opt_*` 列给出等精度视角的对照。
- 概率未校准（等渗回归留到下一阶段）、无推理延迟统计、无图像分支。
- 缓存不含原始长表；要改特征定义直接 `build_dataset.py --force` 重跑（~2.5 min）。
