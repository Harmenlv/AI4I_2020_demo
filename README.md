---

# AI Intelligent Teaching Experiments for Smart Manufacturing & Smart Ocean
> 教材配套人工智能实训开源代码

**Authors:** Haijian Shao, Xing Deng
**Version:** 2.0
**Platform:** Google Colab / Local Python
**Framework:** TensorFlow 2.20.0 (Deep Learning) | Scikit-learn (Machine Learning)
**Hardware:** CPU-based · Zero GPU Required
**Level:** Undergraduate AI General Education

---

## 📌 项目简介 / Introduction

本项目是本科人工智能通识课程及相关专业教材的配套开源实训代码，面向**智能制造、智慧海洋、人工智能通识、环境科学与工程、海洋技术**等高校课程，提供可直接运行、可重复实验、可用于教学演示的 AI 案例。

This open-source repository provides reproducible AI teaching experiments for undergraduate courses in artificial intelligence, intelligent manufacturing, smart ocean, marine technology, environmental science, and related interdisciplinary programs.

项目重点不是追求复杂模型，而是帮助零基础学生理解人工智能模型从**数据输入到智能预测**的完整工作流：

```
Data → Preprocessing → Model → Training → Evaluation → Visualization → Application
```

深度学习案例基于 TensorFlow/Keras 实现，传统机器学习案例基于 Scikit-learn 实现；全部案例均可在普通 CPU 环境运行，**无需 GPU**。

---

## 📚 目录 / Table of Contents

- [Part 1 · 深度学习教学案例（TensorFlow/Keras）](#part-1--深度学习教学案例tensorflowkeras)
  - [案例 A：水质状态智能识别](#案例-a水质状态智能识别)
  - [案例 B：帆船船体阻力预测](#案例-b帆船船体阻力预测)
  - [分类 vs 回归对比](#分类-vs-回归对比)
- [Part 2 · 机器学习教学案例（Scikit-learn，规划中）](#part-2--机器学习教学案例scikit-learn规划中)
- [⚙️ 统一技术特性](#️-统一技术特性)
- [🔬 标准实验工作流](#-标准实验工作流)
- [🎓 教学应用与推荐结构](#-教学应用与推荐结构)
- [📓 Notebook 与学习方式](#-notebook-与学习方式)
- [⚠️ 教学说明](#️-教学说明)
- [📄 许可证](#-许可证)

---

# Part 1 · 深度学习教学案例（TensorFlow/Keras）

## 案例 A：水质状态智能识别

### 应用场景 / Scenario
**Water Quality State Classification**

利用水质监测数据，通过人工神经网络判断水质样本属于 **Normal / 正常** 还是 **Abnormal / 异常**。本案例用于讲解人工智能中的**二分类问题（Binary Classification）**。

### 数据集 / Dataset

| 项目 | 数值 |
|---|---|
| 原始记录 | 480 条 |
| 原始变量 | 23 个 |
| 清洗后有效样本 | 274 条 |
| 输入特征 | 4 个 |
| 训练集 / 测试集 | 219 / 55 |
| 类别分布 | 正常 133 · 异常 141 |

**输入特征：** `PH`、`Air Temperature`、`dissolved_oxygen_ppm`、`Electrical Conductivity`

### 模型结构 / Model

```
Input (4)
   ↓
Dense(32, ReLU)
   ↓
Dense(16, ReLU)
   ↓
Dense(1, Sigmoid)
   ↓
Normal / Abnormal
```

可训练参数共 **705** 个。

### 核心知识点 / Concepts

| 项目 | 内容 |
|---|---|
| Task | Binary Classification |
| Input | 水质监测参数 |
| Output | 正常 / 异常 |
| Output Layer | Sigmoid |
| Loss | Binary Crossentropy |
| Metrics | Accuracy / Precision / Recall / F1 |

### 实验结果 / Results

测试集共 55 个样本：

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0.72 | 0.67 | 0.69 |
| Abnormal | 0.70 | 0.75 | 0.72 |
| **Overall Accuracy** | | | **0.7091** |

> 结果来自实际运行的 TensorFlow 代码，非预设值。该案例仅用于演示监督学习二分类流程，**不应解释为某一地区的正式环境质量监管结论**。

---

## 案例 B：帆船船体阻力预测

### 应用场景 / Scenario
**Sailboat Hull Resistance Regression**

利用船体几何参数与流态参数，建立深度神经网络模型对船舶阻力指标进行连续值预测。本案例用于讲解**回归问题（Regression）**——与案例 A 的离散分类不同，输出为一个连续数值。

### 数据集 / Dataset

- **记录数：** 1018 条（清洗前后一致）
- **变量数：** 21 个
- **预测目标：** `Rt*10^3/Delta`
  - 均值 40.9859 · 标准差 48.5547 · 范围 0.0446 – 252.4227

### 输入特征 / Features

教学中选取 7 个代表性船体与流态参数：

```
Fn · Cp · Cm · Cb · Cw · Lwl/Bwl · Bwl/Tc
```

- `X shape = (1018, 7)`，`y shape = (1018,)`
- 按 80% / 20% 划分：**训练 814 条 · 测试 204 条**

### 模型结构 / Model

```
Input (7)
   ↓
Dense(64, ReLU)
   ↓
Dense(32, ReLU)
   ↓
Dense(16, ReLU)
   ↓
Dense(1, Linear)
   ↓
连续阻力预测
```

可训练参数共 **3,137** 个。

### 核心知识点 / Concepts

| 项目 | 内容 |
|---|---|
| Task | Regression |
| Input | 船体几何与流态参数 |
| Output | 连续阻力值 |
| Output Layer | Linear |
| Loss | MSE |
| Metrics | MAE / RMSE / R² |

### 实验结果 / Results

训练 100 epochs，最终测试集：

```
MAE  = 2.0466
RMSE = 2.9985
R²   = 0.9965
```

- **MAE**：平均预测误差
- **RMSE**：对较大误差更敏感
- **R² = 0.9965**：在当前数据划分与特征设置下，预测值与真实阻力值高度一致

---

## 分类 vs 回归对比

两个案例有意设计为**成对教学结构**，帮助学生建立 AI 中最基础的两类监督学习任务概念：

> **Classification → 预测一个类别**
> **Regression → 预测一个连续值**

| 对比项 | 案例 A · 水质 | 案例 B · 船舶 |
|---|---|---|
| 任务类型 | Classification | Regression |
| 输入 | 水质监测指标 | 船体与流态参数 |
| 输出 | 正常 / 异常（离散） | 连续阻力值 |
| 输出层 | Sigmoid | Linear |
| 损失函数 | Binary Crossentropy | MSE |
| 评价指标 | Accuracy / Precision / Recall / F1 | MAE / RMSE / R² |
| 应用场景 | 水环境状态识别 | 船体阻力预测 |

```text
Classification                     Regression
     ↓                                  ↓
  Discrete Category               Continuous Value
     ↓                                  ↓
  Sigmoid                            Linear
     ↓                                  ↓
Binary Crossentropy                    MSE
     ↓                                  ↓
Accuracy / Precision /             MAE / RMSE / R²
Recall / F1
```

---

## 📊 深度学习部分结果汇总

```text
======================================================================
                    DEEP LEARNING RESULTS
======================================================================
Case A · Water Quality Classification
    Accuracy = 0.7091

Case B · Ship Resistance Regression
    MAE  = 2.0466
    RMSE = 2.9985
    R²   = 0.9965
======================================================================
```

---

# Part 2 · 机器学习教学案例（Scikit-learn，规划中）

> 以下三个扩展案例统一采用 **Random Forest** 作为教学模型：稳定、可解释、零调参、CPU 秒级训练，并统一输出特征重要性分析。

### 1. Intelligent Robot Grasping Control（第 7.2 章）
**工业机器人自适应抓取决策系统**

基于工业机器人物理运动规则与多传感器数据（关节角、三维力、位置偏移、抓取质量），构建"感知—决策—执行"闭环智能控制模型，实现抓取成功/失败的自主判断，替代传统固定示教编程。

- 工业物理规则驱动数据集
- 可视化特征重要性分析，增强控制可解释性
- 完整闭环智能机器人仿真

### 2. Industrial Predictive Maintenance Fault Diagnosis
**工业设备故障分类与预测性维护**

基于公开 **AI4I 2020 Predictive Maintenance Dataset (UCI)**，利用温度、转速、扭矩、刀具磨损等特征，实现工业机械设备的二分类故障识别，是经典的工业 AI 入门分类任务。

- 自动下载官方数据集，无需手动处理
- 标准工业机器学习流水线
- 完整可视化：特征重要性、混淆矩阵、ROC 曲线

> **数据集说明：** 10,000 条工业设备运行样本，结构化表格、无缺失值；特征含空气温度、工艺温度、转速、扭矩、刀具磨损、设备类型；标签为 `Machine failure`。

### 3. China Offshore Marine Monitoring & Red Tide Early Warning（第 7.4 章）
**中国近海生态监测与赤潮预警**

面向渤海、黄海、东海等中国近海浮标监测特征，利用水温、盐度、溶解氧、叶绿素 a、硝酸盐、磷酸盐等关键海洋指标，实现水质异常智能识别与赤潮风险预警。

- 完全符合中国近海海洋生态规律
- 模拟真实低概率海洋灾害样本分布
- 面向海洋学与环境专业的专业教学案例

---

# ⚙️ 统一技术特性

| 特性 | 说明 |
|---|---|
| **CPU 运行** | 全部实验无需 GPU，CPU 环境下秒级完成 |
| **模型轻量** | 全连接小网络 / Random Forest，无 CNN/RNN/Transformer 等复杂结构 |
| **零门槛** | 适配零基础本科生，代码基于 Scikit-learn / Keras 标准库 |
| **完整流水线** | 数据加载 → 预处理 → 训练 → 评估 → 可视化 |
| **可解释性** | 所有机器学习案例输出特征重要性分析 |
| **真实数据** | 基于真实结构化数据集，非纯模拟玩具数据（如船舶案例含 1018 条真实实验记录） |
| **结果稳定** | 随机森林抗过拟合、默认 100 棵树即可稳定复现 |

---

# 🔬 标准实验工作流

每个案例统一遵循以下教学流水线：

```
Raw Data
   ↓
Data Inspection（数据检查）
   ↓
Data Cleaning（数据清洗 / 缺失值处理）
   ↓
Feature Selection（特征选择）
   ↓
Train / Test Split（8:2 分层划分）
   ↓
Feature Scaling（标准化）
   ↓
Model（神经网络 / 随机森林）
   ↓
Training（训练）
   ↓
Prediction（预测）
   ↓
Quantitative Evaluation（量化评估）
   ↓
Visualization（可视化）
```

每个案例标准输出：

- 数据集统计与样例展示
- Accuracy / Precision / Recall / F1-Score、Classification Report
- 混淆矩阵、ROC 曲线与 AUC
- 特征重要性排序图
- 单样本实时智能决策演示

---

# 🎓 教学应用与推荐结构

### 适用课程
- 人工智能通识教育
- 智能制造工程 / 工业人工智能
- 机器人工程
- 海洋技术 / 海洋工程
- 环境科学与工程
- 大数据与人工智能

### 推荐教学顺序

**Part 1 — 什么是分类？**（水质案例）
```
Input → Neural Network → Probability → Category
```

**Part 2 — 什么是回归？**（船舶阻力案例）
```
Input → Neural Network → Continuous Value
```

**Part 3 — 两者有何区别？**
```
Classification          Regression
Sigmoid                 Linear
Binary Crossentropy     MSE
Accuracy / F1           MAE / RMSE / R²
```

---

# 📓 Notebook 与学习方式

核心 Notebook：**`DL4W_S.ipynb`**，包含 16 个教学步骤：

1. TensorFlow 环境检测 · 2. CPU/GPU 确认 · 3–8. 水质数据读取→预处理→网络→训练→Classification Report→可视化 · 9–15. 船舶数据读取→特征选择→标准化→网络→训练→指标计算→预测对比 · 16. Classification vs Regression 总结。

支持两种学习模式：

- **可运行版本：** 学生在 Google Colab 中打开 `DL4W_S.ipynb` 完整执行。
- **静态阅读版本：** Notebook 导出为 HTML，供阅读实验步骤、查看代码与图表、完成教材练习。

推荐学习路径：

```
教材 → 二维码 → GitHub 仓库 → README.md → DL4W_S.ipynb → Google Colab
```

---

# ⚠️ 教学说明

- 本仓库报告的实验结果对应某一可复现的 train/test 划分，主要用于教学演示，**不应简单理解为复杂工程环境中的生产级模型性能**。
- 水质分类案例仅用于教学演示，**不构成官方环境质量评价**。
- 船舶阻力案例的结果应理解为当前数据集、特征选择与划分条件下的教学实验结论。
- 所有实验代码标准化开源，仅供教材配套教学演示与学术研究使用。

---

# 📄 许可证

本项目仅用于本科教学、人工智能通识教育、实验实训、学术教学演示与教育研究，不用于商业用途。

---

## Authors / 作者

**Haijian Shao** · **Xing Deng**
*AI Intelligent Teaching Experiments for Smart Manufacturing & Smart Ocean*
*教材配套人工智能实训开源代码*

---

如果本项目对您的教学、学习或 AI 教育研究有帮助，欢迎 **Star ⭐ · Fork 🍴 · Use · Learn · Share**。

---

5. **加目录**：补了锚点目录，长 README 可跳转。

如果需要我直接把这份内容写成 `README.md` 文件放到项目目录里，告诉我即可。
