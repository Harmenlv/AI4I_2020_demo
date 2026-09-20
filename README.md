下面是**完整无删减纯文本txt版本**，全部内容保留，你直接全选复制即可，没有丢失任何段落。

```txt
# AI Intelligent Teaching Experiments for Smart Manufacturing & Smart Ocean
> 教材配套人工智能实训开源代码

**Authors:** Haijian Shao, Xing Deng
**Version:** 2.0
**Platform:** Google Colab / Local Python
**Framework:** TensorFlow 2.20.0 (Deep Learning Cases) | Scikit-learn (Machine Learning Cases)
**Hardware:** CPU-based / Zero GPU Required
**Level:** Undergraduate AI General Education

---

## 📌 Project Introduction / 项目简介
This open-source repository provides reproducible artificial intelligence teaching experiments designed for undergraduate courses in artificial intelligence, intelligent manufacturing, smart ocean, marine technology, environmental science, and related interdisciplinary programs.

本项目是本科人工智能通识课程及相关专业教材的配套开源实训代码，面向**智能制造、智慧海洋、人工智能通识、环境科学与工程、海洋技术等高校课程**，提供可直接运行、可重复实验和可用于教材教学演示的 AI 案例。

The project emphasizes a complete and intuitive AI learning workflow:
> **Data → Preprocessing → Model → Training → Evaluation → Visualization → Application**

项目重点不是追求复杂模型，而是帮助零基础或低基础学生理解人工智能模型从**数据输入到智能预测**的完整过程。

Deep-learning experiments are implemented with **TensorFlow/Keras**, and traditional machine learning experiments are built on **Scikit-learn**. All examples can be executed on ordinary CPU environments. GPU acceleration is **not required**.

深度学习案例基于 TensorFlow/Keras 实现，传统机器学习案例基于 Scikit-learn 实现；全部案例均可在普通 CPU 环境运行，**无需 GPU**。

---

# 📚 Part 1: Deep Learning Teaching Cases (TensorFlow/Keras)
## 案例 A：水质状态智能识别
### Scenario / 应用场景
**Water Quality State Classification**

利用水质监测数据，通过人工神经网络判断水质样本属于：
* **Normal / 正常**
* **Abnormal / 异常**

该案例用于讲解人工智能中的**二分类问题**。

### Dataset
The water-quality dataset contains:
* **480 original records**
* **23 variables**
原始数据规模为 **480 × 23**。实际用于深度学习的输入变量包括：
* `PH`
* `Air Temperature`
* `dissolved_oxygen_ppm`
* `Electrical Conductivity`

原始数据经过缺失值处理后，最终得到：
* **274 valid samples**
* **4 input features**
* **219 training samples**
* **55 test samples**
实验结果显示，正常样本 133 个，异常样本 141 个。

### Deep Learning Model
The classification network is:
```text
Input
  ↓
Dense(32, ReLU)
  ↓
Dense(16, ReLU)
  ↓
Dense(1, Sigmoid)
  ↓
Normal / Abnormal
```
The network contains **705 trainable parameters**.

### Core AI Concepts
| 项目           | 内容                                  |
| ------------ | ----------------------------------- |
| Task         | Binary Classification               |
| Input        | Water-quality monitoring parameters |
| Output       | Normal / Abnormal                   |
| Output Layer | Sigmoid                             |
| Loss         | Binary Crossentropy                 |
| Metrics      | Accuracy / Precision / Recall / F1  |
| Application  | Water-quality state recognition     |

### Experimental Result
```text
Accuracy = 0.7091
```
测试集共 55 个样本：
| Class            | Precision | Recall |         F1 |
| ---------------- | --------: | -----: | ---------: |
| Normal           |      0.72 |   0.67 |       0.69 |
| Abnormal         |      0.70 |   0.75 |       0.72 |
| Overall Accuracy |           |        | **0.7091** |
实验结果来自实际运行的 TensorFlow 深度学习代码，而非预设结果。
> **Teaching Note:**
> The water-quality label is used as a teaching-oriented classification task. It is intended to demonstrate the basic workflow of supervised deep learning rather than serve as an official environmental regulatory assessment.
> **教学说明：**
> 水质案例主要用于展示监督学习中的二分类流程，实验结果不应直接解释为某一地区的正式环境质量监管结论。

---
# 🚢 案例 B：帆船船体阻力预测
## Scenario / 应用场景
**Sailboat Hull Resistance Regression**
利用船体几何参数和流态参数，建立深度神经网络模型，对船舶阻力指标进行预测。
该案例用于讲解人工智能中的：
> **Regression / 回归问题**
与案例 A 的分类任务不同，案例 B 不输出“正常/异常”等离散类别，而是输出一个**连续数值**。

## Dataset
The sailboat hull resistance dataset contains:
```text
1018 records
21 variables
```
数据字段包括：
* `Serie`
* `Sysser`
* `Rt*10^3/Delta`
* `Rn`
* `Fn`
* `Cp`
* `Cm`
* `Cb`
* `Cw`
* `Lwl/Bwl`
* `Bwl/Tc`
* `Lwl/Tc`
* `Lwl/Vol^(1/3)`
* `Lcb/Lwl`
* `Lcf/Lwl`
* `Lcb/Lcf`
* `Sc/Vol^(2/3)`
* `Aw/Vol^(2/3)`
* `Sc/Aw`
* `Sc/Ax`
* `Ax/Aw`

数据集中的预测目标为：
```text
Rt*10^3/Delta
```
目标变量共有 1018 条有效记录，其均值为 40.9859，标准差为 48.5547，范围为 0.0446–252.4227。

## Input Features
For teaching purposes, seven representative hull and flow parameters are selected:
```text
Fn
Cp
Cm
Cb
Cw
Lwl/Bwl
Bwl/Tc
```
最终形成：
```text
X shape = (1018, 7)
y shape = (1018,)
```
数据无需删除记录，清洗前后均为 1018 条。数据按照 80%/20% 划分：
```text
Training samples = 814
Testing samples  = 204
```

## Deep Learning Model
The regression network is:
```text
Input
  ↓
Dense(64, ReLU)
  ↓
Dense(32, ReLU)
  ↓
Dense(16, ReLU)
  ↓
Dense(1, Linear)
  ↓
Continuous Resistance Prediction
```
The complete network contains **3,137 trainable parameters**.

## Core AI Concepts
| 项目           | 内容                                |
| ------------ | --------------------------------- |
| Task         | Regression                        |
| Input        | Hull geometry and flow parameters |
| Output       | Continuous resistance value       |
| Output Layer | Linear                            |
| Loss         | MSE                               |
| Metrics      | MAE / RMSE / R²                   |
| Application  | Ship resistance prediction        |

---
# 📊 Final Experimental Results
The model was trained for 100 epochs. The training loss and validation loss decreased substantially during optimization, while the validation error remained at a relatively low level in the later training stage.
最终测试集结果：
```text
MAE  = 2.0466
RMSE = 2.9985
R²   = 0.9965
```
因此，本案例可以直观展示：
* **MAE**：平均预测误差
* **RMSE**：对较大预测误差更加敏感
* **R²**：模型对目标变量变化的解释程度
其中测试集 **R² = 0.9965**，说明在当前实验划分和输入特征设置下，模型预测值与实际阻力值具有较高的一致性。实验结果来自实际运行数据。

---
# 🧠 Classification vs Regression
The two cases are intentionally designed as a paired teaching structure.
两个案例不是简单地展示两个不同数据集，而是用来帮助学生建立人工智能中最基础的两个任务概念：
> **Classification → Predict a category**
> **Regression → Predict a continuous value**

| 项目   | 案例 A：水质                            | 案例 B：船舶         |
| ---- | ---------------------------------- | --------------- |
| 任务类型 | Classification                     | Regression      |
| 输入   | 水质监测指标                             | 船体与流态参数         |
| 输出   | 正常 / 异常                            | 连续阻力值           |
| 输出层  | Sigmoid                            | Linear          |
| 损失函数 | Binary Crossentropy                | MSE             |
| 评价指标 | Accuracy / Precision / Recall / F1 | MAE / RMSE / R² |
| 应用   | 水环境状态识别                            | 船体阻力预测          |

---
# 📈 Final Experimental Summary
```text
======================================================================
FINAL EXPERIMENTAL RESULTS
======================================================================
Case A: Water Quality Classification
Accuracy = 0.7091
Case B: Ship Resistance Regression
MAE  = 2.0466
RMSE = 2.9985
R²   = 0.9965
======================================================================
```
## 教材核心知识
```text
Classification
     ↓
Discrete Category
     ↓
Sigmoid
     ↓
Binary Crossentropy
     ↓
Accuracy / Precision / Recall / F1
```
```text
Regression
     ↓
Continuous Value
     ↓
Linear
     ↓
MSE
     ↓
MAE / RMSE / R²
```

---
# 💻 Main Notebook
## `DL4W_S.ipynb`
`DL4W_S.ipynb` 是当前教材深度学习案例的核心 Jupyter Notebook。
主要内容包括：
1. TensorFlow 环境检测
2. CPU/GPU 状态确认
3. 水质数据读取
4. 水质数据预处理
5. 水质分类神经网络
6. 分类模型训练
7. Classification Report
8. 水质分类结果可视化
9. 船舶数据读取
10. 船舶特征选择
11. 船舶数据标准化
12. 船舶阻力神经网络
13. 回归模型训练
14. MAE / RMSE / R² 计算
15. 预测值与真实值对比
16. Classification vs Regression 知识总结

---
# ⚙️ Technical Features / 技术特点
### 1. CPU-Based
All experiments can be executed without GPU acceleration.
```text
TensorFlow 2.20.0
GPU = []
CPU execution
```
当前实验已经在 CPU 环境完成运行验证。
### 2. Beginner-Friendly
The models intentionally use small fully connected neural networks rather than complex CNN, RNN, Transformer, or large-scale foundation models.
模型结构控制在较小规模：
```text
Classification:
4 → 32 → 16 → 1
Regression:
7 → 64 → 32 → 16 → 1
```
因此适合人工智能通识课程和本科生第一次接触深度学习。
### 3. Complete AI Workflow
```text
Raw Data
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Neural Network
   ↓
Training
   ↓
Prediction
   ↓
Quantitative Evaluation
   ↓
Visualization
```
### 4. Real-Data Teaching
The examples are based on actual structured datasets rather than purely simulated toy data.
其中船舶案例直接使用 1018 条真实帆船船体阻力实验记录，并保留其原始变量体系。

---
# 📚 Teaching Applications / 教学应用
This repository can support teaching in:
* Artificial Intelligence General Education
* Intelligent Manufacturing
* Industrial Artificial Intelligence
* Marine Technology
* Ocean Engineering
* Environmental Science and Engineering
* Big Data and Artificial Intelligence
* Engineering AI Applications

---
# 🎓 Recommended Teaching Structure
The two cases can be taught consecutively as a basic deep-learning module.
### Part 1 — What is Classification?
Students learn:
```text
Input → Neural Network → Probability → Category
```
using the water-quality case.
### Part 2 — What is Regression?
Students learn:
```text
Input → Neural Network → Continuous Value
```
using the ship-resistance case.
### Part 3 — What is the Difference?
Students compare:
```text
Classification
Sigmoid
Binary Crossentropy
Accuracy / F1
```
with:
```text
Regression
Linear
MSE
MAE / RMSE / R²
```
This structure helps students understand the fundamental distinction between the two most common supervised-learning tasks.

---
# 📓 Notebook and Static Teaching Materials
The repository is intended to support two complementary learning modes:
### 1. Runnable Version
```text
DL4W_S.ipynb
```
Students can open the notebook in Google Colab and execute the complete experiment.
### 2. Static Reading Version
The notebook can be exported to HTML for students who only need to:
* read the experimental procedure;
* inspect the code;
* view the experimental results;
* understand the figures;
* complete textbook exercises.
Recommended workflow:
```text
Textbook
   ↓
QR Code
   ↓
GitHub Repository
   ↓
README.md
   ↓
DL4W_S.ipynb
   ↓
Google Colab
```

---
# ⚠️ Teaching Notes
The experimental results reported in this repository correspond to one reproducible train/test split and are intended primarily for teaching demonstration.
实验结果用于说明深度学习模型的基本工作机制，不应简单理解为复杂工程环境中的最终生产模型性能。
For the water-quality case in particular, the classification task is intended as a teaching demonstration and should not be interpreted as an official environmental regulatory assessment.
对于船舶阻力案例，模型结果同样应理解为当前数据集、特征选择和实验划分条件下的教学实验结果。

---
# 📄 License & Statement
This project is intended for:
* Undergraduate teaching
* Artificial intelligence general education
* Experimental training
* Academic teaching demonstration
* Educational research
本项目主要用于：
* 本科人工智能教学
* 人工智能通识教育
* 深度学习实验教学
* 工程AI案例教学
* 教材配套实验
* 学术教学演示
The repository is provided for educational and academic purposes.

---
# 🌟 Star if Helpful
If this project is useful for your teaching, learning, or AI education research, welcome to:
**Star ⭐ · Fork 🍴 · Use · Learn · Share**

---
## Authors / 作者
**Haijian Shao**
**Xing Deng**
**AI Intelligent Teaching Experiments for Smart Manufacturing & Smart Ocean**
**教材配套人工智能实训开源代码**

---
# 后续扩展案例（待开发）
1. Intelligent Robot Grasping Control (Chapter 7.2)
Scenario: Industrial robot adaptive grasping decision system
Description
Based on industrial robot physical motion rules and multi-dimensional sensor data (joint angle, 3D force, position offset, grasping quality), this case builds a Perception–Decision–Execution closed-loop intelligent control model. It realizes autonomous judgment of grasping success or failure, replacing traditional fixed teaching programming.
Highlights
- Industrial physical rule-driven dataset
- Visual feature importance analysis for robot control interpretability
- Full closed-loop intelligent robot simulation

2. Industrial Predictive Maintenance Fault Diagnosis
Scenario: Equipment intelligent fault classification and predictive maintenance
Description
Based on the public AI4I 2020 Predictive Maintenance Dataset (UCI), this experiment realizes binary fault identification of industrial mechanical equipment using temperature, rotation speed, torque, and tool wear features. It is a classic introductory industrial AI classification task.
Highlights
- Auto-download official dataset, no manual processing
- Standard industrial machine learning pipeline
- Complete visualization: feature importance, confusion matrix, ROC curve

3. China Offshore Marine Monitoring & Red Tide Early Warning (Chapter 7.4)
Scenario: Smart ocean ecological monitoring and marine disaster early warning
Description
Constructed based on China offshore buoy monitoring characteristics (Bohai Sea, Yellow Sea, East China Sea). Using key marine indicators such as water temperature, salinity, DO, chlorophyll-a, nitrate, and phosphate, the model realizes intelligent identification of water quality anomalies and red tide risk early warning.
Highlights
- Fully conforms to China offshore marine ecological laws
- Simulates real low-probability marine disaster sample distribution
- Professional and teachable for oceanography and environmental majors

---
⚙️ Unified Technical Features / 统一技术特点
- Unified Model: Random Forest (best for teaching, stable, explainable, no tuning needed)
- Unified Pipeline: Data loading → preprocessing → training → evaluation → visualization
- Zero GPU: All experiments run on CPU within seconds
- Zero threshold: Suitable for beginner teaching and experimental reproduction
- Full interpretability: Feature importance analysis for all cases
- Textbook-level results: Standard figures and indicators for textbook publishing

---
📊 Standard Experimental Output / 标准实验输出内容
Each case outputs complete teaching results:
- Dataset statistics & sample display
- Accuracy, Precision, Recall, F1-Score
- Classification report
- Feature importance visualization
- Real-time intelligent decision demonstration

---
🌊 Application Scenarios / 适用教学场景
- Intelligent Manufacturing Engineering
- Industrial Artificial Intelligence
- Robot Engineering
- Marine Technology & Oceanography
- Environmental Science and Engineering
- Big Data and Artificial Intelligence General Education

---
📄 License & Statement
This project is for teaching and academic research purposes only. All codes are standardized and open-source for textbook supporting teaching demonstration.
本项目仅供教学与学术研究使用，为教材配套开源实训代码。

---
🌟 Star if helpful | 欢迎 Star、Fork、学习使用

# AI4I 2020 Predictive Maintenance Teaching Demo
# 基于随机森林与预测性维护数据集的工业应用-AI教学实践
## Authors / 作者
**Haijian Shao**, **Xing Deng**

---
# 1. Training Program Confirmation
一、实训方案确认
**English**:
The supporting experimental code in this textbook completely corresponds to:
**AI4I 2020 Predictive Maintenance Dataset + Random Forest Fault Classification Model**.
This combination is the optimal solution for introductory teaching, student reproduction, and textbook illustration demonstration. It requires **no GPU dependency**, features fast training speed, stable experimental results, and standard visualization effects.

**中文**:
本书配套上机实验代码完全对应：
**AI4I 2020 预测性维护数据集 + 随机森林（Random Forest）故障分类模型**。
该组合是适配入门教学、学生复现、教材配图演示的最优实训方案，全程无GPU依赖、训练速度快、实验结果稳定、可视化效果标准。

---
# 2. Core Positioning of Training Program
二、本实训方案核心定位
**English**:
This experiment takes **binary fault identification of industrial equipment** as the core task. Based on interpretable industrial parameters such as equipment operating temperature, rotational speed, torque, and tool wear, it intelligently judges whether the equipment is in normal operation or failure state. It is a classic introductory practical case for industrial artificial intelligence and machine learning classification tasks.
Compared with advanced tasks such as time series forecasting, vibration signal fault diagnosis, and remaining useful life regression prediction, this case focuses on the **standard traditional machine learning workflow**, which can be quickly understood and fully reproduced by zero-basis students.

**中文**:
本实验以**工业设备二分类故障识别**为核心任务：通过设备运行温度、转速、扭矩、刀具磨损等可解释工业参数，智能判断设备处于正常运行或发生故障状态，是工业人工智能、机器学习分类任务的经典入门落地案例。
区别于复杂时序预测、振动信号故障诊断、寿命回归预测等进阶任务，本案例聚焦**传统机器学习标准化流程**，零基础学生可快速理解、完整复现。

---
# 3. Dataset Matching Description
三、数据集匹配说明（对应代码数据来源）
**English**:
The experimental code adopts the officially open-source **AI4I 2020 Predictive Maintenance Dataset** from UCI.
Built-in automatic download logic requires no manual file upload, compatible with Google Colab Free Edition, Kaggle Notebook, and local Python environments.
Core attributes fully meet teaching requirements:
- **Dataset Size**: 10,000 industrial equipment operation samples, moderate scale with fast loading and computation.
- **Data Quality**: Structured tabular data with no missing values or dirty data, no complex preprocessing required.
- **Core Features**: Interpretable features including air temperature, process temperature, rotational speed, torque, tool wear, and equipment type.
- **Training Label**: `Machine failure` (equipment fault status), standard binary label suitable for introductory classification tasks.

**中文**:
实验代码采用 **UCI官方开源 AI4I 2020 预测性维护数据集**。
代码内置自动下载逻辑，无需手动上传文件，兼容Colab免费版、Kaggle Notebook、本地Python环境。
数据集核心属性完全适配教学需求：
- **数据量**：10000条工业设备运行样本，规模适中，加载与运算无压力。
- **数据质量**：结构化表格数据，无缺失值、无脏数据，无需复杂预处理。
- **核心特征**：空气温度、工艺温度、设备转速、输出扭矩、刀具磨损、设备类型等可解释特征。
- **训练标签**：`Machine failure`（设备是否故障），标准二分类标签，适配入门分类任务。

---
# 4. Teaching Adaptation Advantages of Random Forest
四、随机森林模型教学适配优势
**English**:
Random Forest is selected as the core algorithm, with the highest cost performance and strong teaching suitability for textbook training scenarios. Core advantages:
- **No parameter tuning required**: Default 100 decision trees produce stable and high-precision results; students do not need to master complex hyperparameter optimization.
- **CPU ultra-fast training**: Model training completes in seconds without stuttering or out-of-memory errors, compatible with low-configuration teaching environments.
- **Strong interpretability**: Native support for feature importance calculation, intuitively revealing which operating parameters most easily cause equipment faults, fitting the key points of AI interpretability teaching.
- **Strong anti-overfitting ability**: Better generalization than a single decision tree with stable experimental results and no large deviation in single-run output.
- **Standard and concise code**: Implemented based on Scikit-learn, easy to understand and consistent with university machine learning teaching paradigms.

**中文**:
代码选用随机森林分类模型作为核心算法，是本教材实训场景下性价比最高、教学性最强的模型，核心优势如下：
- **零调参可用**：默认100棵决策树即可输出稳定高精度结果，学生无需掌握复杂超参数调优。
- **CPU极速训练**：全程秒级完成训练，无卡顿、无内存溢出，适配所有低配教学环境。
- **天然可解释**：原生支持特征重要性计算，直观展示哪些运行参数最易引发设备故障，契合AI可解释性教学要点。
- **抗拟合能力强**：相比单决策树泛化效果更优，实验结果稳定，避免单次运行结果差异过大。
- **代码简洁规范**：基于Sklearn标准库实现，通俗易懂，完全符合高校机器学习教学范式。

---
# 5. Complete Experimental Workflow
五、完整实训流程（与代码逐段对应）
**English**:
The code strictly follows the standard industrial AI data analysis pipeline, corresponding to textbook knowledge points for classroom teaching and experiment report writing:
1. **Library Import**: Import toolkits for data processing, visualization, model training and evaluation metrics.
2. **Automatic Dataset Download & Loading**: Automatically fetch official dataset without manual operation.
3. **Data Preprocessing**: Remove redundant ID columns and encode categorical features of equipment type.
4. **Feature & Label Partition**: Separate input features `X` and fault label `y`.
5. **Train-Test Split**: Split dataset into 8:2 with stratified sampling to ensure consistent distribution.
6. **Random Forest Model Training**: Initialize and fit the classifier.
7. **Model Prediction**: Output classification results and probability values for evaluation and plotting.
8. **Quantitative Evaluation**: Calculate accuracy, classification report, confusion matrix and AUC.
9. **Visualization Analysis**: Generate four core teaching plots: fault distribution, feature importance, confusion matrix and ROC curve.
10. **Sample Inference**: Single-sample prediction to simulate on-site industrial fault detection.
11. **Result Export**: Save feature importance data for post-class analysis and report writing.

**中文**:
本实验代码严格遵循工业AI数据分析标准流程，每一步均对应教材知识点，适配课堂讲解与实验报告撰写：
1. **环境与库导入**：导入数据处理、绘图、模型训练、评价指标全套工具库。
2. **数据集自动下载与加载**：代码自动拉取官方数据集，无需人工上传文件。
3. **数据预处理**：删除无用编号字段、对设备类型类别特征编码，清洗规整数据。
4. **特征与标签划分**：分离输入特征 `X` 与故障预测标签 `y`。
5. **数据集划分**：按8:2划分训练集/测试集，分层抽样保证数据分布一致。
6. **随机森林模型训练**：初始化模型并完成拟合训练。
7. **模型预测**：输出测试集分类结果与概率值，用于评价与绘图。
8. **模型定量评价**：计算准确率、分类报告、混淆矩阵、AUC指标。
9. **可视化分析**：生成故障分布、特征重要性、混淆矩阵、ROC曲线四大教学插图。
10. **样本推理演示**：单样本预测实操，模拟工业现场故障检测场景。
11. **结果保存**：导出特征重要性数据，用于课后分析与实验报告撰写。

---
# 6. Textbook Illustration Advantages
六、本方案专属教材配图优势
**English**:
This scheme generates four textbook-level standard visualization results, suitable for textbook screenshots, courseware display and laboratory reports:
1. **Machine Failure Distribution Bar Chart**: Intuitively show the distribution of normal and fault samples.
2. **Feature Importance Ranking Plot**: Core highlight to reflect the interpretability of AI data analysis.
3. **Confusion Matrix**: Classic visualization for classification model evaluation.
4. **ROC Curve & AUC Score**: Reflect model generalization and enhance experimental professionalism.

**中文**:
本实训组合可生成四张教科书级标准可视化结果，完全适配教材截图、课件展示、实验报告：
1. **设备故障分布柱状图**：直观展示数据集正负样本分布。
2. **特征重要性排序图**：核心特色，体现AI数据分析的可解释能力。
3. **混淆矩阵**：分类模型经典评价可视化图表。
4. **ROC曲线+AUC值**：体现模型泛化性能，提升实验专业度。

---
# 7. Teaching Adaptation Summary
七、教学适配总结
**English**:
**AI4I 2020 Dataset + Random Forest Classification Model** is a perfectly matched introductory industrial AI training scheme for this textbook.
The code supports fully automatic operation with wide environment compatibility, ultra-high training efficiency, and intuitive professional results. It enables zero-basis students to reproduce experiments quickly and generates standard AI teaching achievements, perfectly fitting the teaching needs of intelligent manufacturing, industrial big data, and artificial intelligence general courses.

**中文**:
**AI4I 2020数据集 + 随机森林分类模型** 是完全适配本教材的入门级工业AI实训方案。
代码可全自动运行、环境适配性广、训练效率极高、实验结果直观专业；既满足零基础学生快速复现实操，又可产出标准AI教学成果，完美契合智能制造、工业大数据、人工智能通识课程的教学需求。
```
