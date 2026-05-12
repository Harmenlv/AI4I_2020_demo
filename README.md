# AI4I 2020 Predictive Maintenance Teaching Demo
# 基于随机森林与预测性维护数据集的工业应用-AI教学实践

## Authors / 作者
**Haijian Shao**, **Xing Deng**

---

# 1. Training Program Confirmation / 一、实训方案确认
**English**:
The supporting experimental code in this textbook completely corresponds to:
**AI4I 2020 Predictive Maintenance Dataset + Random Forest Fault Classification Model**.
This combination is the optimal solution for introductory teaching, student reproduction, and textbook illustration demonstration. It requires **no GPU dependency**, features fast training speed, stable experimental results, and standard visualization effects.

**中文**:
本书配套上机实验代码完全对应：
**AI4I 2020 预测性维护数据集 + 随机森林（Random Forest）故障分类模型**。
该组合是适配入门教学、学生复现、教材配图演示的最优实训方案，全程无GPU依赖、训练速度快、实验结果稳定、可视化效果标准。

---

# 2. Core Positioning of Training Program / 二、本实训方案核心定位
**English**:
This experiment takes **binary fault identification of industrial equipment** as the core task. Based on interpretable industrial parameters such as equipment operating temperature, rotational speed, torque, and tool wear, it intelligently judges whether the equipment is in normal operation or failure state. It is a classic introductory practical case for industrial artificial intelligence and machine learning classification tasks.

Compared with advanced tasks such as time series forecasting, vibration signal fault diagnosis, and remaining useful life regression prediction, this case focuses on the **standard traditional machine learning workflow**, which can be quickly understood and fully reproduced by zero-basis students.

**中文**:
本实验以**工业设备二分类故障识别**为核心任务：通过设备运行温度、转速、扭矩、刀具磨损等可解释工业参数，智能判断设备处于正常运行或发生故障状态，是工业人工智能、机器学习分类任务的经典入门落地案例。

区别于复杂时序预测、振动信号故障诊断、寿命回归预测等进阶任务，本案例聚焦**传统机器学习标准化流程**，零基础学生可快速理解、完整复现。

---

# 3. Dataset Matching Description / 三、数据集匹配说明（对应代码数据来源）
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

# 4. Teaching Adaptation Advantages of Random Forest / 四、随机森林模型教学适配优势
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

# 5. Complete Experimental Workflow / 五、完整实训流程（与代码逐段对应）
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

# 6. Textbook Illustration Advantages / 六、本方案专属教材配图优势
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

# 7. Teaching Adaptation Summary / 七、教学适配总结
**English**:
**AI4I 2020 Dataset + Random Forest Classification Model** is a perfectly matched introductory industrial AI training scheme for this textbook.
The code supports fully automatic operation with wide environment compatibility, ultra-high training efficiency, and intuitive professional results. It enables zero-basis students to reproduce experiments quickly and generates standard AI teaching achievements, perfectly fitting the teaching needs of intelligent manufacturing, industrial big data, and artificial intelligence general courses.

**中文**:
**AI4I 2020数据集 + 随机森林分类模型** 是完全适配本教材的入门级工业AI实训方案。
代码可全自动运行、环境适配性广、训练效率极高、实验结果直观专业；既满足零基础学生快速复现实操，又可产出标准AI教学成果，完美契合智能制造、工业大数据、人工智能通识课程的教学需求。

---
