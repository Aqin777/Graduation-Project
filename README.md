# 情绪识别冥想辅助系统

基于深度学习的情绪识别冥想辅助系统，通过分析用户的脑电图(EEG)信号和面部表情，实时识别情绪状态，并提供个性化的冥想指导。

## 项目简介

在日常生活中，随着社会压力的增加，越来越多的人开始关注心理健康和情绪管理。冥想作为一种有效的放松和心理调节方法，已经受到广泛关注。然而，许多人在没有指导的情况下难以正确练习冥想，因此需要一个能够基于个人情绪状态提供个性化辅助的系统。

本项目设计并实现了一个深度学习情绪识别冥想辅助系统，该系统通过实时分析用户的脑电图信号和面部表情来识别情绪状态，并根据情绪变化动态调整冥想内容。

## 主要功能

- 多模态情绪识别：结合EEG信号和面部表情进行情绪识别
- 实时情绪监测：持续跟踪用户情绪状态的变化
- 个性化冥想指导：根据情绪状态自动调整背景音乐、视觉效果和语音指导
- 用户反馈机制：收集用户对冥想体验的评价
- 数据分析与可视化：提供情绪变化趋势的数据分析和可视化

## 技术栈

- 深度学习框架：TensorFlow/PyTorch
- 数据处理：NumPy, Pandas
- Web应用：Flask/Django
- 前端：HTML, CSS, JavaScript
- 数据可视化：Matplotlib, D3.js

## 数据集

- EEG信号数据集：SEED数据集和DEAP数据集
- 面部表情数据集：CK+数据集

## 安装与使用

### 环境要求

- Python 3.8+
- [其他依赖项...]

### 安装步骤

```bash
git clone https://[您的仓库地址].git
cd emotion_meditation_system
pip install -r requirements.txt
```

### 运行系统

```bash
python src/web/app.py
```

## 项目结构

```
emotion_meditation_system/
│
├── docs/                    # 文档
│   ├── system_architecture.md   # 系统架构设计
│   └── module_design.md     # 模块化设计文档
│
├── src/                     # 源代码
│   ├── models/              # 深度学习模型
│   │   ├── eeg_model.py     # EEG情绪识别模型
│   │   ├── facial_model.py  # 面部表情识别模型
│   │   └── fusion_model.py  # 多模态融合模型
│   │
│   ├── services/            # 业务逻辑
│   │   ├── emotion_service.py  # 情绪识别服务
│   │   └── content_service.py  # 内容调整服务
│   │
│   └── web/                 # Web应用
│       ├── static/          # 静态资源
│       ├── templates/       # HTML模板
│       └── app.py           # Web应用入口
│
├── data/                    # 数据集和资源
│
└── README.md                # 项目说明
```

## 贡献者

- 秦朗 - 东北林业大学计算机科学与技术专业2021级

## 指导教师

- 牛娜

## 开发进度

- [x] 项目开题
- [x] 系统需求分析
- [x] 系统架构设计
- [x] 模型开发与训练
- [x] 系统实现
- [ ] 测试与评估
- [ ] 论文撰写

## 许可证

本项目为东北林业大学本科毕业设计项目，版权所有。

```
