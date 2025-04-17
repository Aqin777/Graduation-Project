# 数据集和资源

本目录包含项目使用的数据集和资源文件。

## 数据集说明
- SEED数据集：用于EEG信号分析
- DEAP数据集：用于EEG信号分析，映射为三类离散情绪（高兴、平静、悲伤）
- CK+数据集：用于面部表情识别，包含七种情绪（愤怒、蔑视、厌恶、恐惧、高兴、悲伤、惊讶）

## 数据集放置结构说明

请按照以下结构组织数据集文件：

```
data/
├── SEED/                  # SEED脑电图数据集
│   ├── raw/               # 原始EEG数据
│   └── processed/         # 预处理后的数据
│
├── DEAP/                  # DEAP情绪数据集
│   ├── raw/               # 原始数据
│   ├── processed/         # 预处理后的数据
│   └── labels/            # 情绪标签（高兴、平静、悲伤）
│
├── CK+/                   # CK+面部表情数据集
│   ├── images/            # 面部表情图像
│   │   ├── anger/         # 愤怒表情
│   │   ├── contempt/      # 蔑视表情
│   │   ├── disgust/       # 厌恶表情
│   │   ├── fear/          # 恐惧表情
│   │   ├── happiness/     # 高兴表情
│   │   ├── sadness/       # 悲伤表情
│   │   └── surprise/      # 惊讶表情
│   └── labels/            # 表情标签
│
└── meditation_content/    # 冥想内容资源
    ├── music/             # 背景音乐
    ├── guidance/          # 语音引导
    └── visuals/           # 视觉效果资源
```
## 注意事项
1. 由于数据集文件较大，请不要将它们提交到Git仓库中
2. SEED数据集可从[SEED数据集官方网站](http://bcmi.sjtu.edu.cn/~seed/seed.html)下载
3. DEAP数据集可从[DEAP数据集官方网站](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)下载
4. CK+数据集可从[CK+数据集官方网站](https://www.jeffcohn.net/Resources/)下载
5. 下载后按照上述结构放置，或者修改`src/config.py`中的数据路径配置
