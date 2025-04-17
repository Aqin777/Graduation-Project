emotion_meditation_system/
│
├── models/                  # 深度学习模型
│   ├── eeg_model.py         # EEG情绪识别模型
│   ├── facial_model.py      # 面部表情识别模型
│   └── fusion_model.py      # 多模态融合模型
│
├── data/                    # 数据集和资源
│   ├── datasets/            # 公开数据集
│   ├── music/               # 音乐资源
│   ├── visuals/             # 视觉资源
│   └── guides/              # 语音引导资源
│
├── services/                # 业务逻辑
│   ├── emotion_service.py   # 情绪识别服务
│   └── content_service.py   # 内容调整服务
│
├── web/                     # Web应用
│   ├── static/              # 静态资源
│   ├── templates/           # HTML模板
│   └── app.py               # Web应用入口
│
└── docs/                    # 文档
    ├── module_design.md     # 模块化设计文档
    └── presentation/        # 演示文件
