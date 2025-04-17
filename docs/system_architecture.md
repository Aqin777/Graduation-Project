# emotion_meditation_system/

## Project Structure
```markdown
emotion_meditation_system/
│
├── docs/                    # Documentation
│   ├── system_architecture.md   # System architecture design
│   └── module_design.md     # Module design documentation
│
├── src/                     # Source code
│   ├── models/              # Deep learning models
│   │   ├── eeg_model.py     # EEG emotion recognition model
│   │   ├── facial_model.py  # Facial expression recognition model
│   │   └── fusion_model.py  # Multimodal fusion model
│   │
│   ├── services/            # Business logic
│   │   ├── emotion_service.py  # Emotion recognition service
│   │   └── content_service.py  # Content adjustment service
│   │
│   └── web/                 # Web application
│       ├── static/          # Static resources
│       ├── templates/       # HTML templates
│       └── app.py           # Web application entry
│
├── data/                    # Datasets and resources
│
└── README.md                # Project description