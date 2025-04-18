# src/main.py
# 情感识别冥想辅助系统主程序

import os
import numpy as np
import cv2
import time
import argparse
import tensorflow as tf

# 导入各个模块
from models.eeg_model import process_eeg_datasets, train_eeg_model
from models.facial_model import process_facial_dataset, train_facial_model
from models.fusion_model import build_fusion_model, prepare_multimodal_data, train_fusion_model
from services.emotion_service import EmotionRecognitionService
from services.content_service import MeditationContentService

# 设置随机种子以确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 全局参数
BATCH_SIZE = 32
EPOCHS = 50
EEG_CLASSES = 3  # 高兴，平静，悲伤
FACIAL_CLASSES = 7  # 愤怒，蔑视，厌恶，恐惧，高兴，悲伤，惊讶

# 路径设置
EEG_SEED_PATH = './data/SEED/'
EEG_DEAP_PATH = './data/DEAP/'
FACIAL_CK_PATH = './data/CK+/'
MODEL_SAVE_PATH = './models/'
DATA_PROCESSED_PATH = './data/processed/'

def prepare_directories():
    """创建必要的目录"""
    dirs = [
        MODEL_SAVE_PATH,
        DATA_PROCESSED_PATH,
        os.path.join(DATA_PROCESSED_PATH, 'facial'),
        './resources/music',
        './resources/visual',
        './reports'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def train_models():
    """训练所有模型"""
    print("=== 开始模型训练 ===")
    
    # 1. 处理EEG数据并训练EEG模型
    print("\n1. 处理EEG数据...")
    eeg_features, eeg_labels = process_eeg_datasets(
        EEG_SEED_PATH, 
        EEG_DEAP_PATH, 
        DATA_PROCESSED_PATH
    )
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_eeg_train, X_eeg_test, y_eeg_train, y_eeg_test = train_test_split(
        eeg_features, eeg_labels, test_size=0.2, random_state=42
    )
    
    print("\n2. 训练EEG模型...")
    eeg_model, _ = train_eeg_model(
        X_eeg_train, 
        y_eeg_train, 
        X_eeg_test, 
        y_eeg_test, 
        os.path.join(MODEL_SAVE_PATH, 'eeg_model.h5')
    )
    
    # 2. 处理面部表情数据并训练面部表情模型
    print("\n3. 处理面部表情数据...")
    X_facial_train, X_facial_test, y_facial_train, y_facial_test = process_facial_dataset(
        FACIAL_CK_PATH, 
        os.path.join(DATA_PROCESSED_PATH, 'facial')
    )
    
    print("\n4. 训练面部表情模型...")
    facial_model, _ = train_facial_model(
        X_facial_train, 
        y_facial_train, 
        X_facial_test, 
        y_facial_test, 
        os.path.join(MODEL_SAVE_PATH, 'facial_model.h5')
    )
    
    # 3. 构建和训练融合模型
    print("\n5. 构建融合模型...")
    fusion_model = build_fusion_model(
        os.path.join(MODEL_SAVE_PATH, 'eeg_model.h5'),
        os.path.join(MODEL_SAVE_PATH, 'facial_model.h5')
    )
    
    print("\n6. 准备多模态训练数据...")
    # 加载处理后的数据
    eeg_features = np.load(os.path.join(DATA_PROCESSED_PATH, 'eeg_features.npy'))
    eeg_labels = np.load(os.path.join(DATA_PROCESSED_PATH, 'eeg_labels.npy'))
    facial_images = np.load(os.path.join(DATA_PROCESSED_PATH, 'facial/X_train.npy'))
    facial_labels = np.load(os.path.join(DATA_PROCESSED_PATH, 'facial/y_train.npy'))
    
    X_eeg, X_facial, y = prepare_multimodal_data(
        eeg_features, 
        eeg_labels, 
        facial_images, 
        facial_labels
    )
    
    print("\n7. 训练融合模型...")
    train_fusion_model(
        fusion_model,
        X_eeg,
        X_facial,
        y,
        os.path.join(MODEL_SAVE_PATH, 'fusion_model.h5')
    )
    
    print("\n=== 所有模型训练完成 ===")

def simulate_meditation_session():
    """模拟一次冥想会话，测试系统功能"""
    print("\n=== 模拟冥想会话 ===")
    
    # 初始化服务
    emotion_service = EmotionRecognitionService(
        fusion_model_path=os.path.join(MODEL_SAVE_PATH, 'fusion_model.h5'),
        eeg_scaler_path=os.path.join(DATA_PROCESSED_PATH, 'eeg_scaler.pkl')
    )
    
    content_service = MeditationContentService()
    
    # 1. 情感分析
    print("\n1. 情感分析...")
    
    # 模拟数据
    dummy_eeg = np.random.randn(62 * 9)  # 假设EEG特征维度为62*9
    dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 分析情感
    emotion, confidence = emotion_service.analyze_emotion(dummy_eeg, dummy_face)
    print(f"  检测到的情感: {emotion}, 置信度: {confidence:.2f}")
    
    # 2. 获取冥想内容
    print("\n2. 获取冥想内容...")
    content = content_service.get_meditation_content(emotion)
    print(f"  推荐音乐: {content['music']}")
    print(f"  推荐视觉效果: {content['visual']}")
    print(f"  引导语: {content['guidance']}")
    
    # 3. 开始冥想会话
    print("\n3. 开始冥想会话...")
    content_service.start_meditation_session(duration=0.1)  # 6秒
    
    # 记录初始情感
    content_service.record_emotion(emotion)
    
    # 模拟情感变化
    time.sleep(3)
    
    # 再次分析情感
    new_emotion, confidence = emotion_service.analyze_emotion(None, dummy_face)  # 只使用面部
    print(f"\n4. 情感状态更新: {new_emotion}, 置信度: {confidence:.2f}")
    
    # 记录新情感
    content_service.record_emotion(new_emotion)
    
    # 更新冥想内容
    new_guidance = content_service.get_personalized_guidance(new_emotion)
    print(f"\n5. 更新冥想引导语: {new_guidance}")
    
    # 等待会话结束
    time.sleep(3)
    content_service.end_meditation_session()
    
    print("\n=== 模拟会话完成 ===")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='情感识别冥想辅助系统')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--simulate', action='store_true', help='模拟冥想会话')
    args = parser.parse_args()
    
    # 创建必要的目录
    prepare_directories()
    
    if args.train:
        train_models()
    
    if args.simulate:
        simulate_meditation_session()
    
    # 如果没有指定参数，默认执行模拟
    if not args.train and not args.simulate:
        simulate_meditation_session()

if __name__ == "__main__":
    main()