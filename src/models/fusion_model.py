# src/models/fusion_model.py
# 多模态融合模型

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# 设置随机种子以确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

def build_fusion_model(eeg_model_path, facial_model_path, num_emotions=3):
    """
    构建多模态融合模型
    
    参数：
        eeg_model_path: EEG模型路径
        facial_model_path: 面部表情模型路径
        num_emotions: 最终情感类别数量
        
    返回：
        fusion_model: 融合模型
    """
    print("构建多模态融合模型...")
    
    # 加载预训练模型
    eeg_model = load_model(eeg_model_path)
    facial_model = load_model(facial_model_path)
    
    # 冻结基础模型权重
    eeg_model.trainable = False
    facial_model.trainable = False
    
    # 获取EEG模型特征输出层
    # 假设倒数第二层是特征层
    eeg_feature_layer = eeg_model.layers[-2].output
    
    # 获取面部表情模型特征输出层
    facial_feature_layer = facial_model.layers[-2].output
    
    # 创建多模态融合模型
    eeg_input = eeg_model.input
    facial_input = facial_model.input
    
    # 特征融合
    merged_features = concatenate([eeg_feature_layer, facial_feature_layer])
    
    # 添加融合层
    x = Dense(512, activation='relu')(merged_features)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_emotions, activation='softmax')(x)
    
    # 创建最终融合模型
    fusion_model = Model(inputs=[eeg_input, facial_input], outputs=output)
    
    # 编译模型
    fusion_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 打印模型摘要
    fusion_model.summary()
    
    return fusion_model

def prepare_multimodal_data(eeg_features, eeg_labels, facial_images, facial_labels):
    """
    准备多模态训练数据
    
    参数：
        eeg_features: EEG特征
        eeg_labels: EEG标签
        facial_images: 面部图像
        facial_labels: 面部表情标签
        
    返回：
        X_eeg: EEG训练数据
        X_facial: 面部表情训练数据
        y: 统一的标签
    """
    print("准备多模态训练数据...")
    
    # 面部表情标签到EEG标签的映射
    # 0: 愤怒, 1: 蔑视, 2: 厌恶, 3: 恐惧, 4: 高兴, 5: 悲伤, 6: 惊讶
    # ->
    # 0: 高兴, 1: 平静, 2: 悲伤
    facial_to_eeg_map = {
        0: 2,  # 愤怒 -> 悲伤
        1: 2,  # 蔑视 -> 悲伤
        2: 2,  # 厌恶 -> 悲伤
        3: 2,  # 恐惧 -> 悲伤
        4: 0,  # 高兴 -> 高兴
        5: 2,  # 悲伤 -> 悲伤
        6: 1,  # 惊讶 -> 平静
    }
    
    # 转换面部表情标签
    mapped_facial_labels = np.array([facial_to_eeg_map[label] for label in facial_labels])
    
    # 找到两个模态中共同的情感类别
    eeg_emotions = set(eeg_labels)
    facial_emotions = set(mapped_facial_labels)
    common_emotions = eeg_emotions.intersection(facial_emotions)
    
    # 筛选共同情感的样本
    eeg_indices = [i for i, label in enumerate(eeg_labels) if label in common_emotions]
    facial_indices = [i for i, label in enumerate(mapped_facial_labels) if label in common_emotions]
    
    # 确保数据集大小相同 (取最小值)
    min_size = min(len(eeg_indices), len(facial_indices))
    eeg_indices = eeg_indices[:min_size]
    facial_indices = facial_indices[:min_size]
    
    # 提取对应样本
    X_eeg = eeg_features[eeg_indices]
    X_facial = facial_images[facial_indices]
    y_eeg = eeg_labels[eeg_indices]
    y_facial = mapped_facial_labels[facial_indices]
    
    # 使用EEG标签作为最终标签 (因为已经映射过了)
    y = y_eeg
    
    print(f"多模态数据准备完成, 形状: EEG {X_eeg.shape}, 面部 {X_facial.shape}, 标签 {y.shape}")
    
    return X_eeg, X_facial, y

def train_fusion_model(fusion_model, X_eeg, X_facial, y, model_path, batch_size=32, epochs=50):
    """
    训练多模态融合模型
    
    参数：
        fusion_model: 融合模型
        X_eeg: EEG训练数据
        X_facial: 面部表情训练数据
        y: 标签
        model_path: 模型保存路径
        batch_size: 批量大小
        epochs: 训练轮数
        
    返回：
        history: 训练历史
    """
    print("训练多模态融合模型...")
    
    # 划分训练集和验证集
    X_eeg_train, X_eeg_val, X_facial_train, X_facial_val, y_train, y_val = train_test_split(
        X_eeg, X_facial, y, test_size=0.2, random_state=42
    )
    
    # 创建回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    # 训练模型
    history = fusion_model.fit(
        [X_eeg_train, X_facial_train], y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_eeg_val, X_facial_val], y_val),
        callbacks=callbacks
    )
    
    # 评估模型
    val_loss, val_acc = fusion_model.evaluate([X_eeg_val, X_facial_val], y_val)
    print(f"融合模型验证准确率: {val_acc:.4f}, 验证损失: {val_loss:.4f}")
    
    # 可视化训练历史
    plot_fusion_history(history)
    
    return history

def plot_fusion_history(history):
    """
    可视化融合模型训练历史
    
    参数：
        history: 训练历史对象
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制准确率
    ax1.plot(history.history['accuracy'], label='训练')
    ax1.plot(history.history['val_accuracy'], label='验证')
    ax1.set_title('融合模型准确率')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    
    # 绘制损失
    ax2.plot(history.history['loss'], label='训练')
    ax2.plot(history.history['val_loss'], label='验证')
    ax2.set_title('融合模型损失')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('./models/fusion_training_history.png')
    plt.show()

# 如果直接运行此文件
if __name__ == "__main__":
    # 设置路径
    EEG_MODEL_PATH = './models/eeg_model.h5'
    FACIAL_MODEL_PATH = './models/facial_model.h5'
    FUSION_MODEL_PATH = './models/fusion_model.h5'
    EEG_DATA_PATH = './data/processed/eeg_features.npy'
    EEG_LABEL_PATH = './data/processed/eeg_labels.npy'
    FACIAL_DATA_PATH = './data/processed/facial/X_train.npy'
    FACIAL_LABEL_PATH = './data/processed/facial/y_train.npy'
    
    # 确保模型目录存在
    os.makedirs('./models', exist_ok=True)
    
    # 加载数据
    eeg_features = np.load(EEG_DATA_PATH)
    eeg_labels = np.load(EEG_LABEL_PATH)
    facial_images = np.load(FACIAL_DATA_PATH)
    facial_labels = np.load(FACIAL_LABEL_PATH)
    
    # 构建融合模型
    fusion_model = build_fusion_model(EEG_MODEL_PATH, FACIAL_MODEL_PATH)
    
    # 准备多模态数据
    X_eeg, X_facial, y = prepare_multimodal_data(eeg_features, eeg_labels, facial_images, facial_labels)
    
    # 训练融合模型
    history = train_fusion_model(fusion_model, X_eeg, X_facial, y, FUSION_MODEL_PATH)
    
    print("多模态情感识别系统训练完成！")