# src/models/facial_model.py
# 面部表情识别模型

import numpy as np
import os
import glob
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 全局参数
FACIAL_CLASSES = 7  # 愤怒，蔑视，厌恶，恐惧，高兴，悲伤，惊讶

def load_ck_plus_dataset(data_path, image_size=(48, 48)):
    """加载CK+数据集"""
    print("加载CK+数据集...")
    
    # CK+数据集中的7种情感标签
    emotion_map = {
        'anger': 0,      # 愤怒
        'contempt': 1,   # 蔑视
        'disgust': 2,    # 厌恶
        'fear': 3,       # 恐惧
        'happy': 4,      # 高兴
        'sadness': 5,    # 悲伤
        'surprise': 6    # 惊讶
    }
    
    images = []
    labels = []
    
    # 遍历所有情感文件夹
    for emotion, label in emotion_map.items():
        emotion_dir = os.path.join(data_path, emotion)
        
        # 检查目录是否存在
        if not os.path.exists(emotion_dir):
            print(f"警告: 目录 {emotion_dir} 不存在")
            continue
        
        # 获取所有图像文件
        image_files = glob.glob(os.path.join(emotion_dir, '*.png')) + \
                      glob.glob(os.path.join(emotion_dir, '*.jpg')) + \
                      glob.glob(os.path.join(emotion_dir, '*.jpeg'))
        
        for image_file in image_files:
            try:
                # 读取图像
                img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                
                # 调整大小
                img = cv2.resize(img, image_size)
                
                # 添加到数据集
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
    
    # 转换为numpy数组
    images = np.array(images)
    labels = np.array(labels)
    
    # 添加通道维度 (n_samples, height, width) -> (n_samples, height, width, 1)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    
    print(f"CK+数据集加载完成，形状: {images.shape}, 标签形状: {labels.shape}")
    
    return images, labels

def detect_and_align_faces(images, face_cascade=None):
    """检测和对齐面部"""
    print("检测和对齐面部...")
    
    if face_cascade is None:
        # 使用OpenCV预训练的人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    aligned_faces = []
    
    for img in images:
        # 确保图像是灰度图像
        if len(img.shape) == 3 and img.shape[2] == 1:
            gray = img.squeeze()
        elif len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # 取第一个检测到的人脸
            (x, y, w, h) = faces[0]
            
            # 提取人脸区域
            face = gray[y:y+h, x:x+w]
            
            # 调整大小为统一尺寸
            face = cv2.resize(face, (48, 48))
            
            # 添加到结果列表
            aligned_faces.append(face)
        else:
            # 如果没有检测到人脸，使用原始图像
            aligned_faces.append(cv2.resize(gray, (48, 48)))
    
    # 转换为numpy数组并添加通道维度
    aligned_faces = np.array(aligned_faces).reshape(-1, 48, 48, 1)
    
    return aligned_faces

def create_data_generator():
    """创建数据增强生成器"""
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    return datagen

def build_facial_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    """构建面部表情识别CNN模型"""
    print("构建面部表情CNN模型...")
    
    model = Sequential([
        # 第一个卷积块
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),
        
        # 第二个卷积块
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),
        
        # 第三个卷积块
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),
        
        # 全连接层
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    # 打印模型摘要
    model.summary()
    
    return model

def train_facial_model(X_train, y_train, X_val, y_val, model_path, batch_size=32, epochs=50):
    """训练面部表情识别模型"""
    print("训练面部表情识别模型...")
    
    # 创建模型
    model = build_facial_cnn_model()
    
    # 创建数据增强生成器
    datagen = create_data_generator()
    datagen.fit(X_train)
    
    # 创建回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    # 训练模型
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # 评估模型
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"验证准确率: {val_acc:.4f}, 验证损失: {val_loss:.4f}")
    
    # 可视化训练历史
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """可视化训练历史"""
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制准确率
    ax1.plot(history.history['accuracy'], label='训练')
    ax1.plot(history.history['val_accuracy'], label='验证')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    
    # 绘制损失
    ax2.plot(history.history['loss'], label='训练')
    ax2.plot(history.history['val_loss'], label='验证')
    ax2.set_title('模型损失')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('./models/facial_training_history.png')
    plt.show()

def process_facial_dataset(data_path, output_path, test_size=0.2):
    """处理面部表情数据集"""
    # 创建输出目录
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 加载数据集
    images, labels = load_ck_plus_dataset(data_path)
    
    # 数据预处理
    X = images / 255.0  # 归一化像素值
    y = labels
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 保存处理后的数据
    np.save(os.path.join(output_path, 'X_train.npy'), X_train)
    np.save(os.path.join(output_path, 'X_test.npy'), X_test)
    np.save(os.path.join(output_path, 'y_train.npy'), y_train)
    np.save(os.path.join(output_path, 'y_test.npy'), y_test)
    
    print(f"面部表情数据处理完成，保存在 {output_path}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# 如果直接运行此文件
if __name__ == "__main__":
    # 设置路径
    CK_PATH = './data/CK+/'
    OUTPUT_PATH = './data/processed/facial/'
    MODEL_PATH = './models/facial_model.h5'
    
    # 处理数据集
    X_train, X_test, y_train, y_test = process_facial_dataset(CK_PATH, OUTPUT_PATH)
    
    # 训练模型
    model, history = train_facial_model(X_train, y_train, X_test, y_test, MODEL_PATH)
    
    print("面部表情识别模型训练完成！")