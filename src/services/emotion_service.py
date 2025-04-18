# src/services/emotion_service.py
# 情感识别服务

import numpy as np
import cv2
import pickle
import time
from tensorflow.keras.models import load_model

class EmotionRecognitionService:
    """情感识别服务类，提供基于EEG和面部表情的情感分析"""
    
    def __init__(self, fusion_model_path, eeg_scaler_path=None):
        """
        初始化情感识别服务
        
        参数：
            fusion_model_path: 融合模型路径
            eeg_scaler_path: EEG特征标准化器路径
        """
        # 加载融合模型
        self.fusion_model = load_model(fusion_model_path)
        
        # 加载EEG特征标准化器
        if eeg_scaler_path:
            with open(eeg_scaler_path, 'rb') as f:
                self.eeg_scaler = pickle.load(f)
        else:
            self.eeg_scaler = None
        
        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 情感标签
        self.emotion_labels = {
            0: '高兴',
            1: '平静',
            2: '悲伤'
        }
        
        # 当前情感状态
        self.current_emotion = None
        self.emotion_history = []
        
        print("情感识别服务初始化完成")
    
    def analyze_emotion(self, eeg_data=None, facial_image=None):
        """
        分析用户情感状态
        
        参数：
            eeg_data: EEG信号数据
            facial_image: 面部图像
            
        返回：
            emotion: 情感标签
            confidence: 置信度
        """
        # 如果没有提供任何数据，返回None
        if eeg_data is None and facial_image is None:
            return None, 0.0
        
        # 如果只有一种模态数据
        if eeg_data is None:
            # 只使用面部表情
            return self._predict_from_facial(facial_image)
        elif facial_image is None:
            # 只使用EEG
            return self._predict_from_eeg(eeg_data)
        
        # 使用多模态融合预测
        # 预处理EEG数据
        eeg_features = self._preprocess_eeg(eeg_data)
        
        # 预处理面部图像
        facial_features = self._preprocess_facial(facial_image)
        
        # 模型预测
        predictions = self.fusion_model.predict([eeg_features, facial_features])[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        
        # 更新情感历史
        self.current_emotion = self.emotion_labels[emotion_idx]
        self.emotion_history.append((self.current_emotion, time.time()))
        
        return self.current_emotion, confidence
    
    def _predict_from_eeg(self, eeg_data):
        """从EEG数据预测情感"""
        # 预处理EEG数据
        eeg_features = self._preprocess_eeg(eeg_data)
        
        # 创建占位符面部数据（零矩阵）
        dummy_facial = np.zeros((1, 48, 48, 1))
        
        # 模型预测
        predictions = self.fusion_model.predict([eeg_features, dummy_facial])[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        
        # 更新情感历史
        self.current_emotion = self.emotion_labels[emotion_idx]
        self.emotion_history.append((self.current_emotion, time.time()))
        
        return self.current_emotion, confidence
    
    def _predict_from_facial(self, facial_image):
        """从面部图像预测情感"""
        # 预处理面部图像
        facial_features = self._preprocess_facial(facial_image)
        
        # 创建占位符EEG数据（零矩阵）
        dummy_eeg = np.zeros((1, 62 * 9))  # 假设EEG特征形状为 (62通道 * 9特征)
        
        # 模型预测
        predictions = self.fusion_model.predict([dummy_eeg, facial_features])[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        
        # 更新情感历史
        self.current_emotion = self.emotion_labels[emotion_idx]
        self.emotion_history.append((self.current_emotion, time.time()))
        
        return self.current_emotion, confidence
    
    def _preprocess_eeg(self, eeg_data):
        """预处理EEG数据"""
        # 应用标准化
        if self.eeg_scaler:
            eeg_features = self.eeg_scaler.transform(eeg_data.reshape(1, -1))
        else:
            eeg_features = eeg_data.reshape(1, -1)
            
        return eeg_features
    
    def _preprocess_facial(self, facial_image):
        """预处理面部图像"""
        # 灰度转换
        if len(facial_image.shape) == 3 and facial_image.shape[2] == 3:
            gray = cv2.cvtColor(facial_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = facial_image
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # 取第一个检测到的人脸
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
        else:
            # 如果没有检测到人脸，使用整个图像
            face = gray
        
        # 调整大小
        face = cv2.resize(face, (48, 48))
        
        # 归一化
        face = face / 255.0
        
        # 添加批次和通道维度
        face = face.reshape(1, 48, 48, 1)
        
        return face
    
    def get_emotion_history(self):
        """获取情感历史记录"""
        return self.emotion_history
    
    def get_dominant_emotion(self, time_window=60):
        """
        获取指定时间窗口内的主导情感
        
        参数：
            time_window: 时间窗口（秒）
            
        返回：
            dominant_emotion: 主导情感
        """
        if not self.emotion_history:
            return None
        
        # 计算当前时间
        current_time = time.time()
        
        # 筛选时间窗口内的情感记录
        recent_emotions = [emotion for emotion, timestamp in self.emotion_history 
                          if current_time - timestamp <= time_window]
        
        # 统计情感出现次数
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 找出出现次数最多的情感
        if emotion_counts:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            return dominant_emotion
        else:
            return None

# 如果直接运行此文件
if __name__ == "__main__":
    # 测试情感识别服务
    emotion_service = EmotionRecognitionService(
        fusion_model_path='./models/fusion_model.h5',
        eeg_scaler_path='./data/processed/eeg_scaler.pkl'
    )
    
    # 创建模拟数据
    dummy_eeg = np.random.randn(62 * 9)
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 测试情感分析
    emotion, confidence = emotion_service.analyze_emotion(dummy_eeg, dummy_image)
    print(f"检测到的情感: {emotion}, 置信度: {confidence:.2f}")