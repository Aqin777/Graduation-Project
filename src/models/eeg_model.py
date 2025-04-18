# src/models/eeg_model.py
# EEG情感识别模型

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import pickle
import scipy.io as sio
import scipy.signal as signal
import mne
from sklearn.preprocessing import StandardScaler

# 全局参数
EEG_CLASSES = 3  # 高兴，平静，悲伤

def load_seed_dataset(data_path, n_subjects=15, n_sessions=3):
    """加载SEED数据集"""
    print("加载SEED数据集...")
    
    all_data = []
    all_labels = []
    
    # SEED数据集中的情感标签（每个会话15个试验，共3类情感）
    # 注意：SEED的标签为 1(positive), 0(neutral), -1(negative)
    session_labels = np.array([1, -1, 0, 1, 0, -1, -1, 0, 1, 1, 0, -1, 0, 1, -1])
    
    for subject in range(1, n_subjects + 1):
        for session in range(1, n_sessions + 1):
            # 构建文件路径（根据实际SEED数据集格式调整）
            subject_path = os.path.join(data_path, f'sub{subject:02d}', f'session{session}')
            
            # 加载EEG数据文件
            eeg_files = sorted(glob.glob(os.path.join(subject_path, '*.mat')))
            
            for trial_idx, eeg_file in enumerate(eeg_files):
                try:
                    # 加载.mat文件
                    mat_data = sio.loadmat(eeg_file)
                    
                    # 根据SEED数据集的具体结构提取EEG数据
                    # 注意：需要根据实际数据格式调整键名
                    eeg_trial = mat_data['eeg_data']  # 假设键名为'eeg_data'
                    
                    # 添加到总数据中
                    all_data.append(eeg_trial)
                    all_labels.append(session_labels[trial_idx])
                except Exception as e:
                    print(f"加载文件 {eeg_file} 时出错: {e}")
    
    # 将数据转换为numpy数组
    eeg_data = np.array(all_data)
    labels = np.array(all_labels)
    
    # 将SEED标签 (-1,0,1) 转换为 (0,1,2)，对应 (悲伤,平静,高兴)
    labels = labels + 1
    
    print(f"SEED数据加载完成，形状: {eeg_data.shape}, 标签形状: {labels.shape}")
    
    return eeg_data, labels

def load_deap_dataset(data_path, n_subjects=32):
    """加载DEAP数据集"""
    print("加载DEAP数据集...")
    
    all_data = []
    all_labels = []
    
    for subject in range(1, n_subjects + 1):
        # 构建文件路径（根据实际DEAP数据集格式调整）
        subject_file = os.path.join(data_path, f's{subject:02d}.dat')
        
        try:
            # DEAP数据集使用pickle格式存储
            with open(subject_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # 提取数据和标签
            data = subject_data['data']  # 形状为 (40, 40, 8064)
            labels = subject_data['labels']  # 形状为 (40, 4)
            
            # 只保留前32个通道（EEG通道）
            data = data[:, :32, :]
            
            # 添加到总数据中
            for trial in range(data.shape[0]):
                all_data.append(data[trial])
                
                # 映射情绪
                arousal = labels[trial, 0]  # 唤醒度
                valence = labels[trial, 1]  # 效价
                
                if valence > 5:  # 高效价
                    if arousal > 5:  # 高唤醒
                        emotion = 0  # 高兴
                    else:  # 低唤醒
                        emotion = 1  # 平静
                else:  # 低效价
                    if arousal > 5:  # 高唤醒
                        emotion = 2  # 悲伤
                    else:  # 低唤醒
                        emotion = 1  # 平静
                
                all_labels.append(emotion)
        
        except Exception as e:
            print(f"加载文件 {subject_file} 时出错: {e}")
    
    # 将数据转换为numpy数组
    eeg_data = np.array(all_data)
    labels = np.array(all_labels)
    
    print(f"DEAP数据加载完成，形状: {eeg_data.shape}, 标签形状: {labels.shape}")
    
    return eeg_data, labels

def preprocess_eeg(eeg_data, sfreq=128, l_freq=0.5, h_freq=45.0, notch_freq=50.0):
    """EEG信号预处理"""
    print("预处理EEG信号...")
    
    n_trials, n_channels, n_samples = eeg_data.shape
    processed_data = np.zeros_like(eeg_data)
    
    for i in range(n_trials):
        # 创建原始信息对象
        info = mne.create_info(n_channels, sfreq, ch_types='eeg')
        
        # 创建原始对象
        raw = mne.io.RawArray(eeg_data[i], info)
        
        # 应用带通滤波器
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
        
        # 应用陷波滤波器去除电源干扰
        raw.notch_filter(freqs=notch_freq)
        
        # 获取处理后的数据
        processed_data[i] = raw.get_data()
    
    # Z-score标准化（针对每个通道）
    for i in range(n_trials):
        for j in range(n_channels):
            processed_data[i, j] = (processed_data[i, j] - np.mean(processed_data[i, j])) / (np.std(processed_data[i, j]) + 1e-8)
    
    return processed_data

def extract_features(eeg_data, sfreq=128):
    """从EEG信号中提取特征"""
    print("提取EEG特征...")
    
    n_trials, n_channels, n_samples = eeg_data.shape
    
    # 定义频带
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # 特征：5个频带的功率 + 时域统计特征（均值、标准差、偏度、峰度）
    n_features = len(bands) + 4
    features = np.zeros((n_trials, n_channels, n_features))
    
    for i in range(n_trials):
        for j in range(n_channels):
            # 提取单个试验单个通道的信号
            x = eeg_data[i, j]
            
            # 计算频谱
            f, psd = signal.welch(x, fs=sfreq, nperseg=sfreq)
            
            # 计算各频带的功率
            feature_idx = 0
            for band_name, (low, high) in bands.items():
                # 找到频带对应的索引
                idx_band = np.logical_and(f >= low, f <= high)
                # 计算平均功率
                band_power = np.mean(psd[idx_band])
                # 存储特征
                features[i, j, feature_idx] = band_power
                feature_idx += 1
            
            # 时域统计特征
            features[i, j, feature_idx] = np.mean(x)  # 均值
            feature_idx += 1
            features[i, j, feature_idx] = np.std(x)   # 标准差
            feature_idx += 1
            features[i, j, feature_idx] = scipy.stats.skew(x)  # 偏度
            feature_idx += 1
            features[i, j, feature_idx] = scipy.stats.kurtosis(x)  # 峰度
    
    return features

def build_eeg_model(input_shape):
    """构建EEG情感识别模型 (CNN-LSTM结构)"""
    print("构建EEG模型...")
    
    model = Sequential([
        # CNN部分 - 学习空间特征
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # LSTM部分 - 学习时间特征
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        
        # 全连接层
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(EEG_CLASSES, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_eeg_model(X_train, y_train, X_test, y_test, model_path, batch_size=32, epochs=50):
    """训练EEG模型"""
    model = build_eeg_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # 创建回调函数
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    return model, history

def process_eeg_datasets(seed_path, deap_path, output_path, use_seed=True, use_deap=True):
    """处理SEED和DEAP数据集，提取特征并保存"""
    all_features = []
    all_labels = []
    
    # 处理SEED数据集
    if use_seed:
        # 加载数据
        seed_data, seed_labels = load_seed_dataset(seed_path)
        
        # 预处理
        seed_data_processed = preprocess_eeg(seed_data)
        
        # 特征提取
        seed_features = extract_features(seed_data_processed)
        
        # 重塑特征为2D形式 (n_trials, n_channels * n_features)
        n_trials, n_channels, n_features = seed_features.shape
        seed_features_2d = seed_features.reshape(n_trials, -1)
        
        all_features.append(seed_features_2d)
        all_labels.append(seed_labels)
    
    # 处理DEAP数据集
    if use_deap:
        # 加载数据
        deap_data, deap_labels = load_deap_dataset(deap_path)
        
        # 预处理
        deap_data_processed = preprocess_eeg(deap_data, sfreq=128)
        
        # 特征提取
        deap_features = extract_features(deap_data_processed, sfreq=128)
        
        # 重塑特征为2D形式 (n_trials, n_channels * n_features)
        n_trials, n_channels, n_features = deap_features.shape
        deap_features_2d = deap_features.reshape(n_trials, -1)
        
        all_features.append(deap_features_2d)
        all_labels.append(deap_labels)
    
    # 合并数据集
    if all_features:
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 保存处理后的数据
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        np.save(os.path.join(output_path, 'eeg_features.npy'), X_scaled)
        np.save(os.path.join(output_path, 'eeg_labels.npy'), y)
        
        # 保存特征标准化器
        with open(os.path.join(output_path, 'eeg_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"EEG特征处理完成，保存在 {output_path}")
        print(f"特征形状: {X_scaled.shape}, 标签形状: {y.shape}")
        
        return X_scaled, y
    
    return None, None

# 如果直接运行此文件
if __name__ == "__main__":
    # 设置路径
    SEED_PATH = './data/SEED/'
    DEAP_PATH = './data/DEAP/'
    OUTPUT_PATH = './data/processed/'
    MODEL_PATH = './models/eeg_model.h5'
    
    # 处理数据集
    X, y = process_eeg_datasets(SEED_PATH, DEAP_PATH, OUTPUT_PATH)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model, history = train_eeg_model(X_train, y_train, X_test, y_test, MODEL_PATH)
    
    print("EEG模型训练完成！")