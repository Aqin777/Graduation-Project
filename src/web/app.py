# src/web/app.py
# 冥想辅助系统Web应用

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
import time
import threading
import base64
import json
from datetime import datetime

# 导入服务模块
import sys
sys.path.append('../')
from services.emotion_service import EmotionRecognitionService
from services.content_service import MeditationContentService

app = Flask(__name__)

# 配置路径
MODEL_PATH = '../../models/fusion_model.h5'
EEG_SCALER_PATH = '../../data/processed/eeg_scaler.pkl'
RESOURCES_PATH = '../../resources'
REPORTS_PATH = '../../reports'

# 初始化服务
emotion_service = None
content_service = None

def init_services():
    """初始化服务"""
    global emotion_service, content_service
    
    # 确保路径存在
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # 初始化情感识别服务
    try:
        emotion_service = EmotionRecognitionService(
            fusion_model_path=MODEL_PATH,
            eeg_scaler_path=EEG_SCALER_PATH
        )
    except Exception as e:
        print(f"初始化情感识别服务失败: {e}")
        emotion_service = None
    
    # 初始化冥想内容服务
    content_service = MeditationContentService()

# 路由
@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    """分析情感"""
    if emotion_service is None:
        return jsonify({'error': '情感识别服务未初始化'}), 500
    
    # 获取面部图像数据
    image_data = request.json.get('image')
    
    if image_data:
        # 解码Base64图像
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # 分析情感
        emotion, confidence = emotion_service.analyze_emotion(None, image)
        
        # 记录情感
        if content_service.session_active:
            content_service.record_emotion(emotion)
        
        # 获取冥想内容
        content = content_service.get_meditation_content(emotion)
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'content': content
        })
    
    return jsonify({'error': '未提供图像数据'}), 400

@app.route('/session/start', methods=['POST'])
def start_session():
    """开始冥想会话"""
    if content_service is None:
        return jsonify({'error': '冥想内容服务未初始化'}), 500
    
    duration = request.json.get('duration', 10)
    
    if content_service.start_meditation_session(duration):
        return jsonify({'status': 'success', 'message': f'已开始{duration}分钟的冥想会话'})
    else:
        return jsonify({'error': '无法开始冥想会话'}), 400

@app.route('/session/end', methods=['POST'])
def end_session():
    """结束冥想会话"""
    if content_service is None:
        return jsonify({'error': '冥想内容服务未初始化'}), 500
    
    if content_service.end_meditation_session():
        return jsonify({'status': 'success', 'message': '冥想会话已结束'})
    else:
        return jsonify({'error': '无法结束冥想会话'}), 400

@app.route('/music/play', methods=['POST'])
def play_music():
    """播放音乐"""
    if content_service is None:
        return jsonify({'error': '冥想内容服务未初始化'}), 500
    
    music_file = request.json.get('music_file')
    emotion = request.json.get('emotion')
    
    if content_service.play_meditation_music(music_file, emotion):
        return jsonify({'status': 'success', 'message': '音乐已开始播放'})
    else:
        return jsonify({'error': '无法播放音乐'}), 400

@app.route('/music/stop', methods=['POST'])
def stop_music():
    """停止音乐"""
    if content_service is None:
        return jsonify({'error': '冥想内容服务未初始化'}), 500
    
    content_service.stop_music()
    return jsonify({'status': 'success', 'message': '音乐已停止'})

@app.route('/reports')
def list_reports():
    """列出所有报告"""
    if not os.path.exists(REPORTS_PATH):
        return jsonify([])
    
    reports = []
    for filename in os.listdir(REPORTS_PATH):
        if filename.startswith('meditation_session_') and filename.endswith('.json'):
            file_path = os.path.join(REPORTS_PATH, filename)
            try:
                with open(file_path, 'r') as f:
                    report = json.load(f)
                    report['filename'] = filename
                    reports.append(report)
            except Exception as e:
                print(f"读取报告{filename}时出错: {e}")
    
    # 按时间戳排序
    reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify(reports)

@app.route('/reports/<filename>')
def get_report(filename):
    """获取特定报告"""
    file_path = os.path.join(REPORTS_PATH, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': '报告不存在'}), 404
    
    try:
        with open(file_path, 'r') as f:
            report = json.load(f)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': f'读取报告时出错: {e}'}), 500

@app.route('/resources/<path:path>')
def get_resource(path):
    """获取资源文件"""
    return send_from_directory(RESOURCES_PATH, path)

# 启动应用
if __name__ == '__main__':
    # 初始化服务
    init_services()
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)