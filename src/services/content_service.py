# src/services/content_service.py
# 冥想内容服务

import numpy as np
import os
import json
import time
import random
import pygame
import threading

class MeditationContentService:
    """冥想内容服务类，提供基于情感状态的冥想内容推荐"""
    
    def __init__(self):
        """初始化冥想内容服务"""
        # 冥想内容库
        self.meditation_library = {
            '高兴': {
                'music': ['happy_meditation1.mp3', 'happy_meditation2.mp3'],
                'visual': ['happy_visual1.jpg', 'happy_visual2.jpg'],
                'guidance': [
                    "保持这种愉快的心情，让我们一起呼吸。吸气，感受快乐，呼气，传播快乐。",
                    "感受这份喜悦，让它流经你的全身。每一次呼吸都加深这种感觉。",
                    "带着微笑冥想，让积极的能量充满全身。保持平静，享受当下的快乐。"
                ]
            },
            '平静': {
                'music': ['calm_meditation1.mp3', 'calm_meditation2.mp3'],
                'visual': ['calm_visual1.jpg', 'calm_visual2.jpg'],
                'guidance': [
                    "保持这种平静的状态，深呼吸。吸气，感受平静，呼气，释放所有紧张。",
                    "感受每一次呼吸如何使你更加平静。让思绪像云一样漂浮而过。",
                    "专注于呼吸的节奏，感受身体中的平静能量。允许自己完全放松。"
                ]
            },
            '悲伤': {
                'music': ['sad_meditation1.mp3', 'sad_meditation2.mp3'],
                'visual': ['sad_visual1.jpg', 'sad_visual2.jpg'],
                'guidance': [
                    "接受当前的情绪，不要抵抗。吸气，感受情绪，呼气，给自己一些空间。",
                    "让每一次呼吸都带走一点痛苦。记住，所有情绪都是暂时的。",
                    "温柔地关注你的呼吸，允许自己感受。随着每一次呼吸，找到内心的平静。"
                ]
            }
        }
        
        # 资源路径
        self.music_path = './resources/music/'
        self.visual_path = './resources/visual/'
        
        # 确保资源目录存在
        os.makedirs(self.music_path, exist_ok=True)
        os.makedirs(self.visual_path, exist_ok=True)
        
        # 初始化pygame用于音频播放
        pygame.mixer.init()
        self.current_music = None
        self.music_playing = False
        
        # 冥想会话状态
        self.session_active = False
        self.session_start_time = None
        self.session_duration = 0  # 默认单位：分钟
        self.emotion_records = []
        
        print("冥想内容服务初始化完成")
    
    def get_meditation_content(self, emotion=None):
        """
        获取适合特定情感状态的冥想内容
        
        参数：
            emotion: 情感状态（可选）
            
        返回：
            content: 冥想内容字典
        """
        if emotion is None or emotion not in self.meditation_library:
            emotion = '平静'  # 默认为平静
        
        library = self.meditation_library[emotion]
        
        # 随机选择内容
        content = {
            'music': random.choice(library['music']),
            'visual': random.choice(library['visual']),
            'guidance': random.choice(library['guidance'])
        }
        
        return content
    
    def play_meditation_music(self, music_file=None, emotion=None):
        """
        播放冥想音乐
        
        参数：
            music_file: 音乐文件名（可选）
            emotion: 情感状态（可选）
            
        返回：
            success: 是否成功播放
        """
        if music_file is None:
            if emotion is None or emotion not in self.meditation_library:
                emotion = '平静'  # 默认为平静
            
            # 随机选择一个音乐文件
            music_file = random.choice(self.meditation_library[emotion]['music'])
        
        # 构建完整路径
        music_path = os.path.join(self.music_path, music_file)
        
        # 检查文件是否存在
        if not os.path.exists(music_path):
            print(f"音乐文件 {music_path} 不存在")
            return False
        
        # 停止当前播放的音乐
        if self.music_playing:
            pygame.mixer.music.stop()
        
        # 播放新音乐
        try:
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.play(-1)  # -1表示循环播放
            self.music_playing = True
            self.current_music = music_file
            print(f"正在播放：{music_file}")
            return True
        except Exception as e:
            print(f"播放音乐时出错：{e}")
            return False
    
    def stop_music(self):
        """停止播放音乐"""
        if self.music_playing:
            pygame.mixer.music.stop()
            self.music_playing = False
            self.current_music = None
            print("音乐已停止")
    
    def start_meditation_session(self, duration=10):
        """
        开始冥想会话
        
        参数：
            duration: 会话时长（分钟）
        """
        if self.session_active:
            print("已有会话正在进行中")
            return False
        
        self.session_active = True
        self.session_start_time = time.time()
        self.session_duration = duration
        self.emotion_records = []
        
        print(f"开始冥想会话，时长：{duration}分钟")
        
        # 启动会话监控线程
        threading.Thread(target=self._session_monitor, daemon=True).start()
        
        return True
    
    def _session_monitor(self):
        """会话监控线程"""
        while self.session_active:
            # 检查会话是否结束
            elapsed = (time.time() - self.session_start_time) / 60  # 转换为分钟
            if elapsed >= self.session_duration:
                self.end_meditation_session()
                break
            
            # 每30秒检查一次
            time.sleep(30)
    
    def end_meditation_session(self):
        """结束冥想会话"""
        if not self.session_active:
            print("没有正在进行的会话")
            return False
        
        self.session_active = False
        elapsed = (time.time() - self.session_start_time) / 60  # 转换为分钟
        
        print(f"冥想会话结束，持续时间：{elapsed:.2f}分钟")
        
        # 停止音乐
        if self.music_playing:
            self.stop_music()
        
        # 生成会话报告
        self._generate_session_report()
        
        return True
    
    def record_emotion(self, emotion, timestamp=None):
        """
        记录冥想过程中的情感变化
        
        参数：
            emotion: 情感状态
            timestamp: 时间戳（可选）
        """
        if not self.session_active:
            return False
        
        if timestamp is None:
            timestamp = time.time()
        
        self.emotion_records.append((emotion, timestamp))
        return True
    
    def _generate_session_report(self):
        """
        生成冥想会话报告
        
        返回：
            report: 会话报告字典
        """
        # 计算情绪分布
        emotion_counts = {}
        for emotion, _ in self.emotion_records:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = max(1, sum(emotion_counts.values()))  # 避免除以零
        emotion_percentages = {e: (c / total) * 100 for e, c in emotion_counts.items()}
        
        # 创建报告
        report = {
            "session_duration": (time.time() - self.session_start_time) / 60,
            "emotion_distribution": emotion_percentages,
            "initial_emotion": self.emotion_records[0][0] if self.emotion_records else None,
            "final_emotion": self.emotion_records[-1][0] if self.emotion_records else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存报告
        report_dir = "./reports"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"meditation_session_{int(time.time())}.json")
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)
        
        print(f"会话报告已生成：{report_file}")
        
        return report
    
    def get_personalized_guidance(self, emotion=None):
        """
        获取个性化冥想指导语
        
        参数：
            emotion: 情感状态（可选）
            
        返回：
            guidance: 指导语
        """
        content = self.get_meditation_content(emotion)
        return content['guidance']

# 如果直接运行此文件
if __name__ == "__main__":
    # 测试冥想内容服务
    content_service = MeditationContentService()
    
    # 测试获取冥想内容
    content = content_service.get_meditation_content('高兴')
    print("冥想内容示例:")
    print(f"音乐: {content['music']}")
    print(f"视觉: {content['visual']}")
    print(f"引导语: {content['guidance']}")
    
    # 测试冥想会话
    content_service.start_meditation_session(duration=0.1)  # 6秒
    content_service.record_emotion('高兴')
    time.sleep(3)
    content_service.record_emotion('平静')
    time.sleep(3)
    content_service.end_meditation_session()