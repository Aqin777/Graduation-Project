# Web应用

本目录包含情绪识别冥想辅助系统的Web应用界面和交互功能。

## 目录结构

- `static/` - 静态资源文件夹，包含CSS、JavaScript和图像资源
- `templates/` - HTML模板文件夹
- `app.py` - Web应用程序入口文件

## 功能描述

Web应用提供以下功能：
- 用户注册和登录
- 冥想会话启动和控制界面
- 情绪状态可视化显示
- 冥想内容选择和自定义
- 历史数据查看和分析
- 用户反馈收集

## 技术栈

本Web应用基于以下技术构建：
- 后端框架：Flask/Django
- 前端技术：HTML, CSS, JavaScript
- 数据可视化：D3.js
- WebSocket：用于实时情绪监测和内容调整

## 开发和调试

### 本地开发环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 启动Web应用
python app.py