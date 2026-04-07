# -*- coding:utf-8 -*-
"""
RTSP 流监控工具 (FFmpeg Pipe 版)
使用 FFmpeg 进程管道读取视频流，比 OpenCV 直接读取更稳定

模块拆分：
- stream_capture: 负责 RTSP 流捕获和帧获取
- motion_detector: 负责运动检测和画面分析
- 本文件：封装类，供 rtsp_monitor_main.py 使用（只负责抓图）
"""
import sys
import os
import time
from datetime import datetime
import logging
import warnings
import cv2
import numpy as np

# 导入自定义模块
from stream_capture import StreamCapture
from motion_detector import MotionDetector

# 屏蔽日志
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)


class RTSPMonitorFFmpeg:
    """使用 FFmpeg pipe 的 RTSP 监控类（主流程）"""

    def __init__(self, rtsp_url: str):
        """
        初始化 RTSP 监控器

        Args:
            rtsp_url: RTSP 流地址
        """
        self.rtsp_url = rtsp_url

        # 初始化流捕获器（设置 FPS=1，每秒1帧）
        self.stream_capture = StreamCapture(rtsp_url, fps=1, quality=5)

    def connect(self) -> bool:
        """连接到 RTSP 流"""
        return self.stream_capture.connect()

    def get_frame(self, max_retries: int = 1):
        """
        获取一帧

        Args:
            max_retries: 最大重试次数

        Returns:
            tuple: (success, frame)
        """
        return self.stream_capture.get_frame(max_retries)
