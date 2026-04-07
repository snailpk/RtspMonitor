# -*- coding:utf-8 -*-
"""
运动检测模块
使用背景减除法进行运动检测和变化分析
"""
import os
import cv2
import numpy as np
from datetime import datetime


class MotionDetector:
    """运动检测器类"""

    def __init__(self, threshold: int = 1000, min_contour_area: float = 0.001,
                 stable_frames: int = 2, save_screenshot: bool = True):
        """
        初始化运动检测器
        
        Args:
            threshold: 像素差异阈值，默认 1000
            min_contour_area: 最小轮廓面积比例，默认 0.001（0.1%）
            stable_frames: 确认运动需要的稳定帧数，默认 2
            save_screenshot: 是否保存截图，默认 True
        """
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.stable_frames = stable_frames
        self.save_screenshot = save_screenshot
        
        # 背景减除器
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=20, detectShadows=False
        )
        
        # 上一帧
        self.last_frame = None
        
        # 截图目录
        self.screenshot_dir = 'screenshot'
        if self.save_screenshot and not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            print(f"📁 已创建截图目录：{os.path.abspath(self.screenshot_dir)}")

    def detect(self, frame) -> tuple:
        """
        检测运动
        
        Args:
            frame: 输入帧
            
        Returns:
            tuple: (is_motion, motion_ratio, diff_pixels)
                - is_motion: 是否检测到运动
                - motion_ratio: 运动区域比例
                - diff_pixels: 差异像素数量
        """
        try:
            # 转灰度图 + 高斯模糊
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # 背景减除
            fgmask = self.bg_subtractor.apply(gray, learningRate=0.01)

            # 二值化
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 过滤小轮廓
            min_area = int(self.min_contour_area * gray.shape[0] * gray.shape[1])
            significant = [c for c in contours if cv2.contourArea(c) >= min_area]

            # 计算变化比例
            motion_ratio = sum(cv2.contourArea(c) for c in significant) / (gray.shape[0] * gray.shape[1])

            # 计算差异像素
            diff_pixels = 0
            if self.last_frame is not None and self.last_frame.shape == frame.shape:
                gray1 = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray1, gray)
                _, diff_bin = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                diff_pixels = cv2.countNonZero(diff_bin)

            return len(significant) > 0, motion_ratio, diff_pixels

        except Exception as e:
            print(f"⚠️ 运动检测异常：{e}")
            return False, 0.0, 0

    def update_last_frame(self, frame):
        """
        更新上一帧
        
        Args:
            frame: 当前帧
        """
        if frame is not None and frame.size > 0:
            self.last_frame = frame.copy()

    def save_screenshot_file(self, frame: np.ndarray, diff_pixels: int, count: int):
        """
        保存变化截图
        
        Args:
            frame: 帧图像
            diff_pixels: 差异像素数
            count: 截图计数
        """
        try:
            current_hour = datetime.now().strftime("%Y-%m-%d_%H")
            hour_dir = os.path.join(self.screenshot_dir, current_hour)
            if not os.path.exists(hour_dir):
                os.makedirs(hour_dir)
                print(f"   📁 创建目录：{os.path.abspath(hour_dir)}")

            timestamp = datetime.now().strftime("%M_%S_%f")[:-3]
            filename = f"{timestamp}_diff_{diff_pixels}_#{count}.jpg"
            filepath = os.path.join(hour_dir, filename)

            # 复制帧避免影响原始数据
            frame_copy = frame.copy()
            success = cv2.imwrite(filepath, frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                abs_path = os.path.abspath(filepath)
                print(f"   📸 已保存截图：{filename}")
                print(f"      路径：{abs_path}")
            else:
                print(f"   ❌ 保存失败：{filename}")
        except Exception as e:
            print(f"   ❌ 保存截图异常：{e}")

    def warmup(self, get_frame_func, frames_count: int = 10):
        """
        预热背景模型
        
        Args:
            get_frame_func: 获取帧的函数
            frames_count: 预热的帧数，默认 10
        """
        print(f"🔥 正在预热，跳过前 {frames_count} 帧...")
        for i in range(frames_count):
            ret, frame = get_frame_func()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                self.bg_subtractor.apply(gray, learningRate=1.0)
                self.update_last_frame(frame)
        print(f"✅ 预热完成，开始监控...")
