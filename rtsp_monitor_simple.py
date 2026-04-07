# -*- coding:utf-8 -*-
"""
RTSP 流监控工具 (FFmpeg 简化版)
使用最简化的 FFmpeg 参数，确保兼容性
"""
import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
import logging
import warnings
import subprocess
import threading

# 屏蔽日志
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)


class RTSPMonitorSimple:
    """使用简化 FFmpeg 的 RTSP 监控类"""

    def __init__(self, rtsp_url: str, threshold: int = 1000,
                 check_interval: float = 1.0, save_screenshot: bool = True,
                 min_contour_area: float = 0.001, stable_frames: int = 2):
        self.rtsp_url = rtsp_url
        self.threshold = threshold
        self.check_interval = check_interval
        self.save_screenshot = save_screenshot
        self.min_contour_area = min_contour_area
        self.stable_frames = stable_frames

        self.cap = None
        self.last_frame = None
        self.screenshot_dir = 'screenshot'
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=20, detectShadows=False
        )

        # 创建截图目录
        if self.save_screenshot and not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            print(f"📁 已创建截图目录：{os.path.abspath(self.screenshot_dir)}")

    def connect(self) -> bool:
        """使用 OpenCV 连接 RTSP 流（FFmpeg 作为后端）"""
        try:
            print(f"\n📡 正在连接 RTSP 流...")
            print(f"   URL: {self.rtsp_url}")

            # 关闭旧连接
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass

            # 使用优化的 FFmpeg 参数
            ffmpeg_params = (
                '-rtsp_transport', 'tcp',
                '-buffer_size', '4194304',
                '-fflags', '+discardcorrupt',
                '-flags', 'low_delay',
                '-strict', 'experimental',
                '-probesize', '32',
                '-analyzeduration', '0'
            )
            
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = '|'.join(ffmpeg_params)

            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            print("   ⏳ 等待视频流初始化...")
            for i in range(20):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"✅ 连接成功！分辨率：{w}x{h}")
                    self.last_frame = frame.copy()
                    return True
                time.sleep(0.2)

            print("❌ 连接失败：无法读取首帧")
            return False

        except Exception as e:
            print(f"❌ 连接异常：{type(e).__name__}: {e}")
            return False

    def detect_motion(self, frame) -> tuple:
        """运动检测"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            fgmask = self.bg_subtractor.apply(gray, learningRate=0.01)
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = int(self.min_contour_area * gray.shape[0] * gray.shape[1])
            significant = [c for c in contours if cv2.contourArea(c) >= min_area]
            motion_ratio = sum(cv2.contourArea(c) for c in significant) / (gray.shape[0] * gray.shape[1])
            
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

    def save_screenshot_file(self, frame: np.ndarray, diff_pixels: int, count: int):
        """保存变化截图"""
        try:
            current_hour = datetime.now().strftime("%Y-%m-%d_%H")
            hour_dir = os.path.join(self.screenshot_dir, current_hour)
            if not os.path.exists(hour_dir):
                os.makedirs(hour_dir)
                print(f"   📁 创建目录：{os.path.abspath(hour_dir)}")

            timestamp = datetime.now().strftime("%M_%S_%f")[:-3]
            filename = f"{timestamp}_diff_{diff_pixels}_#{count}.jpg"
            filepath = os.path.join(hour_dir, filename)

            frame_copy = frame.copy()
            success = cv2.imwrite(filepath, frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                abs_path = os.path.abspath(filepath)
                print(f"   📸 已保存截图：{filename}")
                print(f"      路径：{abs_path}")
        except Exception as e:
            print(f"   ❌ 保存截图异常：{e}")

    def monitor(self):
        """监控主循环"""
        if not self.connect():
            return

        print(f"🔥 正在预热，跳过前 10 帧...")
        for i in range(10):
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                self.bg_subtractor.apply(gray, learningRate=1.0)
                self.last_frame = frame.copy()
        print(f"✅ 预热完成，开始监控...")

        print(f"\n🔍 开始监控...")
        print(f"📊 阈值：{self.threshold} | 间隔：{self.check_interval}s")
        print(f"🛡️ 已启用优化模式")
        print(f"⏹️ 按 Ctrl+C 停止\n")

        frame_count = 0
        motion_count = 0
        error_count = 0
        consecutive_errors = 0
        motion_confirm = 0
        last_stats_time = time.time()
        MAX_CONSECUTIVE_ERRORS = 30

        try:
            while True:
                current_time = time.time()
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    error_count += 1
                    consecutive_errors += 1
                    print(f"\r⚠️ 读取失败 (连续{consecutive_errors}次)       ", end='', flush=True)

                    if consecutive_errors % 20 == 0:
                        elapsed = consecutive_errors * 0.05
                        print(f"\n⚠️ 解码异常 {consecutive_errors} 次 (已容忍 {elapsed:.2f}s)", flush=True)

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(f"\n🔄 重新连接...")
                        time.sleep(1)
                        if self.connect():
                            consecutive_errors = 0
                            error_count = 0
                            motion_confirm = 0
                        else:
                            print("   重连失败，3 秒后重试...")
                            time.sleep(3)
                    else:
                        time.sleep(0.05)
                    continue

                consecutive_errors = 0
                frame_count += 1

                if frame_count <= 3 or frame_count % 30 == 0:
                    h, w = frame.shape[:2]
                    print(f"\r📊 帧：{w}x{h}, {frame.nbytes//1024}KB", flush=True)

                is_motion, ratio, diff_pixels = self.detect_motion(frame)
                
                if is_motion and motion_confirm >= self.stable_frames - 1:
                    print(f"   🔍 检测到轮廓：ratio={ratio*100:.2f}%")

                if is_motion:
                    motion_confirm += 1
                    if motion_confirm >= self.stable_frames:
                        motion_count += 1
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] ⚠️ 检测到变化！({ratio * 100:.1f}%) [#{motion_count}]")
                        print(f"   📊 差异像素：{diff_pixels}, 阈值：{self.threshold}")

                        if self.save_screenshot:
                            self.save_screenshot_file(frame, diff_pixels, motion_count)
                        motion_confirm = 0
                else:
                    motion_confirm = 0

                if frame is not None and frame.size > 0:
                    self.last_frame = frame.copy()

                if frame_count % 60 == 0:
                    loop_time = (time.time() - current_time) * 1000
                    print(f"\n⏱️  单帧耗时：{loop_time:.1f}ms", flush=True)

                time.sleep(self.check_interval)

                if current_time - last_stats_time >= 120:
                    success_rate = (frame_count / (frame_count + error_count)) * 100
                    print(f"   📈 已处理 {frame_count} 帧 | 成功率：{success_rate:.1f}%")
                    last_stats_time = current_time

        except KeyboardInterrupt:
            print(f"\n\n⏹️ 停止监控")
        finally:
            total = frame_count + error_count
            success_rate = (frame_count / total * 100) if total > 0 else 0
            print(f"📊 统计：总帧数={frame_count}, 变化次数={motion_count}, 错误={error_count}, 成功率={success_rate:.1f}%")
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("RTSP 流监控工具 (优化版)")
        print("=" * 60)
        print("\n用法:")
        print("  python rtsp_monitor_simple.py <RTSP 地址> [阈值] [间隔]")
        print("=" * 60)
        sys.exit(1)

    rtsp_url = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    monitor = RTSPMonitorSimple(rtsp_url, threshold, interval)
    monitor.monitor()


if __name__ == "__main__":
    main()
