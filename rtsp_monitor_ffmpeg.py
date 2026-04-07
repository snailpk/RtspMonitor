# -*- coding:utf-8 -*-
"""
RTSP 流监控工具 (FFmpeg Pipe 版)
使用 FFmpeg 进程管道读取视频流，比 OpenCV 直接读取更稳定

模块拆分：
- stream_capture: 负责 RTSP 流捕获和帧获取
- motion_detector: 负责运动检测和画面分析
- 本文件：主流程协调
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

    def __init__(self, rtsp_url: str, threshold: int = 1000,
                 check_interval: float = 1.0, save_screenshot: bool = True,
                 min_contour_area: float = 0.001, stable_frames: int = 2):
        """
        初始化 RTSP 监控器
        
        Args:
            rtsp_url: RTSP 流地址
            threshold: 像素差异阈值，默认 1000
            check_interval: 检测间隔秒数，默认 1.0
            save_screenshot: 是否保存截图，默认 True
            min_contour_area: 最小轮廓面积比例，默认 0.001
            stable_frames: 确认运动需要的稳定帧数，默认 2
        """
        self.rtsp_url = rtsp_url
        self.threshold = threshold
        self.check_interval = check_interval
        self.save_screenshot = save_screenshot
        self.min_contour_area = min_contour_area
        self.stable_frames = stable_frames

        # 初始化流捕获器和运动检测器（设置 FPS=1，每秒1帧）
        self.stream_capture = StreamCapture(rtsp_url, fps=1, quality=5)
        self.motion_detector = MotionDetector(
            threshold=threshold,
            min_contour_area=min_contour_area,
            stable_frames=stable_frames,
            save_screenshot=save_screenshot
        )

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

    def detect_motion(self, frame) -> tuple:
        """
        检测运动
        
        Args:
            frame: 输入帧
            
        Returns:
            tuple: (is_motion, motion_ratio, diff_pixels)
        """
        return self.motion_detector.detect(frame)

    def save_screenshot_file(self, frame, diff_pixels: int, count: int):
        """
        保存变化截图
            
        Args:
            frame: 帧图像
            diff_pixels: 差异像素数
            count: 截图计数
        """
        self.motion_detector.save_screenshot_file(frame, diff_pixels, count)

    def monitor(self):
        """监控主循环（简化版 - 仅测试取流）"""
        if not self.connect():
            return

        print(f"\n🔍 开始监控 (仅测试取流)...")
        print(f"📊 间隔：{self.check_interval}s")
        print(f"🛡️ 已启用 FFmpeg Pipe 模式（最稳定）")
        print(f"⏹️ 按 Ctrl+C 停止\n")

        # 统计变量
        frame_count = 0
        error_count = 0
        consecutive_errors = 0
        last_stats_time = time.time()
        last_frame_time = time.time()  # 上一帧的时间戳
        last_sample_time = 0  # 上次采样的时间戳（用于控制每秒1帧）

        # 错误容忍度
        MAX_CONSECUTIVE_ERRORS = 10  # 降低阈值，更快重启

        try:
            while True:
                current_time = time.time()

                # 读取帧
                ret, frame = self.get_frame(max_retries=5)

                if not ret or frame is None:
                    error_count += 1
                    consecutive_errors += 1

                    # 每次失败都打印
                    print(f"\r⚠️ 读取失败 (连续{consecutive_errors}次)       ", end='', flush=True)

                    # 检查 FFmpeg 进程状态（每次失败都检查）
                    if self.stream_capture.process and self.stream_capture.process.poll() is not None:
                        returncode = self.stream_capture.process.returncode
                        print(f"\n   ❌ FFmpeg 进程已退出，返回码：{returncode}")
                        # 立即重启，不等待连续错误阈值
                        print(f"   🔄 立即重启 FFmpeg 进程...")
                        time.sleep(1)
                        if self.connect():
                            print(f"   ✅ 重启成功，重置错误计数")
                            consecutive_errors = 0
                            error_count = 0
                        else:
                            print(f"   ❌ 重启失败，3 秒后重试...")
                            time.sleep(3)
                        continue

                    # 每 5 次错误显示一次统计（更频繁）
                    if consecutive_errors % 5 == 0:
                        elapsed = consecutive_errors * 0.05
                        print(f"\n⚠️ 解码异常 {consecutive_errors} 次 (已容忍 {elapsed:.2f}s)", flush=True)

                    # 连续错误过多重启 FFmpeg
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(f"\n🔄 重启 FFmpeg 进程...")
                        time.sleep(1)
                        if self.connect():
                            consecutive_errors = 0
                            error_count = 0
                        else:
                            print("   重启失败，3 秒后重试...")
                            time.sleep(3)
                    else:
                        time.sleep(0.05)

                    continue

                # 成功读取帧
                consecutive_errors = 0
                frame_count += 1
                
                # Python 层控制采样率：每秒只处理1帧
                current_ts = time.time()
                if current_ts - last_sample_time < 1.0:
                    # 距离上次采样不足1秒，跳过此帧（但不跳过更新时间戳）
                    last_frame_time = current_ts  # 更新时间戳
                    continue
                
                # 更新采样时间
                last_sample_time = current_ts

                # 诊断：检查帧质量（每帧都显示）
                h, w = frame.shape[:2]
                total_bytes = frame.nbytes
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\r[{ts}] 📊 帧：{w}x{h}, {total_bytes//1024}KB", flush=True)

                # 保存每帧为图片（带质量检查）
                try:
                    # 检查帧是否有效：计算像素标准差，雪花图的标准差会异常高或低
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    std_dev = np.std(gray)
                    mean_val = np.mean(gray)
                    
                    # 雪花/横条检测：标准差过大(>80)或过小(<5)表示帧异常
                    if std_dev < 5 or std_dev > 80:
                        print(f"\n   ⚠️ 跳过异常帧：std={std_dev:.2f}, mean={mean_val:.2f}")
                    else:
                        current_hour = datetime.now().strftime("%Y-%m-%d_%H")
                        hour_dir = os.path.join('frames', current_hour)
                        if not os.path.exists(hour_dir):
                            os.makedirs(hour_dir)
                        
                        timestamp = datetime.now().strftime("%M_%S_%f")[:-3]
                        filename = f"{timestamp}_frame_{frame_count}.jpg"
                        filepath = os.path.join(hour_dir, filename)
                        
                        # 复制帧避免解码异常
                        frame_copy = frame.copy()
                        success = cv2.imwrite(filepath, frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if success and frame_count % 30 == 0:  # 每 30 帧提示一次
                            print(f"\n   💾 已保存帧：{filename} (std={std_dev:.2f})")
                except Exception as e:
                    print(f"\n   ⚠️ 保存帧异常：{e}")

                # 控制检测频率（FFmpeg 输出原始帧率，Python 层已控制采样）
                loop_time = (time.time() - current_time) * 1000
                if frame_count % 60 == 0:
                    elapsed_total = time.time() - last_frame_time
                    actual_fps = 60 / elapsed_total if elapsed_total > 0 else 0
                    print(f"\n⏱️  单帧耗时：{loop_time:.1f}ms | 实际 FPS：{actual_fps:.2f}", flush=True)
                
                # 更新上一帧时间戳
                last_frame_time = time.time()

                # 定期显示统计（每 2 分钟）
                if current_time - last_stats_time >= 120:
                    success_rate = (frame_count / (frame_count + error_count)) * 100
                    print(f"   📈 已处理 {frame_count} 帧 | 成功率：{success_rate:.1f}%")
                    last_stats_time = current_time

        except KeyboardInterrupt:
            print(f"\n\n⏹️ 停止监控")

        finally:
            total = frame_count + error_count
            success_rate = (frame_count / total * 100) if total > 0 else 0
            print(
                f"📊 统计：总帧数={frame_count}, 错误={error_count}, 成功率={success_rate:.1f}%")

            # 清理资源
            self.stream_capture.close()


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("RTSP 流监控工具 (FFmpeg Pipe 版)")
        print("=" * 60)
        print("\n用法:")
        print("  python rtsp_monitor_ffmpeg.py <RTSP 地址> [阈值] [间隔]")
        print("\n提示:")
        print("  - 需要安装 FFmpeg (确保 ffmpeg 命令可用)")
        print("  - 使用 FFmpeg 进程管道读取，比 OpenCV 更稳定")
        print("=" * 60)
        sys.exit(1)

    rtsp_url = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    monitor = RTSPMonitorFFmpeg(rtsp_url, threshold, interval)
    monitor.monitor()


if __name__ == "__main__":
    main()
