# -*- coding:utf-8 -*-
"""
RTSP 流监控主流程 (多线程版)
使用 rtsp_monitor_ffmpeg.py 的 RTSPMonitorFFmpeg 类
"""
import sys
import os
import time
import threading
import queue
from datetime import datetime
import cv2
import numpy as np

# 导入自定义模块
from config import Config
from logger import get_logger
from motion_detector import MotionDetector
from rtsp_monitor_ffmpeg import RTSPMonitorFFmpeg


class FrameBuffer:
    """线程安全的帧缓冲区（生产者-消费者模式）"""

    def __init__(self, maxsize: int = 1):
        self.frame_queue = queue.Queue(maxsize=maxsize)
        self.dropped_frames = 0
        self.total_frames = 0
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)

    def put_frame(self, frame) -> bool:
        try:
            # 强制保持队列长度为1，立即丢弃旧帧
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
                self.dropped_frames += 1

            self.frame_queue.put_nowait(frame)
            with self.lock:
                self.total_frames += 1
            return True
        except queue.Full:
            self.dropped_frames += 1
            self.logger.debug("帧缓冲区已满，丢弃旧帧")
            return False

    def get_frame(self, timeout: float = 0.5):
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            self.logger.debug("帧缓冲区为空")
            return False, None

    def get_stats(self) -> dict:
        with self.lock:
            return {
                'total_frames': self.total_frames,
                'dropped_frames': self.dropped_frames,
                'queue_size': self.frame_queue.qsize()
            }


class FFmpegErrorMonitor(threading.Thread):
    """FFmpeg 错误监控线程：持续读取 FFmpeg stderr"""

    def __init__(self, stream_capture, logger):
        super().__init__(daemon=True)
        self.stream_capture = stream_capture
        self.logger = logger
        self.running = False

    def run(self):
        self.running = True
        self.logger.info("FFmpeg 错误监控线程已启动")

        while self.running:
            try:
                if not self.stream_capture.process or self.stream_capture.process.poll() is not None:
                    time.sleep(0.1)
                    continue

                if self.stream_capture.process.stderr:
                    line = self.stream_capture.process.stderr.readline()
                    if line:
                        line = line.decode('utf-8', errors='ignore').strip()
                        if line:
                            self.logger.warning(f"FFmpeg: {line}")
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(0.1)

            except Exception as e:
                self.logger.debug(f"FFmpeg 错误监控异常: {e}")
                time.sleep(0.1)

        self.logger.info("FFmpeg 错误监控线程已停止")

    def stop(self):
        self.running = False


class CaptureThread(threading.Thread):
    """抓图线程：使用 RTSPMonitorFFmpeg 抓图"""

    def __init__(self, config, frame_buffer: FrameBuffer):
        super().__init__(daemon=True)
        self.config = config
        self.frame_buffer = frame_buffer
        self.save_frames = config.getboolean("capture", "save_frames")
        self.quality = config.getint("capture", "quality")

        # 支持两种抓帧模式: fps 或 interval
        self.interval = config.getfloat("capture", "interval")
        if self.interval > 0:
            # 使用间隔模式（每 N 秒抓取1帧）
            self.fps_limit = int(1.0 / self.interval) if self.interval <= 1 else 1
            self.frame_interval = self.interval
        else:
            # 使用 FPS 模式（每秒 N 帧）
            self.fps_limit = config.getint("capture", "fps")
            self.frame_interval = 1.0 / self.fps_limit if self.fps_limit > 0 else 0

        # 使用 RTSPMonitorFFmpeg（传递 FPS 和 quality 参数）
        transport = config.get("rtsp", "transport", "tcp").lower()
        self.monitor = RTSPMonitorFFmpeg(
            config.get("rtsp", "url"),
            fps=self.fps_limit,
            quality=self.quality,
            transport=transport
        )

        self.running = False
        self.frame_count = 0
        self.error_count = 0
        self.last_frame_time = 0
        self.logger = get_logger(__name__)

        # FFmpeg 错误监控线程
        self.error_monitor = None

        self.frames_dir = 'frames'
        if self.save_frames and not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
            self.logger.info(f"已创建帧保存目录: {os.path.abspath(self.frames_dir)}")

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("🎥 抓图线程启动")
        self.logger.info(f"RTSP URL: {self.config.get('rtsp', 'url')}")
        
        # 显示抓帧模式
        if self.interval > 0:
            self.logger.info(f"抓帧模式: 间隔模式 - 每 {self.interval} 秒抓取1帧")
        else:
            self.logger.info(f"抓帧模式: FPS模式 - {self.fps_limit} FPS (间隔: {self.frame_interval:.2f}s)")
        
        self.logger.info(f"JPEG 质量: {self.quality} (1-31, 越小越好)")
        self.logger.info(f"保存每帧截图: {'是' if self.save_frames else '否'}")
        if self.save_frames:
            self.logger.info(f"截图目录: {os.path.abspath(self.frames_dir)}")
        else:
            self.logger.info("⚠️  仅进行运动检测，不保存每帧截图")
        self.logger.info("=" * 60)

        # 使用 RTSPMonitorFFmpeg 连接
        self.logger.info("🔌 正在连接 RTSP 流...")
        if not self.monitor.connect():
            self.logger.error("❌ 抓图线程连接失败！")
            return
        self.logger.info("✅ RTSP 流连接成功！")

        # 启动 FFmpeg 错误监控线程
        self.error_monitor = FFmpegErrorMonitor(self.monitor.stream_capture, self.logger)
        self.error_monitor.start()

        self.running = True
        consecutive_errors = 0
        MAX_ERRORS = 3  # 降低阈值，更快触发重连

        try:
            while self.running:
                # 精确控制采样间隔
                current_time = time.time()
                if self.last_frame_time > 0:
                    elapsed = current_time - self.last_frame_time
                    if elapsed < self.frame_interval:
                        time.sleep(self.frame_interval - elapsed)

                # 直接读取一帧
                ret, frame = self.monitor.get_frame(max_retries=1, timeout=1.5)

                if not ret or frame is None:
                    self.error_count += 1
                    consecutive_errors += 1

                    self.logger.warning(f"⚠️  读取失败 (连续{consecutive_errors}次)")

                    # 立即检查 FFmpeg 进程状态
                    if self.monitor.stream_capture.process and self.monitor.stream_capture.process.poll() is not None:
                        return_code = self.monitor.stream_capture.process.returncode

                        # 读取 FFmpeg 详细信息（非阻塞）
                        stderr_output = ""
                        try:
                            if self.monitor.stream_capture.process.stderr:
                                # 只读取部分错误信息，避免阻塞
                                import select
                                if hasattr(select, 'select'):
                                    ready, _, _ = select.select(
                                        [self.monitor.stream_capture.process.stderr], [], [], 0.5
                                    )
                                    if ready:
                                        stderr_output = self.monitor.stream_capture.process.stderr.read(4096).decode('utf-8', errors='ignore')
                                else:
                                    stderr_output = self.monitor.stream_capture.process.stderr.read(4096).decode('utf-8', errors='ignore')
                        except:
                            pass

                        # 分析错误类型
                        if return_code == 0:
                            self.logger.info(f"ℹ️ FFmpeg 进程正常结束（返回码 0）")
                        else:
                            self.logger.error(f"❌ FFmpeg 进程异常退出！返回码: {return_code}")
                            if stderr_output:
                                for line in stderr_output.split('\n')[:5]:  # 只显示前5行
                                    line = line.strip()
                                    if line:
                                        self.logger.error(f"  {line}")

                        # 判断是否是 -10054 网络错误
                        is_network_error = False
                        if stderr_output:
                            if "-10054" in stderr_output or "WSAECONNRESET" in stderr_output or "Connection reset" in stderr_output:
                                is_network_error = True
                                self.logger.warning("🔍 检测到 -10054 网络连接重置，将启用更稳健的重连策略")

                        # 使用适当的重连策略
                        self.logger.info("🔄 立即重启 FFmpeg 进程...")
                        if self.monitor.stream_capture.connect(fast_mode=not is_network_error):
                            consecutive_errors = 0
                            self.logger.info("✅ FFmpeg 进程重启成功！")
                            # 如果是网络错误，给服务器一点时间恢复
                            if is_network_error:
                                time.sleep(0.5)
                        else:
                            self.logger.error("❌ FFmpeg 进程重启失败，0.5秒后重试...")
                            time.sleep(0.5)
                            continue

                    # 进程还在但读取失败，也尝试快速重连
                    if consecutive_errors >= MAX_ERRORS:
                        self.logger.warning(f"🔄 连续{consecutive_errors}次失败，强制重启连接...")
                        # 先关闭旧连接
                        self.monitor.stream_capture.close()
                        time.sleep(0.3)  # 短暂等待
                        
                        if self.monitor.stream_capture.connect(fast_mode=True):
                            consecutive_errors = 0
                            self.logger.info("✅ 连接重启成功！")
                        else:
                            self.logger.error("❌ 连接重启失败，0.5秒后重试...")
                            time.sleep(0.5)

                    continue

                consecutive_errors = 0
                self.frame_count += 1
                current_frame_time = time.time()

                # 严格检查跳秒（超过预期间隔的 1.2 倍）
                if self.last_frame_time > 0:
                    actual_interval = current_frame_time - self.last_frame_time
                    if actual_interval > self.frame_interval * 1.2:
                        self.logger.warning(
                            f"⚠️  跳秒检测! 预期间隔: {self.frame_interval:.2f}s, "
                            f"实际间隔: {actual_interval:.2f}s"
                        )

                self.last_frame_time = current_frame_time

                # 输出每帧取完的日志（INFO 级别）
                h, w = frame.shape[:2]
                frame_size_kb = frame.nbytes / 1024
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.logger.info(
                    f"[{timestamp}] ✔️ 成功取帧: {w}x{h}px, {frame_size_kb:.1f}KB"
                )

                self.frame_buffer.put_frame(frame.copy())

                if self.save_frames:
                    self._save_frame_to_disk(frame)

                if self.frame_count % 50 == 0:
                    stats = self.frame_buffer.get_stats()
                    self.logger.info(
                        f"📊 统计: 已抓 {self.frame_count} 帧, 错误 {self.error_count} 次, "
                        f"缓冲区 {stats['queue_size']} 帧, 丢弃 {stats['dropped_frames']} 帧"
                    )

        except KeyboardInterrupt:
            self.logger.info("抓图线程收到停止信号")
        finally:
            self.running = False
            # 停止 FFmpeg 错误监控线程
            if self.error_monitor:
                self.error_monitor.stop()
                self.error_monitor.join(timeout=1)
            self.monitor.stream_capture.close()
            self.logger.info(f"抓图线程结束: {self.frame_count} 帧, 错误: {self.error_count} 次")

    def _save_frame_to_disk(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(gray)

            if std_dev < 5 or std_dev > 80:
                return

            current_hour = datetime.now().strftime("%Y-%m-%d_%H")
            hour_dir = os.path.join(self.frames_dir, current_hour)
            if not os.path.exists(hour_dir):
                os.makedirs(hour_dir)

            timestamp = datetime.now().strftime("%M_%S_%f")[:-3]
            filename = f"{timestamp}_frame_{self.frame_count}.jpg"
            filepath = os.path.join(hour_dir, filename)

            frame_copy = frame.copy()
            success = cv2.imwrite(filepath, frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                self.logger.debug(f"保存帧: {filename} (std={std_dev:.2f})")
        except Exception as e:
            self.logger.error(f"保存帧异常: {e}")

    def stop(self):
        self.running = False


class DetectionThread(threading.Thread):
    """检测线程：使用 MotionDetector 进行运动检测"""

    def __init__(self, config, frame_buffer: FrameBuffer):
        super().__init__(daemon=True)
        self.config = config
        self.frame_buffer = frame_buffer
        self.threshold = config.getint("detection", "threshold")
        self.min_contour_area = config.getfloat("detection", "min_contour_area")
        self.stable_frames = config.getint("detection", "stable_frames")
        self.save_screenshot = config.getboolean("detection", "save_screenshot")

        # 独立的运动检测器（为了线程安全）
        self.motion_detector = MotionDetector(
            threshold=self.threshold,
            min_contour_area=self.min_contour_area,
            stable_frames=self.stable_frames,
            save_screenshot=self.save_screenshot
        )

        self.running = False
        self.detection_count = 0
        self.motion_count = 0
        self.last_detection_time = 0
        self.logger = get_logger(__name__)

    def run(self):
        self.logger.info("检测线程启动")
        self.logger.debug(f"阈值: {self.threshold}")
        self.logger.debug(f"最小面积比例: {self.min_contour_area*100:.3f}%")
        self.logger.info(f"稳定帧数: {self.stable_frames}")
        self.logger.info(f"保存截图: {'是' if self.save_screenshot else '否'}")

        self.running = True
        motion_confirm = 0

        self.logger.info("🔥 正在预热背景模型...")
        warmup_count = 0
        while warmup_count < 10 and self.running:
            ret, frame = self.frame_buffer.get_frame(timeout=2.0)
            if ret and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                self.motion_detector.bg_subtractor.apply(gray, learningRate=1.0)
                self.motion_detector.update_last_frame(frame)
                warmup_count += 1
                time.sleep(0.1)
        self.logger.info("✅ 预热完成！")

        try:
            while self.running:
                ret, frame = self.frame_buffer.get_frame(timeout=1.0)

                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                self.detection_count += 1
                current_time = time.time()

                is_motion, ratio, diff_pixels = self.motion_detector.detect(frame)

                if is_motion:
                    motion_confirm += 1
                    if motion_confirm >= self.stable_frames:
                        self.motion_count += 1
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self.logger.warning(
                            f"[{timestamp}] ⚠️  检测到运动！({ratio * 100:.1f}%) [#{self.motion_count}]"
                        )
                        self.logger.info(f"差异像素: {diff_pixels}, 阈值: {self.threshold}")

                        if self.save_screenshot:
                            self.motion_detector.save_screenshot_file(frame, diff_pixels, self.motion_count)

                        motion_confirm = 0
                else:
                    motion_confirm = 0

                self.motion_detector.update_last_frame(frame)

                if self.detection_count % 30 == 0:
                    elapsed = current_time - self.last_detection_time
                    if elapsed > 0:
                        fps = 30 / elapsed
                        self.logger.info(
                            f"📊 检测统计: 已检 {self.detection_count} 帧, "
                            f"运动 {self.motion_count} 次, 检测速度: {fps:.2f} FPS"
                        )
                    self.last_detection_time = current_time

        except KeyboardInterrupt:
            self.logger.info("检测线程收到停止信号")
        finally:
            self.running = False
            self.logger.info(
                f"检测线程结束: {self.detection_count} 帧, "
                f"运动检测: {self.motion_count} 次"
            )

    def stop(self):
        self.running = False


class RTSPMonitor:
    """RTSP 监控主类（整合 RTSPMonitorFFmpeg）"""

    def __init__(self, config):
        self.config = config
        self.rtsp_url = config.get("rtsp", "url")
        self.logger = get_logger(__name__)

        self.frame_buffer = FrameBuffer(maxsize=1)
        self.capture_thread = CaptureThread(config, self.frame_buffer)
        self.detection_thread = DetectionThread(config, self.frame_buffer)

    def start(self):
        self.logger.info("=" * 60)
        self.logger.info("🎬 RTSP 流监控工具 (整合 RTSPMonitorFFmpeg 版)")
        self.logger.info("=" * 60)
        self.logger.info(f"📡 RTSP URL: {self.rtsp_url}")
        self.logger.info(f"🎯 检测阈值: {self.config.get('detection', 'threshold')}")
        fps = self.config.getint('capture', 'fps')
        quality = self.config.getint('capture', 'quality')
        self.logger.info(f"📸 抓图帧率: {fps} FPS (每 {1.0/fps if fps > 0 else 0:.2f} 秒)")
        self.logger.info(f"🎨 JPEG 质量: {quality} (1-31, 越小质量越好)")
        save_frames = self.config.getboolean('capture', 'save_frames')
        self.logger.info(
            f"💾 保存每帧截图: {'是' if save_frames else '否'} "
            f"{'(frames/)' if save_frames else '(仅运动检测)'}"
        )
        self.logger.info(
            f"🖼️  运动检测截图: {'是' if self.config.getboolean('detection', 'save_screenshot') else '否'} "
            f"{'(screenshot/)' if self.config.getboolean('detection', 'save_screenshot') else ''}"
        )
        self.logger.info("=" * 60)

        self.capture_thread.start()
        time.sleep(1)
        self.detection_thread.start()

        self.logger.info("✅ 所有线程已启动，按 Ctrl+C 停止监控...")

        try:
            while True:
                time.sleep(1)

                if not self.capture_thread.is_alive():
                    self.logger.warning("抓图线程已退出")
                    break

                if not self.detection_thread.is_alive():
                    self.logger.warning("检测线程已退出")
                    break

        except KeyboardInterrupt:
            self.logger.info("收到停止信号，正在关闭...")
        finally:
            self.stop()

    def stop(self):
        self.logger.info("正在停止所有线程...")

        self.detection_thread.stop()
        self.capture_thread.stop()

        self.capture_thread.join(timeout=5)
        self.detection_thread.join(timeout=5)

        self.logger.info("所有线程已停止")

        capture_stats = self.frame_buffer.get_stats()
        self.logger.info("=" * 60)
        self.logger.info("最终统计:")
        self.logger.info(
            f"抓图线程: {self.capture_thread.frame_count} 帧, "
            f"错误 {self.capture_thread.error_count} 次"
        )
        self.logger.info(
            f"检测线程: {self.detection_thread.detection_count} 帧, "
            f"运动 {self.detection_thread.motion_count} 次"
        )
        self.logger.info(
            f"缓冲区: 总计 {capture_stats['total_frames']} 帧, "
            f"丢弃 {capture_stats['dropped_frames']} 帧"
        )
        self.logger.info("=" * 60)


def main():
    config = Config()

    if len(sys.argv) > 1:
        config.set("rtsp", "url", sys.argv[1])
        if len(sys.argv) > 2:
            config.set("detection", "threshold", sys.argv[2])
        if len(sys.argv) > 3:
            config.set("capture", "fps", sys.argv[3])

    if not config.get("rtsp", "url"):
        get_logger().error("缺少 RTSP 地址参数")
        print("=" * 60)
        print("RTSP 流监控工具 (整合 RTSPMonitorFFmpeg 版)")
        print("=" * 60)
        print("\n用法:")
        print("  python rtsp_monitor_main.py <RTSP地址> [阈值] [抓图FPS]")
        print("\n参数说明:")
        print("  RTSP地址      : RTSP 视频流地址")
        print("  阈值          : 像素差异阈值，默认 1000")
        print("  抓图FPS       : 每秒抓取帧数，默认 1")
        print("\n配置说明 (config.ini):")
        print("  [capture] save_frames = True/False  # 是否保存每帧截图")
        print("  [detection] save_screenshot = True/False  # 是否保存运动检测截图")
        print("\n示例:")
        print("  python rtsp_monitor_main.py rtsp://admin:pass@192.168.1.100/sub")
        print("  python rtsp_monitor_main.py rtsp://admin:pass@192.168.1.100/sub 800 2")
        print("\n特性:")
        print("  ✅ 已引入 rtsp_monitor_ffmpeg 模块")
        print("  ✅ 使用 RTSPMonitorFFmpeg 类进行抓图")
        print("  ✅ 抓图和检测独立线程运行，互不阻塞")
        print("  ✅ 线程安全的生产者-消费者模式")
        print("  ✅ 自动重连机制")
        print("  ✅ 详细的统计信息")
        print("=" * 60)
        sys.exit(1)

    monitor = RTSPMonitor(config)
    monitor.start()


if __name__ == "__main__":
    main()
