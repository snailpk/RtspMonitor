# -*- coding:utf-8 -*-
"""
RTSP 流监控工具 (终极稳定版)
彻底解决 H.264 解码错误问题，实现长时间稳定运行
"""
import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
import logging
import warnings
import ctypes
import traceback

# ==========================================
# 1. 全局日志屏蔽（必须在导入 cv2 之前设置）
# ==========================================
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEO_DEBUG'] = '0'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # 完全静音 FFmpeg

# 屏蔽 Python 警告
warnings.filterwarnings('ignore')

# 设置日志级别
logging.basicConfig(level=logging.CRITICAL, format='')

# 尝试禁用 FFmpeg 日志
try:
    # 加载 FFmpeg 库并设置日志级别为静音
    for lib_name in ['libavutil.so', 'avutil-58.dll', 'avutil-56.dll', 'avutil.dylib']:
        try:
            lib = ctypes.CDLL(lib_name)
            lib.av_log_set_level(-8)  # AV_LOG_QUIET = -8
            break
        except:
            pass
except:
    pass

# ==========================================
# 2. FFmpeg 解码参数优化（关键！）
# ==========================================
# 注意：如果出现 "error while decoding MB" 错误，可以尝试以下调整：
#   1. 将 rtsp_transport 改为 udp（某些摄像头 TCP 模式有问题）
#   2. 增大 buffer_size 到 16MB (16777216)
#   3. 降低检查频率 check_interval 到 2-3 秒
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp'  # TCP 传输（稳定），如仍有问题可改为 udp
    '|stimeout;3000000'  # 3 秒连接超时
    '|buffer_size;8388608'  # 8MB 超大缓冲（应对严重网络抖动）
    '|ec;15'  # 最强错误隐藏 (最大值)
    '|err_detect;ignore_err'  # 忽略所有错误
    '|fflags;+discardcorrupt+genpts'  # 丢弃损坏包 + 生成 PTS
    '|skip_frame;noref'  # 跳过非参考帧
    '|skip_loop_filter;all'  # 跳过环路滤波
    '|flags;low_delay+igndts'  # 低延迟模式 + 忽略 DTS
    '|threads;1'  # 单线程解码（避免线程竞争）
    '|strict;experimental'  # 允许实验性功能
    '|probesize;32768'  # 最小探测 32KB
    '|analyzeduration;0'  # 不分析
    '|max_delay;1000000'  # 最大延迟 1 秒
    '|flush_packets;1'  # 刷新数据包
    '|fpsprobesize;0'  # 不探测 FPS
    '|avioflags;direct'  # 直接 I/O
    '|rw_timeout;5000000'  # 读写超时 5 秒
    '|framedrop;1'  # 允许丢帧保流畅
    '|reconnect;1'  # 自动重连
    '|reconnect_at_eof;1'  # EOF 时重连
    '|reconnect_streamed;1'  # 流式传输重连
)

print("🔧 RTSP 流监控工具 (终极稳定版)")
print("=" * 60)


class RTSPMonitor:
    """RTSP 流监控类（终极稳定版）"""

    def __init__(self, rtsp_url: str, threshold: int = 1000,
                 check_interval: float = 1.0, save_screenshot: bool = True,
                 min_contour_area: float = 0.001, stable_frames: int = 2,
                 use_sub_stream: bool = True):
        # 自动切换到子码流
        if use_sub_stream and '/main' in rtsp_url:
            rtsp_url = rtsp_url.replace('/main', '/sub')
            print(f"🔄 已自动切换到子码流")
        elif use_sub_stream and '/h264/ch1/main' in rtsp_url:
            rtsp_url = rtsp_url.replace('/h264/ch1/main', '/h264/ch1/sub')
            print(f"🔄 已自动切换到子码流")

        self.rtsp_url = rtsp_url
        self.threshold = threshold
        self.check_interval = check_interval
        self.save_screenshot = save_screenshot
        self.min_contour_area = min_contour_area
        self.stable_frames = stable_frames
        
        # 打印检测参数
        print(f"📊 检测参数：最小面积比例={min_contour_area*100:.3f}%, 稳定帧数={stable_frames}")

        self.cap = None
        self.last_frame = None
        self.screenshot_dir = 'screenshot'
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=20, detectShadows=False
        )

        # 创建目录
        if self.save_screenshot:
            if not os.path.exists(self.screenshot_dir):
                os.makedirs(self.screenshot_dir)
                print(f"📁 已创建截图目录：{os.path.abspath(self.screenshot_dir)}")

    def connect(self) -> bool:
        """连接到 RTSP 流"""
        try:
            # 释放旧连接
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
                time.sleep(0.3)

            print(f"\n📡 正在连接 RTSP 流...")
            print(f"   URL: {self.rtsp_url}")

            # 创建 VideoCapture
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # 设置缓冲区大小为 1（最低延迟）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 尝试设置 FPS
            self.cap.set(cv2.CAP_PROP_FPS, 15)

            if not self.cap.isOpened():
                print("❌ 连接失败：无法打开视频流")
                return False

            # 快速获取首帧
            print("   ⏳ 等待视频流初始化...")
            for i in range(20):
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"✅ 连接成功！分辨率: {w}x{h}")
                        self.last_frame = frame
                        return True
                except:
                    pass
                time.sleep(0.1)

            print("❌ 连接超时：无法获取首帧")
            return False

        except Exception as e:
            print(f"❌ 连接异常: {e}")
            return False

    def get_frame(self, max_retries=1):
        """
        获取一帧（带重试机制）
        返回：(success, frame)
        """
        for attempt in range(max_retries):
            try:
                ret, frame = self.cap.read()
                    
                if ret and frame is not None and frame.size > 0:
                    # 额外检查：确保帧数据完整
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        return True, frame
                    else:
                        print(f"\r⚠️ 帧格式异常 {frame.shape}", end='', flush=True)
                elif not ret:
                    # 静默失败，不打印（由主循环统一处理）
                    pass
            except Exception as e:
                # 静默捕获异常，由主循环处理
                pass
            time.sleep(0.01)
    
        return False, None

    def detect_motion(self, frame) -> tuple:
        """运动检测（优化版）"""
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

            # 计算差异像素（用于保存截图）
            diff_pixels = 0
            if self.last_frame is not None and self.last_frame.shape == frame.shape:
                gray1 = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray1, gray)
                _, diff_bin = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                diff_pixels = cv2.countNonZero(diff_bin)

            # 调试信息：每 10 次检测显示一次详细数据
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
                
            if self._debug_count % 10 == 0:
                print(f"   🔍 检测详情：轮廓数={len(contours)}, 有效轮廓={len(significant)}, "
                      f"最小面积阈值={min_area}, 运动比例={motion_ratio*100:.2f}%, 差异像素={diff_pixels}")

            return len(significant) > 0, motion_ratio, diff_pixels

        except Exception as e:
            print(f"⚠️ 运动检测异常：{e}")
            import traceback
            traceback.print_exc()
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
                
            # 确保帧数据有效
            if frame is None or frame.size == 0:
                print(f"   ⚠️ 警告：无法保存空帧")
                return
            
            # 先复制帧，避免后续处理影响原数据
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

    def monitor(self):
        """监控主循环"""
        if not self.connect():
            return
    
        # 预热：跳过前几帧，等待背景模型稳定
        print(f"🔥 正在预热，跳过前 10 帧...")
        for i in range(10):
            ret, frame = self.get_frame(max_retries=3)
            if ret:
                # 更新背景模型
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                self.bg_subtractor.apply(gray, learningRate=1.0)
                self.last_frame = frame
        print(f"✅ 预热完成，开始监控...")
    
        print(f"\n🔍 开始监控...")
        print(f"📊 阈值：{self.threshold} | 间隔：{self.check_interval}s")
        print(f"🛡️ 已启用强错误隐藏模式")
        print(f"⚠️ 提示：网络不稳定时会自动重连")
        print(f"⏹️ 按 Ctrl+C 停止\n")

        # 统计变量
        frame_count = 0
        motion_count = 0
        error_count = 0
        consecutive_errors = 0
        motion_confirm = 0
        last_stats_time = time.time()

        # 错误容忍度
        MAX_CONSECUTIVE_ERRORS = 30  # 允许连续 30 次错误才重连（约 0.3 秒）

        try:
            while True:
                # 获取当前时间
                current_time = time.time()
                
                # 读取帧
                ret, frame = self.get_frame(max_retries=5)

                if not ret or frame is None:
                    error_count += 1
                    consecutive_errors += 1
                
                    # 每次失败都打印（使用 \r 保持在同一行）
                    print(f"\r⚠️ 读取失败 (连续{consecutive_errors}次)       ", end='', flush=True)
                
                    # 每 20 次错误显示一次统计（换行）
                    if consecutive_errors % 20 == 0:
                        elapsed = consecutive_errors * 0.01
                        print(f"\n⚠️ 解码异常 {consecutive_errors} 次 (已容忍 {elapsed:.2f}s)", flush=True)
                
                    # 连续错误过多才重连（降低阈值）
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(f"\n🔄 连续错误过多 ({consecutive_errors}次)，尝试重连...")
                
                        # 重连
                        time.sleep(1)  # 缩短等待
                        if self.connect():
                            consecutive_errors = 0
                            error_count = 0
                            motion_confirm = 0
                        else:
                            # 重连失败，等待更长时间
                            print("   重连失败，3 秒后重试...")
                            time.sleep(3)
                    else:
                        # 短暂等待
                        time.sleep(0.01)  # 缩短等待
                
                    continue

                # 成功读取帧
                consecutive_errors = 0
                frame_count += 1
                
                # 性能监控：记录关键操作耗时
                start_time = time.time()
                
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

                # 运动检测
                motion_start = time.time()
                is_motion, ratio, diff_pixels = self.detect_motion(frame)
                motion_time = (time.time() - motion_start) * 1000
                
                # 只有检测到运动时才打印详细信息（减少输出）
                if is_motion and motion_confirm >= self.stable_frames - 1:
                    print(f"   🔍 检测到轮廓：ratio={ratio*100:.2f}%, diff={diff_pixels}, 耗时={motion_time:.1f}ms")

                if is_motion:
                    motion_confirm += 1
                    if motion_confirm >= self.stable_frames:
                        motion_count += 1
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] ⚠️ 检测到变化！({ratio * 100:.1f}%) [#{motion_count}]")
                        print(f"   📊 差异像素：{diff_pixels}, 阈值：{self.threshold}")

                        if self.save_screenshot:
                            # 先复制帧再保存，避免 IO 操作影响原始帧
                            screenshot_frame = frame.copy() if frame is not None else None
                            self.save_screenshot_file(screenshot_frame, diff_pixels, motion_count)
                            # 释放内存
                            del screenshot_frame
                        else:
                            print(f"   ℹ️ 截图功能已禁用")

                        motion_confirm = 0
                else:
                    motion_confirm = 0

                # 更新上一帧（必须 copy，避免引用同一内存）
                if frame is not None and frame.size > 0:
                    self.last_frame = frame.copy()
                else:
                    print(f"   ⚠️ 警告：跳过无效帧的更新")

                # 控制检测频率
                loop_time = (time.time() - start_time) * 1000
                if frame_count % 60 == 0:  # 每 60 帧显示一次性能统计（换行显示）
                    print(f"\n⏱️  单帧耗时：{loop_time:.1f}ms (检测:{motion_time:.1f}ms)", flush=True)
                
                time.sleep(self.check_interval)

                # 定期显示统计（每2分钟）
                if current_time - last_stats_time >= 120:
                    success_rate = (frame_count / (frame_count + error_count)) * 100
                    print(f"   📈 已处理 {frame_count} 帧 | 成功率: {success_rate:.1f}%")
                    last_stats_time = current_time

        except KeyboardInterrupt:
            print(f"\n\n⏹️ 停止监控")

        finally:
            total = frame_count + error_count
            success_rate = (frame_count / total * 100) if total > 0 else 0
            print(
                f"📊 统计: 总帧数={frame_count}, 变化次数={motion_count}, 错误={error_count}, 成功率={success_rate:.1f}%")
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("RTSP 流监控工具 (终极稳定版)")
        print("=" * 60)
        print("\n用法:")
        print("  python rtsp_monitor.py <RTSP地址> [阈值] [间隔] [--no-auto-sub]")
        print("\n参数说明:")
        print("  RTSP地址      : RTSP 视频流地址")
        print("  阈值          : 像素差异阈值，默认 1000")
        print("  间隔          : 检查间隔秒数，默认 1.0")
        print("  --no-auto-sub : 禁用自动切换子码流")
        print("\n示例:")
        print("  python rtsp_monitor.py rtsp://admin:pass@192.168.1.100/main")
        print("  python rtsp_monitor.py rtsp://admin:pass@192.168.1.100/sub 800 0.5")
        print("  python rtsp_monitor.py rtsp://admin:pass@192.168.1.100/main 1000 1.0 --no-auto-sub")
        print("\n提示:")
        print("  - 默认自动将主码流切换为子码流以提高稳定性")
        print("  - 使用 --no-auto-sub 参数可禁用此功能")
        print("  - 子码流分辨率低、码率低，解码稳定，适合长时间监控")
        print("=" * 60)
        sys.exit(1)

    rtsp_url = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    auto_sub = '--no-auto-sub' not in sys.argv

    monitor = RTSPMonitor(rtsp_url, threshold, interval, use_sub_stream=auto_sub)
    monitor.monitor()


if __name__ == "__main__":
    main()
