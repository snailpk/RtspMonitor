# -*- coding:utf-8 -*-
"""
RTSP 流捕获模块 (FFmpeg Pipe 版)
使用 FFmpeg 进程管道读取视频流，比 OpenCV 直接读取更稳定
"""
import os
import time
import cv2
import numpy as np
import subprocess
import logging
import warnings
from logger import get_logger

# 屏蔽日志
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
warnings.filterwarnings('ignore')


class StreamCapture:
    """使用 FFmpeg pipe 的 RTSP 流捕获类"""

    def __init__(self, rtsp_url: str, fps: int = 15, quality: int = 5, transport: str = "tcp"):
        """
        初始化流捕获器

        Args:
            rtsp_url: RTSP 流地址
            fps: 输出帧率，默认 15
            quality: JPEG 质量等级 (1-31)，越小越好，默认 5
            transport: RTSP 传输协议 ("tcp" 或 "udp")
        """
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.quality = quality
        self.transport = transport.lower()
        self.process = None
        self.logger = get_logger(__name__)

    def connect(self, fast_mode: bool = False) -> bool:
        """
        使用 FFmpeg 连接 RTSP 流

        Args:
            fast_mode: 快速模式，跳过首帧验证（用于快速重连）

        Returns:
            bool: 连接成功返回 True
        """
        try:
            self.logger.info("正在连接 RTSP 流 (FFmpeg Pipe 模式)...")
            self.logger.debug(f"URL: {self.rtsp_url}")

            # 关闭旧进程（快速模式）
            if self.process is not None:
                try:
                    # 先尝试快速终止
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1)  # 缩短等待时间到1秒
                    except subprocess.TimeoutExpired:
                        # 如果1秒内未退出，强制杀死
                        self.process.kill()
                        self.process.wait(timeout=1)
                except Exception as e:
                    self.logger.debug(f"关闭旧进程: {e}")
                finally:
                    self.process = None

            # 根据传输协议选择 FFmpeg 命令参数（完全兼容旧版本）
            cmd = [
                'ffmpeg',
                '-rtsp_transport', self.transport,
            ]

            # 添加传输协议特定参数
            if self.transport == "tcp":
                cmd.extend(['-rtsp_flags', 'prefer_tcp'])
            else:  # udp
                cmd.extend(['-max_delay', '5000000'])

            cmd.extend([
                '-buffer_size', '524288',   # 512KB 缓冲，平衡延迟和稳定性
                '-fflags', '+discardcorrupt+genpts+igndts+nobuffer',
                '-flags', 'low_delay',       # 低延迟
                '-probesize', '32768',       # 32KB 探测
                '-analyzeduration', '1000000', # 分析 1 秒
                '-i', self.rtsp_url,
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',
                '-q:v', str(self.quality),
                '-vf', f'fps={self.fps}',
                '-flush_packets', '1',
                '-loglevel', 'error',        # 使用 error 级别减少输出
                '-'
            ])

            self.logger.debug(f"启动 FFmpeg 进程，命令: {' '.join(cmd)}")

            # 启动 FFmpeg 进程（使用非阻塞模式）
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # 捕获 stderr 以查看错误
                stdin=subprocess.DEVNULL,
                bufsize=0  # 无缓冲模式
            )

            # 等待 FFmpeg 启动（快速模式缩短等待）
            startup_wait = 0.8 if fast_mode else 1.2
            time.sleep(startup_wait)

            # 检查进程是否存活
            if self.process.poll() is not None:
                # 进程已退出，读取详细错误信息
                stdout_data, stderr_data = self.process.communicate(timeout=5)
                self.logger.error("FFmpeg 进程启动失败")
                self.logger.error(f"返回码: {self.process.returncode}")
                if stderr_data:
                    error_msg = stderr_data.decode('utf-8', errors='ignore')
                    self.logger.error("错误信息:")
                    for line in error_msg.split('\n'):
                        line = line.strip()
                        if line:
                            self.logger.error(f"  {line}")
                return False

            # 快速模式下跳过首帧验证
            if fast_mode:
                self.logger.info(f"快速连接成功！")
                return True

            self.logger.debug("尝试读取首帧...")

            # 尝试读取第一帧
            ret, frame = self._read_frame()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                self.logger.info(f"连接成功！分辨率: {w}x{h}")
                return True
            else:
                self.logger.error("连接失败: 无法读取首帧")
                # 检查进程状态
                if self.process.poll() is not None:
                    stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore') if self.process.stderr else "无错误信息"
                    self.logger.error(f"FFmpeg 进程已退出，返回码: {self.process.returncode}")
                    self.logger.error(f"错误信息:")
                    for line in stderr_output.split('\n'):
                        line = line.strip()
                        if line:
                            self.logger.error(f"  {line}")
                return False

        except FileNotFoundError:
            self.logger.critical("找不到 FFmpeg 程序！")
            self.logger.critical("请确保已安装 FFmpeg 并添加到 PATH")
            return False
        except Exception as e:
            self.logger.error(f"连接异常: {type(e).__name__}: {e}", exc_info=True)
            return False

    def _read_frame(self, timeout: float = 5.0):
        """
        从 FFmpeg 进程管道读取一帧（MJPEG 格式 + JPEG 帧边界检测）

        Args:
            timeout: 超时时间（秒）

        Returns:
            tuple: (success, frame)
        """
        try:
            if self.process is None or self.process.poll() is not None:
                return False, None

            # 读取 JPEG 帧: 寻找 SOI (0xFFD8) 和 EOI (0xFFD9) 标记
            return self._read_jpeg_frame(timeout)

        except Exception as e:
            self.logger.error(f"读取帧异常: {type(e).__name__}: {e}", exc_info=True)
            return False, None

    def _read_jpeg_frame(self, timeout: float = 1.5):
        """
        读取一帧 JPEG 数据（通过 SOI/EOI 标记 + 低延迟读取）

        Args:
            timeout: 超时时间（秒）

        Returns:
            tuple: (success, frame)
        """
        try:
            start_time = time.time()

            # 步骤 1: 寻找 SOI 标记 (0xFFD8)
            soi_found = False
            buffer = bytearray()
            search_buffer = bytearray()
            max_search_iterations = 100  # 低延迟：快速超时
            search_count = 0

            while not soi_found:
                if time.time() - start_time > timeout:
                    self.logger.debug("等待 SOI 标记超时")
                    return False, None

                search_count += 1
                if search_count > max_search_iterations:
                    self.logger.debug("SOI 搜索次数过多，放弃当前帧")
                    return False, None

                # 低延迟：单次读取较大块
                chunk = self.process.stdout.read(8192)
                if not chunk or len(chunk) == 0:
                    if self.process.poll() is not None:
                        self.logger.debug("FFmpeg 进程已退出")
                        return False, None
                    time.sleep(0.001)  # 极短等待
                    continue

                search_buffer.extend(chunk)

                # 在缓冲区中搜索 SOI
                soi_pos = search_buffer.find(b'\xff\xd8')

                if soi_pos >= 0:
                    buffer = bytearray(search_buffer[soi_pos:])
                    search_buffer.clear()
                    soi_found = True
                else:
                    # 保留最后几个字节
                    if len(search_buffer) > 10:
                        search_buffer = bytearray(search_buffer[-10:])

            # 步骤 2: 读取直到 EOI 标记 (0xFFD9)
            eoi_found = False
            max_jpeg_size = 500 * 1024  # 500KB 上限，低延迟
            max_read_iterations = 50  # 更快放弃
            read_count = 0

            while not eoi_found:
                if time.time() - start_time > timeout:
                    self.logger.debug("读取 JPEG 数据超时")
                    return False, None

                read_count += 1
                if read_count > max_read_iterations:
                    self.logger.debug("EOI 读取次数过多，放弃当前帧")
                    return False, None

                if len(buffer) > max_jpeg_size:
                    self.logger.debug(f"JPEG 数据过大 ({len(buffer)//1024}KB)，放弃")
                    return False, None

                chunk = self.process.stdout.read(32768)
                if not chunk or len(chunk) == 0:
                    if self.process.poll() is not None:
                        self.logger.debug("FFmpeg 进程已退出")
                        return False, None
                    time.sleep(0.005)
                    continue

                buffer.extend(chunk)

                eoi_pos = buffer.find(b'\xff\xd9')
                if eoi_pos >= 0:
                    eoi_found = True
                    jpeg_data = bytes(buffer[:eoi_pos+2])
                    # 剩余数据丢弃（避免累积旧数据）
                    buffer.clear()

            # 步骤 3: 解码 JPEG 为 OpenCV 帧
            if len(jpeg_data) < 100:
                self.logger.debug(f"JPEG 数据过小 ({len(jpeg_data)} 字节)")
                return False, None

            np_arr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None or frame.size == 0:
                self.logger.debug("JPEG 解码失败")
                return False, None

            return True, frame

        except Exception as e:
            self.logger.error(f"JPEG 帧读取异常: {type(e).__name__}: {e}", exc_info=True)
            return False, None

    def _flush_pipe(self):
        """刷新管道中的旧数据"""
        try:
            import fcntl
            import os
            # 设置非阻塞模式
            fd = self.process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            # 读取所有可用数据
            flushed = 0
            while True:
                try:
                    data = self.process.stdout.read(4096)
                    if not data:
                        break
                    flushed += len(data)
                except:
                    break

            # 恢复阻塞模式
            fcntl.fcntl(fd, fcntl.F_SETFL, fl)

            if flushed > 0:
                self.logger.debug(f"已刷新管道: {flushed} 字节")
        except (ImportError, AttributeError):
            # Windows 或不支持 fcntl，跳过
            pass

    def get_frame(self, max_retries: int = 1, timeout: float = 5.0):
        """
        获取一帧（带重试机制）

        Args:
            max_retries: 最大重试次数
            timeout: 超时时间（秒）

        Returns:
            tuple: (success, frame)
        """
        for attempt in range(max_retries):
            ret, frame = self._read_frame(timeout)
            if ret and frame is not None:
                return True, frame
            time.sleep(0.05)

        return False, None

    def is_connected(self) -> bool:
        """
        检查连接状态

        Returns:
            bool: 连接正常返回 True
        """
        return self.process is not None and self.process.poll() is None

    def close(self):
        """关闭流连接"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception as e:
                self.logger.debug(f"关闭进程时出错: {e}")
            self.process = None
