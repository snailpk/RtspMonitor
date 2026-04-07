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

    def __init__(self, rtsp_url: str, fps: int = 15, quality: int = 5):
        """
        初始化流捕获器

        Args:
            rtsp_url: RTSP 流地址
            fps: 输出帧率，默认 15
            quality: JPEG 质量等级 (1-31)，越小越好，默认 5
        """
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.quality = quality
        self.process = None
        self.logger = get_logger(__name__)

    def connect(self) -> bool:
        """
        使用 FFmpeg 连接 RTSP 流

        Returns:
            bool: 连接成功返回 True
        """
        try:
            self.logger.info("正在连接 RTSP 流 (FFmpeg Pipe 模式)...")
            self.logger.debug(f"URL: {self.rtsp_url}")

            # 关闭旧进程
            if self.process is not None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                except Exception as e:
                    self.logger.warning(f"关闭旧进程失败: {e}")

            # FFmpeg 命令参数（使用 fps 过滤器控制输出帧率）
            cmd = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-buffer_size', '65536',  # 64KB 小缓冲
                '-fflags', '+discardcorrupt+genpts+igndts+nobuffer',  # 无缓冲模式
                '-flags', 'low_delay',  # 低延迟
                '-probesize', '16384',  # 最小探测 16KB
                '-analyzeduration', '0',  # 不分析，立即开始
                '-i', self.rtsp_url,
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',         # MJPEG 格式
                '-q:v', str(self.quality),  # JPEG 质量
                '-vf', f'fps={self.fps}',   # FFmpeg 层精确控制帧率
                '-flush_packets', '1',      # 立即刷新
                '-loglevel', 'error',
                '-'
            ]

            self.logger.debug("启动 FFmpeg 进程...")

            # 启动 FFmpeg 进程（使用非阻塞模式）
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # 捕获 stderr 以查看错误
                stdin=subprocess.DEVNULL,
                bufsize=0  # 无缓冲模式
            )

            # 等待 FFmpeg 启动
            time.sleep(3)

            # 检查进程是否存活
            if self.process.poll() is not None:
                # 进程已退出，读取错误信息
                stdout_data, stderr_data = self.process.communicate()
                self.logger.error("FFmpeg 进程启动失败")
                self.logger.error(f"返回码: {self.process.returncode}")
                if stderr_data:
                    error_msg = stderr_data.decode('utf-8', errors='ignore')
                    self.logger.error("错误信息:")
                    for line in error_msg.split('\n')[:10]:  # 只显示前 10 行
                        if line.strip():
                            self.logger.error(f"  {line}")
                if stdout_data:
                    output_msg = stdout_data.decode('utf-8', errors='ignore')
                    if output_msg.strip():
                        self.logger.debug(f"输出信息: {output_msg[:200]}")
                return False

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
                    self.logger.error(f"错误信息: {stderr_output[:300]}")
                return False

        except FileNotFoundError:
            self.logger.critical("找不到 FFmpeg 程序！")
            self.logger.critical("请确保已安装 FFmpeg 并添加到 PATH")
            self.logger.critical("下载地址: https://www.gyan.dev/ffmpeg/builds/")
            return False
        except Exception as e:
            self.logger.error(f"连接异常: {type(e).__name__}: {e}", exc_info=True)
            return False

    def _read_frame(self):
        """
        从 FFmpeg 进程管道读取一帧（MJPEG 格式 + JPEG 帧边界检测）

        Returns:
            tuple: (success, frame)
        """
        try:
            if self.process is None or self.process.poll() is not None:
                return False, None

            # 读取 JPEG 帧: 寻找 SOI (0xFFD8) 和 EOI (0xFFD9) 标记
            return self._read_jpeg_frame()

        except Exception as e:
            self.logger.error(f"读取帧异常: {type(e).__name__}: {e}", exc_info=True)
            return False, None

    def _read_jpeg_frame(self):
        """
        读取一帧 JPEG 数据（通过 SOI/EOI 标记 + 批量读取优化 + 容错处理）

        Returns:
            tuple: (success, frame)
        """
        try:
            start_time = time.time()
            timeout = 5.0

            # 步骤 1: 寻找 SOI 标记 (0xFFD8) - 使用批量读取加速
            soi_found = False
            buffer = bytearray()
            search_buffer = bytearray()  # 用于搜索的滑动窗口
            max_search_iterations = 1000  # 最大搜索次数，防止死循环
            search_count = 0

            while not soi_found:
                if time.time() - start_time > timeout:
                    self.logger.debug("等待 SOI 标记超时")
                    return False, None

                search_count += 1
                if search_count > max_search_iterations:
                    self.logger.debug("SOI 搜索次数过多，重置缓冲区")
                    search_buffer.clear()
                    search_count = 0

                # 批量读取加速
                chunk = self.process.stdout.read(8192)
                if not chunk or len(chunk) == 0:
                    if self.process.poll() is not None:
                        self.logger.debug("FFmpeg 进程已退出")
                        return False, None
                    time.sleep(0.01)
                    continue

                search_buffer.extend(chunk)

                # 在缓冲区中搜索 SOI
                soi_pos = -1
                for i in range(len(search_buffer) - 1):
                    if search_buffer[i] == 0xFF and search_buffer[i+1] == 0xD8:
                        soi_pos = i
                        break

                if soi_pos >= 0:
                    # 找到 SOI，保留从 SOI 开始的数据
                    buffer = bytearray(search_buffer[soi_pos:])
                    search_buffer.clear()
                    soi_found = True
                else:
                    # 没找到，保留最后几个字节（可能是部分 SOI）
                    if len(search_buffer) > 10:
                        search_buffer = bytearray(search_buffer[-10:])

            # 步骤 2: 读取直到 EOI 标记 (0xFFD9)
            eoi_found = False
            max_jpeg_size = 2 * 1024 * 1024  # 最大 2MB
            max_read_iterations = 500  # 最大读取次数
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

                chunk = self.process.stdout.read(65536)  # 批量读取 64KB
                if not chunk or len(chunk) == 0:
                    if self.process.poll() is not None:
                        self.logger.debug("FFmpeg 进程已退出")
                        return False, None
                    time.sleep(0.01)
                    continue

                buffer.extend(chunk)

                # 检查是否找到 EOI (0xFF 0xD9)
                eoi_pos = -1
                for i in range(len(buffer) - 1):
                    if buffer[i] == 0xFF and buffer[i+1] == 0xD9:
                        eoi_pos = i
                        break

                if eoi_pos >= 0:
                    eoi_found = True
                    jpeg_data = bytes(buffer[:eoi_pos+2])  # 包含 EOI
                    # 保留剩余数据供下一帧使用
                    remaining = buffer[eoi_pos+2:]
                    buffer.clear()
                    buffer.extend(remaining)

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

    def get_frame(self, max_retries: int = 1):
        """
        获取一帧（带重试机制）

        Args:
            max_retries: 最大重试次数

        Returns:
            tuple: (success, frame)
        """
        for attempt in range(max_retries):
            ret, frame = self._read_frame()
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
