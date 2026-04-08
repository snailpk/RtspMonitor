# -*- coding:utf-8 -*-
"""
日志系统模块
统一的日志配置，支持控制台和文件输出，日志轮转功能
"""
import logging
import logging.handlers
import os
import sys
from config import Config


class Logger:
    """单例日志管理类"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config = Config()
        self._setup_logger()
        self._initialized = True

    def _setup_logger(self):
        """配置日志系统"""
        log_level = getattr(logging, self.config.get("logging", "level").upper(), logging.INFO)
        log_file = self.config.get("logging", "file")
        max_bytes = self.config.getint("logging", "max_bytes")
        backup_count = self.config.getint("logging", "backup_count")

        # 获取根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # 清除已有的处理器
        root_logger.handlers.clear()

        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器 - 处理Windows编码问题
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # 自定义编码过滤器，处理emoji字符
        class SafeEncoder(logging.Filter):
            def filter(self, record):
                # 过滤掉无法编码的字符，如emoji
                try:
                    # 尝试编码为gbk
                    record.msg.encode('gbk')
                except UnicodeEncodeError:
                    # 替换无法编码的字符为?
                    record.msg = record.msg.encode('gbk', 'replace').decode('gbk')
                return True

        console_handler.addFilter(SafeEncoder())
        root_logger.addHandler(console_handler)

        # 文件处理器（带日志轮转）
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"警告: 无法初始化文件日志处理器: {e}")

        self.logger = logging.getLogger("RTSPMonitor")

    def get_logger(self, name: str = None):
        """
        获取日志器

        Args:
            name: 日志器名称，None 则返回默认日志器

        Returns:
            logging.Logger 实例
        """
        if name:
            return logging.getLogger(name)
        return self.logger


# 全局日志器实例
_logger_instance = None


def get_logger(name: str = None):
    """
    获取日志器（便捷函数）

    Args:
        name: 日志器名称

    Returns:
        logging.Logger 实例
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    return _logger_instance.get_logger(name)
