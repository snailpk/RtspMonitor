# -*- coding:utf-8 -*-
"""
配置管理模块
使用 configparser 读取 INI 配置文件
支持默认配置和用户配置的合并
提供类型安全的配置访问
"""
import configparser
import os
from typing import Optional


class Config:
    """配置管理类"""

    def __init__(self, config_file: str = "config.ini"):
        """
        初始化配置管理

        Args:
            config_file: 配置文件路径，默认 config.ini
        """
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self._load_config()

    def _load_config(self):
        """加载配置"""
        # 默认配置
        self._set_defaults()

        # 读取用户配置文件（使用 UTF-8 编码支持中文注释）
        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')

    def _set_defaults(self):
        """设置默认配置"""
        # RTSP 配置
        if not self.config.has_section("rtsp"):
            self.config.add_section("rtsp")
        if not self.config.has_option("rtsp", "url"):
            self.config.set("rtsp", "url", "")
        if not self.config.has_option("rtsp", "transport"):
            self.config.set("rtsp", "transport", "tcp")
        if not self.config.has_option("rtsp", "buffer_size"):
            self.config.set("rtsp", "buffer_size", "1048576")

        # 抓图配置
        if not self.config.has_section("capture"):
            self.config.add_section("capture")
        if not self.config.has_option("capture", "fps"):
            self.config.set("capture", "fps", "1")
        if not self.config.has_option("capture", "save_frames"):
            self.config.set("capture", "save_frames", "False")
        if not self.config.has_option("capture", "quality"):
            self.config.set("capture", "quality", "5")

        # 检测配置
        if not self.config.has_section("detection"):
            self.config.add_section("detection")
        if not self.config.has_option("detection", "threshold"):
            self.config.set("detection", "threshold", "1000")
        if not self.config.has_option("detection", "min_contour_area"):
            self.config.set("detection", "min_contour_area", "0.001")
        if not self.config.has_option("detection", "stable_frames"):
            self.config.set("detection", "stable_frames", "2")
        if not self.config.has_option("detection", "save_screenshot"):
            self.config.set("detection", "save_screenshot", "True")

        # 日志配置
        if not self.config.has_section("logging"):
            self.config.add_section("logging")
        if not self.config.has_option("logging", "level"):
            self.config.set("logging", "level", "INFO")
        if not self.config.has_option("logging", "file"):
            self.config.set("logging", "file", "rtsp_monitor.log")
        if not self.config.has_option("logging", "max_bytes"):
            self.config.set("logging", "max_bytes", "10485760")
        if not self.config.has_option("logging", "backup_count"):
            self.config.set("logging", "backup_count", "5")

        # 性能监控配置
        if not self.config.has_section("monitoring"):
            self.config.add_section("monitoring")
        if not self.config.has_option("monitoring", "enabled"):
            self.config.set("monitoring", "enabled", "True")
        if not self.config.has_option("monitoring", "interval"):
            self.config.set("monitoring", "interval", "60")

    def get(self, section: str, option: str, fallback: Optional[str] = None) -> str:
        """
        获取字符串配置值

        Args:
            section: 配置节
            option: 配置项
            fallback: 备选值

        Returns:
            配置值
        """
        return self.config.get(section, option, fallback=fallback)

    def getint(self, section: str, option: str, fallback: int = 0) -> int:
        """
        获取整数配置值

        Args:
            section: 配置节
            option: 配置项
            fallback: 备选值

        Returns:
            配置值
        """
        return self.config.getint(section, option, fallback=fallback)

    def getfloat(self, section: str, option: str, fallback: float = 0.0) -> float:
        """
        获取浮点数配置值

        Args:
            section: 配置节
            option: 配置项
            fallback: 备选值

        Returns:
            配置值
        """
        return self.config.getfloat(section, option, fallback=fallback)

    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        """
        获取布尔值配置值

        Args:
            section: 配置节
            option: 配置项
            fallback: 备选值

        Returns:
            配置值
        """
        return self.config.getboolean(section, option, fallback=fallback)

    def set(self, section: str, option: str, value: str) -> None:
        """
        设置配置值

        Args:
            section: 配置节
            option: 配置项
            value: 值
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, value)

    def save(self) -> None:
        """保存配置到文件"""
        with open(self.config_file, "w", encoding="utf-8") as f:
            self.config.write(f)

    def reload(self) -> None:
        """重新加载配置"""
        self._load_config()
