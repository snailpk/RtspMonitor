# -*- coding:utf-8 -*-
"""
FFmpeg 诊断工具
测试 FFmpeg 是否正确安装并能读取 RTSP 流
"""
import subprocess
import sys

def test_ffmpeg():
    """测试 FFmpeg 是否可用"""
    print("=" * 60)
    print("FFmpeg 诊断工具")
    print("=" * 60)
    
    # 1. 检查 FFmpeg 版本
    print("\n1️⃣  检查 FFmpeg 是否已安装...")
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ FFmpeg 已安装")
            version_line = result.stdout.split('\n')[0]
            print(f"   版本：{version_line}")
        else:
            print("❌ FFmpeg 未正确安装")
            return False
    except FileNotFoundError:
        print("❌ 找不到 FFmpeg 程序！")
        print("\n请安装 FFmpeg:")
        print("  方法 1: choco install ffmpeg")
        print("  方法 2: 从 https://www.gyan.dev/ffmpeg/builds/ 下载并添加到 PATH")
        return False
    except Exception as e:
        print(f"❌ 检查失败：{e}")
        return False
    
    # 2. 测试 RTSP 连接
    rtsp_url = input("\n2️⃣  输入 RTSP 地址 (直接回车使用默认): ").strip()
    if not rtsp_url:
        rtsp_url = "rtsp://admin:Bossien1@192.168.0.140/h264/ch1/sub/av_stream"
    
    print(f"\n📡 测试连接：{rtsp_url}")
    print("⏳ 等待 5 秒...\n")
    
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-stimeout', '5000000',
        '-i', rtsp_url,
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # 分析错误输出
        stderr = result.stderr.lower()
        
        if 'connection refused' in stderr or 'connection failed' in stderr:
            print("❌ 连接被拒绝")
            print("   可能原因:")
            print("   - 摄像头 IP 地址错误")
            print("   - 用户名密码错误")
            print("   - 摄像头未启动")
        elif 'timeout' in stderr:
            print("❌ 连接超时")
            print("   可能原因:")
            print("   - 网络不通")
            print("   - 防火墙阻止")
        elif 'invalid data stream' in stderr or 'no streams' in stderr:
            print("❌ 无效的数据流")
            print("   可能原因:")
            print("   - RTSP 地址格式错误")
            print("   - 摄像头编码格式不支持")
        elif result.returncode != 0:
            print(f"⚠️  FFmpeg 返回错误码：{result.returncode}")
            print(f"   错误信息:\n{result.stderr[:500]}")
        else:
            print("✅ RTSP 流可以正常访问！")
            
    except subprocess.TimeoutExpired:
        print("⏱️  测试超时（可能是正常的，RTSP 需要时间建立连接）")
    except Exception as e:
        print(f"❌ 测试失败：{e}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    test_ffmpeg()
