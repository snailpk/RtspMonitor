# -*- coding:utf-8 -*-
"""
FFmpeg 连接测试工具
用于诊断 RTSP 流连接问题
"""
import subprocess
import sys

def test_ffmpeg():
    """测试 FFmpeg 是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg 已安装")
            print(f"   版本：{result.stdout.split('\\n')[0]}")
            return True
        else:
            print("❌ FFmpeg 执行失败")
            return False
    except FileNotFoundError:
        print("❌ 找不到 FFmpeg 命令")
        print("   请安装 FFmpeg 并添加到 PATH")
        print("   下载地址：https://www.gyan.dev/ffmpeg/builds/")
        return False
    except Exception as e:
        print(f"❌ 测试异常：{e}")
        return False

def test_rtsp(rtsp_url):
    """测试 RTSP 流连接"""
    print(f"\n📡 测试 RTSP 连接...")
    print(f"   URL: {rtsp_url}\n")
    
    # 简化的 FFmpeg 命令（最少参数）
    cmd_simple = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-f', 'null',
        '-'
    ]
    
    print("🔧 使用简化参数测试...")
    try:
        process = subprocess.Popen(
            cmd_simple,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL
        )
        
        # 等待 5 秒
        import time
        time.sleep(5)
        
        # 检查进程状态
        if process.poll() is None:
            print("✅ FFmpeg 进程运行正常（5秒后）")
            print("   正在读取流数据...")
            
            # 再等 5 秒
            time.sleep(5)
            
            if process.poll() is None:
                print("✅ 连接成功！RTSP 流正常工作")
                process.terminate()
                process.wait(timeout=3)
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ FFmpeg 进程在 10 秒后退出")
                print(f"   返回码：{process.returncode}")
                if stderr:
                    print(f"\n错误信息：")
                    for line in stderr.decode('utf-8', errors='ignore').split('\n')[:15]:
                        if line.strip():
                            print(f"  {line}")
                return False
        else:
            stdout, stderr = process.communicate()
            print(f"❌ FFmpeg 进程立即退出")
            print(f"   返回码：{process.returncode}")
            if stderr:
                print(f"\n错误信息：")
                for line in stderr.decode('utf-8', errors='ignore').split('\n')[:15]:
                    if line.strip():
                        print(f"  {line}")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("FFmpeg RTSP 连接诊断工具")
    print("=" * 60)
    
    # 测试 FFmpeg
    if not test_ffmpeg():
        sys.exit(1)
    
    # 测试 RTSP
    if len(sys.argv) > 1:
        rtsp_url = sys.argv[1]
    else:
        rtsp_url = "rtsp://admin:Bossien1@192.168.0.140/h264/ch1/sub/av_stream"
        print(f"\n使用默认 URL: {rtsp_url}")
    
    success = test_rtsp(rtsp_url)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 诊断完成：RTSP 流连接正常")
    else:
        print("❌ 诊断完成：RTSP 流连接失败")
        print("\n建议：")
        print("  1. 检查摄像头是否在线")
        print("  2. 确认 RTSP URL 是否正确")
        print("  3. 检查网络连接")
        print("  4. 尝试其他子码流路径")
    print("=" * 60)
