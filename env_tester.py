"""
LectureFlow - env_tester.py
环境心跳检测脚本  (v2 · sounddevice 版)

运行方式：
    python env_tester.py

功能：
  1. 检测麦克风 - 使用 sounddevice 列出所有音频设备，
                   并尝试打开默认输入流 1 秒以验证驱动和权限。
  2. 检测 Ollama  - 向本地 Ollama 发送测试请求，确认模型可用。
"""

import sys
import json
import time
import requests


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_header(title: str):
    width = 52
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_ok(msg: str):
    print(f"  ✅  {msg}")


def print_fail(msg: str):
    print(f"  ❌  {msg}")


def print_info(msg: str):
    print(f"  ℹ️   {msg}")


# ─────────────────────────────────────────────
# 检测 1: 麦克风 (sounddevice)
# ─────────────────────────────────────────────

def check_microphone() -> bool:
    print_header("检测 1/2 · 麦克风 (sounddevice)")

    # ── 1a. 导入检查 ───────────────────────────
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as e:
        print_fail(f"依赖库未安装: {e}")
        print("       请运行: pip install sounddevice numpy")
        return False

    # ── 1b. 枚举所有设备 ──────────────────────
    try:
        devices = sd.query_devices()
    except Exception as e:
        print_fail(f"无法查询音频设备: {e}")
        return False

    input_devices = [
        (i, d) for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]

    if not input_devices:
        print_fail("未检测到任何音频输入设备。")
        return False

    print_info(f"检测到 {len(input_devices)} 个音频输入设备：")
    print()
    print(f"    {'索引':>4}  {'设备名称':<42}  {'通道':>4}  {'默认采样率':>10}")
    print(f"    {'─'*4}  {'─'*42}  {'─'*4}  {'─'*10}")

    default_in_idx = sd.default.device[0]  # 默认输入设备索引

    for idx, dev in input_devices:
        name  = dev["name"][:42]
        chs   = dev["max_input_channels"]
        rate  = int(dev["default_samplerate"])
        tag   = " ◀ 默认" if idx == default_in_idx else ""
        print(f"    [{idx:>3}]  {name:<42}  {chs:>4}  {rate:>8}Hz{tag}")

    print()

    # ── 1c. 打开默认麦克风流 1 秒 ─────────────
    print_info("正在尝试打开默认麦克风流（1 秒录音测试）...")

    captured_frames = []
    SAMPLE_RATE = 16000
    DURATION    = 1.0        # 秒
    CHANNELS    = 1
    DTYPE       = "float32"

    def _callback(indata, frames, time_info, status):
        if status:
            print(f"  ⚠️   音频回调状态: {status}")
        captured_frames.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=_callback,
        ):
            time.sleep(DURATION)

        total_samples = sum(f.shape[0] for f in captured_frames)
        rms = float(np.sqrt(np.mean(np.concatenate(captured_frames) ** 2))) if captured_frames else 0.0

        print_ok(f"麦克风流测试成功！采集样本数: {total_samples}，RMS 音量: {rms:.5f}")
        if rms < 1e-6:
            print("  ⚠️   RMS 极低，麦克风可能静音或增益为零，但驱动正常。")
        return True

    except sd.PortAudioError as e:
        print_fail(f"PortAudio 错误（驱动或权限问题）: {e}")
        return False
    except Exception as e:
        print_fail(f"打开麦克风流时出错: {e}")
        return False


# ─────────────────────────────────────────────
# 检测 2: Ollama（逻辑不变）
# ─────────────────────────────────────────────

def check_ollama(config: dict) -> bool:
    print_header("检测 2/2 · Ollama 本地模型")

    base_url = config["ollama"]["base_url"]
    model    = config["ollama"]["model"]
    api_url  = f"{base_url}{config['ollama']['api_endpoint']}"

    print_info(f"API 地址 : {api_url}")
    print_info(f"目标模型 : {model}")
    print_info("发送心跳请求，请稍等...")

    payload = {
        "model":  model,
        "prompt": "Reply with only the word: OK",
        "stream": False,
    }

    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        data  = response.json()
        reply = data.get("response", "").strip()

        if reply:
            print_ok(f"Ollama 响应正常！模型回复: \"{reply}\"")
        else:
            print_info("Ollama 连接成功，但响应内容为空（模型可能正在初始化）。")
        return True

    except requests.exceptions.ConnectionError:
        print_fail(f"无法连接到 Ollama ({base_url})。")
        print("       请确认 Ollama 正在运行: ollama serve")
        return False
    except requests.exceptions.Timeout:
        print_fail("请求超时（30秒）。模型可能尚未加载，请稍后重试。")
        return False
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "未知"
        print_fail(f"HTTP 错误 {status}。")
        if e.response is not None and e.response.status_code == 404:
            print(f"       模型 '{model}' 可能未下载。请运行: ollama pull {model}")
        return False
    except Exception as e:
        print_fail(f"未知错误: {e}")
        return False


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main():
    print("\n╔════════════════════════════════════════════════════╗")
    print("║       LectureFlow · 环境心跳检测  v2               ║")
    print("╚════════════════════════════════════════════════════╝")

    try:
        config = load_config()
        print_info("config.json 加载成功")
    except FileNotFoundError:
        print_fail("找不到 config.json，请确保脚本在项目根目录运行。")
        sys.exit(1)

    results = {
        "麦克风 (sounddevice)": check_microphone(),
        "Ollama":               check_ollama(config),
    }

    # ── 汇总报告 ──────────────────────────────
    print_header("检测汇总")
    all_pass = True
    for name, passed in results.items():
        if passed:
            print_ok(f"{name}: 正常")
        else:
            print_fail(f"{name}: 异常")
            all_pass = False

    print()
    if all_pass:
        print("  🎉  所有检测通过！可以进入第二阶段开发。")
    else:
        print("  ⚠️   部分检测未通过，请按照上方提示修复后重试。")
    print()


if __name__ == "__main__":
    main()
