"""
LectureFlow - main.py
CLI MVP 主程序 v4 (会话分文存储版)

运行:
    python main.py
    python main.py --whisper_model small --llm_model qwen2.5:3b
    python main.py --device 1 --save-audio --subject "深度学习"
    python main.py --flush 3.0 --rms 0.005

模型优先级: 命令行参数 > config.json 的 current_model
"""

import argparse
import json
import queue
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from core.audio_engine import AudioEngine
from core.translator import Translator


# ─────────────────────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# 历史记录（会话分文模式）
# ─────────────────────────────────────────────────────────────

HISTORY_DIR = Path("history")


def _create_session_file() -> Path:
    """
    在 history/ 目录下创建本次会话的 Markdown 文件。

    命名格式: Lecture_YYYYMMDD_HHMMSS.md
    自动写入元数据标题行。

    返回该文件的 Path 对象（会话内全程共用）。
    """
    HISTORY_DIR.mkdir(exist_ok=True)
    now      = datetime.now()
    filename = now.strftime("Lecture_%Y%m%d_%H%M%S.md")
    path     = HISTORY_DIR / filename

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    header   = f"# Lecture Note - {date_str} {time_str}\n\n"
    path.write_text(header, encoding="utf-8")
    return path


# 全局会话文件路径（在 main() 中调用 _create_session_file() 后赋值）
SESSION_FILE: Path | None = None


def append_history(en: str, cn: str):
    """Append one translated sentence to the current session file."""
    if SESSION_FILE is None:
        return
    ts    = datetime.now().strftime("%H:%M:%S")
    entry = f"{ts} - EN: {en}\n      CN: {cn}\n\n"
    with SESSION_FILE.open("a", encoding="utf-8") as f:
        f.write(entry)


# ─────────────────────────────────────────────────────────────
# LLM 模型体检（启动门控）
# ─────────────────────────────────────────────────────────────

def run_model_health_check(translator: "Translator", model_name: str) -> bool:
    """
    启动前检查 LLM 模型是否已安装。
    若未安装，询问用户是否立即下载。

    返回:
        True  — 模型就绪，可以继续启动
        False — 用户拒绝或下载失败，应退出程序

    UI 对接 (Phase 3):
        将 input() 替换为弹出对话框，
        将 download_llm_model() 的 progress_callback 绑定到进度条组件。
    """
    print("\n  🔍  正在检查 LLM 模型...")
    exists = translator.ensure_llm_model_exists(model_name)

    if exists:
        return True

    # ── 模型缺失：询问用户 ────────────────────────────────────
    print()
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  ⚠️   模型 [{model_name}] 尚未下载              │")
    print(f"  │  是否立即自动下载并安装？                        │")
    print(f"  └─────────────────────────────────────────────────┘")

    try:
        choice = input("  请输入 y 下载，n 退出: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  ⏹  已取消。")
        return False

    if choice != "y":
        print(f"  ℹ️   未下载。请手动运行: ollama pull {model_name}")
        return False

    # ── 用户同意：开始下载 ────────────────────────────────────
    print()
    ok = translator.download_llm_model(model_name)
    if not ok:
        print("  ❌  下载失败，程序退出。请检查网络或 Ollama 状态。")
    return ok


# ─────────────────────────────────────────────────────────────
# UI 回调槽（Phase 3 对接 OverlayWindow 信号）
# ─────────────────────────────────────────────────────────────

def on_subtitle_update(data: dict):
    """
    每句翻译完成后被调用。

    data = {"en": str, "cn": str, "timestamp": str}

    Phase 3: 替换为向 PyQt6 OverlayWindow 发送信号的实现。
    """
    pass   # CLI 版已在 consumer_loop 直接打印，此处留空


# ─────────────────────────────────────────────────────────────
# 消费者线程
# ─────────────────────────────────────────────────────────────

def consumer_loop(
    audio_queue: queue.Queue,
    translator:  Translator,
    subject:     str,
    stop_event:  threading.Event,
):
    """
    两阶段流式输出：
      1. Whisper 转录完成 → 立即打印 [EN]（用户不需等翻译）
      2. Ollama 翻译完成 → 打印 [CN]，写入历史，触发 UI 回调
    """
    bar = "─" * 58

    while not stop_event.is_set():
        try:
            chunk: np.ndarray = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 第一步：Whisper ASR（立即反馈）─────────────────────
        en = translator.transcribe(chunk)
        if not en:
            continue

        print(f"\n  {bar}")
        print(f"  ⏱  {ts}")
        print(f"  🇬🇧  {en}")
        print(f"  ⏳  翻译中...", end="", flush=True)

        # ── 第二步：Ollama 翻译（稍慢，完成后补打）──────────────
        cn = translator.llm_translate(en, subject_context=subject)

        # 用回车覆盖"翻译中..."行，打印中文
        print(f"\r  🇨🇳  {cn}")
        print(f"  {bar}")

        # ① UI 回调槽
        on_subtitle_update({"en": en, "cn": cn, "timestamp": ts})

        # ② 历史记录
        append_history(en, cn)



# ─────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LectureFlow CLI v3 — 实时讲座翻译")
    p.add_argument("--device",        type=int,   default=None,          help="麦克风索引（默认自动）")
    p.add_argument("--save-audio",    action="store_true",                help="保存 WAV 录音")
    p.add_argument("--subject",       type=str,   default="",            help="课程主题（如 '机器学习'）")
    p.add_argument("--flush",         type=float, default=4.0,           help="切片间隔（秒，默认 4.0）")
    p.add_argument("--rms",           type=float, default=0.01,          help="静音阈值 RMS（默认 0.01）")
    p.add_argument("--config",        type=str,   default="config.json", help="配置文件路径")
    p.add_argument("--whisper_model", type=str,   default=None,
                   help="Whisper 模型 (tiny/base/small/medium/large-v3)；覆盖 config.json")
    p.add_argument("--llm_model",     type=str,   default=None,
                   help="Ollama 模型 (如 gemma3:1b / qwen2.5:3b)；覆盖 config.json")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def main():
    global SESSION_FILE

    args   = parse_args()
    config = load_config(args.config)

    # ── 创建本次会话历史文件 ───────────────────────────────
    SESSION_FILE = _create_session_file()

    print("\n╔════════════════════════════════════════════════════╗")
    print("║    LectureFlow · 实时讲座翻译  CLI v4 (会话分文版)   ║")
    print("╚════════════════════════════════════════════════════╝\n")
    print(f"  📚  主题上下文 : {args.subject  or '(未指定)'}")
    print(f"  🎙  麦克风索引 : {args.device   or '自动'}")
    print(f"  ⏱  切片间隔   : {args.flush}s")
    print(f"  🔊  静音阈值   : {args.rms}")
    print(f"  💾  录音保存   : {'是' if args.save_audio else '否'}")
    print(f"  📝  本次记录将保存至: {SESSION_FILE.resolve()}")
    print("\n  按 Ctrl+C 停止...\n")
    print("  " + "─" * 56)

    # ── 初始化 ──────────────────────────────────────────────
    audio_queue = queue.Queue()
    stop_event  = threading.Event()

    translator = Translator(config)

    # ── 命令行模型覆盖（优先级高于 config.json）──────────────
    # Whisper 覆盖
    if args.whisper_model and args.whisper_model != translator.current_whisper_model:
        print(f"\n  🔁  命令行指定 Whisper 模型: {args.whisper_model}")
        ok = translator.switch_whisper_model(args.whisper_model)
        if not ok:
            print("  ⚠️   模型切换失败，将使用 config.json 中的默认模型继续运行。")

    # LLM 覆盖
    if args.llm_model and args.llm_model != translator.ollama_model:
        print(f"  🔁  命令行指定 LLM 模型: {args.llm_model}")
        translator.switch_llm_model(args.llm_model)

    print(f"  🧠  Whisper 模型 : {translator.current_whisper_model}")
    print(f"  🤖  LLM 模型     : {translator.ollama_model}")
    print()

    # ── LLM 模型体检（门控：不通过则退出）────────────────────
    if not run_model_health_check(translator, translator.ollama_model):
        sys.exit(1)

    engine = AudioEngine(
        audio_queue    = audio_queue,
        device_index   = args.device,
        sample_rate    = config["audio"]["sample_rate"],
        flush_interval = args.flush,
        save_audio     = args.save_audio,
    )
    engine.rms_threshold = args.rms   # 可在运行期间由 UI 滑块覆盖

    # ── Ctrl+C 优雅退出 ───────────────────────────────────────
    def _sigint_handler(sig, frame):
        print("\n\n  ⏹  正在停止，请稍候...")
        stop_event.set()
        engine.stop()

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── 启动线程 ──────────────────────────────────────────────
    engine.start()

    consumer_thread = threading.Thread(
        target  = consumer_loop,
        args    = (audio_queue, translator, args.subject, stop_event),
        daemon  = True,
        name    = "TranslatorConsumer",
    )
    consumer_thread.start()

    stop_event.wait()
    consumer_thread.join(timeout=5)

    print(f"\n  ✅  已退出。本次讲座记录已保存至: {SESSION_FILE.resolve()}\n")



if __name__ == "__main__":
    main()
