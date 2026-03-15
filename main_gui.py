"""
LectureFlow - main_gui.py
PyQt6 双窗口入口 (Phase 3)

架构:
  ┌─────────────────────────────────────────────┐
  │               QApplication                  │
  │                                             │
  │  ControlPanel ──signals──► AppController    │
  │                              │              │
  │  SubtitleOverlay ◄──signal── TranslatorWorker (QThread)
  │                              │              │
  │                          AudioEngine        │
  └─────────────────────────────────────────────┘

线程模型:
  - 主线程     : Qt 事件循环 + UI 渲染
  - AudioFlushTimer (AudioEngine 内部 daemon 线程) : 定时推送音频块
  - TranslatorWorker (QThread)                    : Whisper + Ollama IO
"""

import os
import sys

# 【核心修复】防止 OpenMP 库冲突导致的闪退（必须放在最前面）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 如确认 CPU 不支持 AVX2+ 可取消注释，默认让 ctranslate2 自动探测最佳指令集
# os.environ["CTRANSLATE2_INSTRUCTIONS_SET"] = "AVX"

# 【最高优先级】在任何其他导入之前，彻底解决权限和 DPI 问题
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import queue
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore    import QThread, pyqtSignal, QObject, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

# 延迟导入自己的模块，确保环境变量已生效
from core.audio_engine import AudioEngine
from core.translator   import Translator
from ui.overlay_window import SubtitleOverlay
from ui.control_panel  import ControlPanel

# 在文件顶层定义一个全局变量，防止被自动回收
global_controller = None

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


HISTORY_DIR = Path("history")


def create_session_file() -> Path:
    HISTORY_DIR.mkdir(exist_ok=True)
    now      = datetime.now()
    filename = now.strftime("Lecture_%Y%m%d_%H%M%S.md")
    path     = HISTORY_DIR / filename
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    path.write_text(f"# Lecture Note - {date_str} {time_str}\n\n", encoding="utf-8")
    return path


def append_history(session_file: Path, en: str, cn: str):
    ts    = datetime.now().strftime("%H:%M:%S")
    entry = f"{ts} - EN: {en}\n      CN: {cn}\n\n"
    with session_file.open("a", encoding="utf-8") as f:
        f.write(entry)

class ModelLoader(QThread):
    """专门负责程序启动时后台加载模型，不卡死 UI"""
    sig_finished = pyqtSignal(object)
    sig_error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            # 真正的后台加载
            t = Translator(self.config)
            self.sig_finished.emit(t)
        except Exception as e:
            self.sig_error.emit(str(e))

# ─────────────────────────────────────────────────────────────
# TranslatorWorker — QThread 包装消费者循环
# ─────────────────────────────────────────────────────────────

class TranslatorWorker(QThread):
    """
    在独立线程中消费音频队列，执行 Whisper ASR + Ollama 翻译。

    发出的信号:
        sig_asr_ready   (str)        — Whisper 转录完成（立即）
        sig_subtitle    (str, str)   — (en, cn) 翻译完成
        sig_status      (str)        — 状态文本更新
    """

    sig_asr_ready = pyqtSignal(str)
    sig_subtitle  = pyqtSignal(str, str)
    sig_status    = pyqtSignal(str)

    def __init__(
        self,
        audio_queue:  queue.Queue,
        translator:   Translator,
        session_file: Path,
        parent=None,
    ):
        super().__init__(parent)
        self._queue        = audio_queue
        self._translator   = translator
        self._session_file = session_file
        self._subject      = ""
        self._stop_flag    = False

    def set_subject(self, subject: str):
        self._subject = subject

    def stop(self):
        self._stop_flag = True

    def run(self):
        """QThread.run() — 在工作线程中执行。"""
        self._stop_flag = False

        while not self._stop_flag:
            try:
                chunk: np.ndarray = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── Step 1: Whisper ASR（立即通知 UI）────────────
            self.sig_status.emit("🎤  识别中...")
            en = self._translator.transcribe(chunk)
            if not en:
                self.sig_status.emit("🔴  录音中...")
                continue

            self.sig_asr_ready.emit(en)
            self.sig_status.emit("🌏  翻译中...")

            # ── Step 2: Ollama 翻译（稍慢）──────────────────
            cn = self._translator.llm_translate(en, self._subject)

            # ── Step 3: 推送结果 & 持久化 ────────────────────
            self.sig_subtitle.emit(en, cn)
            self.sig_status.emit("🔴  录音中...")
            append_history(self._session_file, en, cn)


# ─────────────────────────────────────────────────────────────
# AppController — 协调所有组件
# ─────────────────────────────────────────────────────────────

class AppController(QObject):
    def __init__(self, config: dict):
        super().__init__()
        self._config       = config
        self._audio_queue  = queue.Queue()
        self._session_file: Path | None = None
        self._translator   = None
        self._worker       = None # 预定义，防止报错

        # 1. 初始化音频引擎
        self._engine = AudioEngine(
            audio_queue    = self._audio_queue,
            sample_rate    = config["audio"]["sample_rate"],
            flush_interval = 4.0,
        )

        # 2. 初始化 UI 窗口
        self._panel   = ControlPanel(config)
        self._overlay = SubtitleOverlay(show_english=True)

        self._connect_signals()
        
        # 初始禁用开始按钮，直到模型加载完成
        self._panel._btn_start.setEnabled(False)
        self._panel.show()
        
        # 3. 开启真正的后台线程加载模型
        self._panel.set_status("🔄 正在初始化翻译引擎 (请稍候)...")
        self._loader = ModelLoader(config)
        self._loader.sig_finished.connect(self._on_init_finished)
        self._loader.sig_error.connect(self._on_init_error)
        self._loader.start()

    def _on_init_finished(self, translator):
        """模型加载成功后的回调"""
        self._translator = translator
        self._panel.set_status("就绪")
        self._panel._btn_start.setEnabled(True)
        print("[System] ✅ 翻译引擎已在后台就绪。")

    def _on_init_error(self, err_msg):
        """模型加载失败后的回调"""
        self._panel.set_status("❌ 初始化失败")
        QMessageBox.critical(self._panel, "核心错误", f"模型加载失败: {err_msg}")

    # ── 信号配线 ──────────────────────────────────────────────

    def _connect_signals(self):
        p = self._panel

        p.sig_start_translation.connect(self._on_start)
        p.sig_stop_translation.connect(self._on_stop)
        p.sig_subject_changed.connect(self._on_subject_changed)
        p.sig_rms_changed.connect(self._on_rms_changed)
        p.sig_show_english.connect(self._overlay.set_show_english)
        p.sig_switch_whisper.connect(self._on_switch_whisper)
        p.sig_switch_llm.connect(self._on_switch_llm)

    # ── 槽函数 ────────────────────────────────────────────────

    def _on_start(self):
        # 创建新会话文件
        self._session_file = create_session_file()
        self._panel.set_status(f"📝  {self._session_file.name}")

        # 清空音频队列（防止上次残留数据）
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # 启动工作线程
        self._worker = TranslatorWorker(
            audio_queue  = self._audio_queue,
            translator   = self._translator,
            session_file = self._session_file,
        )
        self._worker.set_subject(self._panel.get_subject())
        self._worker.sig_asr_ready.connect(self._on_asr_ready)
        self._worker.sig_subtitle.connect(self._on_subtitle)
        self._worker.sig_status.connect(self._panel.set_status)
        self._worker.start()

        # 启动音频采集
        self._engine.rms_threshold = self._panel.get_rms()
        self._engine.start()

        # 显示字幕条
        self._overlay.update_subtitle(en="", cn="● 开始翻译")
        self._overlay.show()

    def _on_stop(self):
        self._engine.stop()

        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        self._overlay.update_subtitle(en="", cn="")
        self._overlay.hide()

        if self._session_file:
            self._panel.set_status(
                f"✅  已保存: {self._session_file.name}"
            )

    def _on_asr_ready(self, en: str):
        """Whisper 完成后立即在字幕条显示英文，中文显示省略号。"""
        self._overlay.update_subtitle(en=en, cn="…")

    def _on_subtitle(self, en: str, cn: str):
        """翻译完成后更新字幕条。"""
        self._overlay.update_subtitle(en=en, cn=cn)

    def _on_subject_changed(self, subject: str):
        if self._worker:
            self._worker.set_subject(subject)

    def _on_rms_changed(self, rms: float):
        self._engine.rms_threshold = rms

    def _on_switch_whisper(self, model_id: str):
        """在工作线程停止期间切换 Whisper 模型（防止并发冲突）。"""
        # 【修复 1】如果 Translator 尚未就绪，忧断退出
        if not self._translator:
            return

        was_running = self._engine.is_recording
        if was_running:
            self._on_stop()

        self._panel.set_status(f"🔄  切换 Whisper → {model_id} ...")
        ok = self._translator.switch_whisper_model(model_id)

        if ok:
            self._panel.set_status(f"✅  Whisper 已切换: {model_id}")
        else:
            self._panel.set_status(f"❌  Whisper 切换失败")
            QMessageBox.warning(
                self._panel, "切换失败",
                f"Whisper 模型 [{model_id}] 加载失败，请检查控制台输出。"
            )

        if was_running and ok:
            self._on_start()

    def _on_switch_llm(self, model_id: str):
        """LLM 切换为零开销热更新，检查模型是否已安装。"""
        # 【修复 1】如果 Translator 尚未就绪，忧断退出
        if not self._translator:
            return

        self._panel.set_status(f"🔍  检查 LLM 模型: {model_id} ...")
        exists = self._translator.ensure_llm_model_exists(model_id)

        if exists:
            self._translator.switch_llm_model(model_id)
            self._panel.set_status(f"✅  LLM 已切换: {model_id}")
            return

        # 模型未安装 → 弹出确认框
        reply = QMessageBox.question(
            self._panel,
            "模型未安装",
            f"模型 [{model_id}] 尚未下载。\n是否立即自动下载并安装？\n（下载过程中可继续使用当前模型）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._panel.set_status("下载已取消")
            return

        # 【修复 4】开启下载线程，通过 pyqtSignal 安全更新进度条
        self._panel.set_download_progress(0)

        class _DownloadThread(QThread):
            sig_progress = pyqtSignal(float, str)   # 【修复 4】替换 QMetaObject.invokeMethod
            finished     = pyqtSignal(bool)

            def __init__(self, translator, model_id):
                super().__init__()
                self._t = translator
                self._m = model_id

            def run(self):
                def _cb(pct: float, speed: str):
                    self.sig_progress.emit(pct, speed)  # Qt 信号自动跨线程排队
                ok = self._t.download_llm_model(self._m, _cb)
                self.finished.emit(ok)

        self._dl_thread = _DownloadThread(self._translator, model_id)
        # 连接进度信号（有了 @pyqtSlot 装饰器，跨线程调用安全）
        self._dl_thread.sig_progress.connect(self._panel.set_download_progress)

        def _on_dl_done(ok: bool):
            self._panel.set_download_progress(-1)   # 隐藏进度条
            if ok:
                self._translator.switch_llm_model(model_id)
                self._panel.set_status(f"✅  LLM 已下载并切换: {model_id}")
            else:
                self._panel.set_status("❌  下载失败，请检查网络和 Ollama 状态")

        self._dl_thread.finished.connect(_on_dl_done)
        self._dl_thread.start()


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def main():
    global global_controller # 声明我们要使用全局变量

    # 1. 设置 DPI 策略 (必须在 QApplication 实例化之前)
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QGuiApplication
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception as e:
        print(f"DPI 设置警告: {e}")

    app = QApplication(sys.argv)
    app.setApplicationName("LectureFlow")
    app.setStyle("Fusion")

    # 2. 安全加载配置与控制器
    try:
        print("[System] 正在读取配置文件...")
        current_config = load_config()
        
        print("[System] 正在启动核心控制器 (加载模型中，请稍候)...")
        # 将实例赋值给全局变量，这样它在整个程序运行期间都不会被销毁
        global_controller = AppController(current_config)
        
        print("[System] 界面已就绪。")
        # 开始 Qt 的事件循环
        sys.exit(app.exec())
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n[致命错误] 程序启动失败: {error_msg}")
        
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setWindowTitle("启动失败")
        error_box.setText("LectureFlow 无法启动")
        error_box.setInformativeText(f"错误详情: {error_msg}")
        error_box.exec()
        sys.exit(1)

if __name__ == "__main__":
    main()