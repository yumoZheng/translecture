"""
LectureFlow - ui/control_panel.py
控制中心面板

功能:
  - Whisper / LLM 模型下拉切换（读取 config.json）
  - 学科上下文文本框（实时更新 subject_context）
  - RMS 灵敏度滑动条（0.001 – 0.05）
  - 开始 / 停止翻译按钮
  - 英文字幕显示开关
  - 打开 history/ 会话目录
  - 模型下载进度条（弹出）
"""

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore    import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui     import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QPushButton, QLineEdit,
    QProgressBar, QFrame, QCheckBox, QSizePolicy,
    QGroupBox, QSpacerItem,
)


# ─────────────────────────────────────────────────────────────
# 样式常量
# ─────────────────────────────────────────────────────────────

DARK_BG     = "#0f1117"
PANEL_BG    = "#1a1d27"
CARD_BG     = "#22263a"
BORDER      = "#2e3250"
ACCENT      = "#38bdf8"
ACCENT_DARK = "#0ea5e9"
TEXT_PRI    = "#f1f5f9"
TEXT_SEC    = "#94a3b8"
SUCCESS     = "#4ade80"
DANGER      = "#f87171"

STYLESHEET = f"""
QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRI};
    font-family: 'Segoe UI', 'SF Pro Display', 'PingFang SC', sans-serif;
}}
QGroupBox {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px 14px 10px 14px;
    font-size: 12px;
    color: {TEXT_SEC};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {ACCENT};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QLabel {{
    background: transparent;
    color: {TEXT_SEC};
    font-size: 12px;
}}
QLabel#value_label {{
    color: {ACCENT};
    font-size: 12px;
    font-weight: 600;
    min-width: 42px;
}}
QComboBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 10px;
    color: {TEXT_PRI};
    font-size: 13px;
    min-height: 32px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SEC};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DARK};
    color: {TEXT_PRI};
    padding: 4px;
}}
QLineEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 10px;
    color: {TEXT_PRI};
    font-size: 13px;
    min-height: 32px;
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    border: 2px solid {DARK_BG};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QPushButton {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 18px;
    color: {TEXT_PRI};
    font-size: 13px;
    font-weight: 500;
    min-height: 36px;
}}
QPushButton:hover {{ background-color: {PANEL_BG}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background-color: {BORDER}; }}
QPushButton#btn_start {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: {DARK_BG};
    font-weight: 700;
    font-size: 14px;
}}
QPushButton#btn_start:hover {{ background-color: {ACCENT_DARK}; border-color: {ACCENT_DARK}; }}
QPushButton#btn_stop {{
    background-color: {DANGER};
    border-color: {DANGER};
    color: #fff;
    font-weight: 700;
    font-size: 14px;
}}
QPushButton#btn_stop:hover {{ background-color: #ef4444; }}
QProgressBar {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 10px;
    text-align: center;
    font-size: 11px;
    color: {TEXT_SEC};
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {ACCENT}, stop:1 {SUCCESS});
    border-radius: 6px;
}}
QCheckBox {{
    color: {TEXT_SEC};
    font-size: 12px;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {PANEL_BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
"""


# ─────────────────────────────────────────────────────────────
# 控制面板
# ─────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    """
    LectureFlow 控制中心。

    发出的信号:
        sig_start_translation  → 用户点击"开始翻译"
        sig_stop_translation   → 用户点击"停止翻译"
        sig_subject_changed    → (str) 学科上下文文本变化
        sig_rms_changed        → (float) RMS 阈值变化
        sig_show_english       → (bool) 英文字幕显示开关
        sig_switch_whisper     → (str)  用户选择新 Whisper 模型
        sig_switch_llm         → (str)  用户选择新 LLM 模型
    """

    sig_start_translation = pyqtSignal()
    sig_stop_translation  = pyqtSignal()
    sig_subject_changed   = pyqtSignal(str)
    sig_rms_changed       = pyqtSignal(float)
    sig_show_english      = pyqtSignal(bool)
    sig_switch_whisper    = pyqtSignal(str)
    sig_switch_llm        = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config   = config
        self._running  = False

        self.setWindowTitle("LectureFlow · 控制中心")
        self.setMinimumWidth(420)
        self.setMaximumWidth(520)
        self.setStyleSheet(STYLESHEET)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )

        self._build_ui()

    # ── UI 构建 ───────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        # ── 标题栏 ──────────────────────────────────────────
        title = QLabel("🎧  LectureFlow")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {TEXT_PRI}; background: transparent;"
        )
        subtitle = QLabel("实时讲座翻译控制中心")
        subtitle.setStyleSheet(f"font-size: 12px; color: {TEXT_SEC}; background: transparent;")

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addSpacing(4)

        # ── 模型选择 ────────────────────────────────────────
        model_group = QGroupBox("模型配置")
        mg_layout   = QVBoxLayout(model_group)
        mg_layout.setSpacing(10)

        # Whisper
        whisper_row    = QHBoxLayout()
        whisper_label  = QLabel("Whisper ASR")
        self._whisper_combo = QComboBox()
        for m in self._config.get("available_whisper_models", []):
            label = f"{m['name']}  —  {m['ram_requirement']}"
            self._whisper_combo.addItem(label, userData=m["id"])
        # 【修复 1】先 setCurrentIndex，再连接信号，防止构造期间误触发
        current_whisper = self._config.get("whisper", {}).get("current_model", "small")
        for i in range(self._whisper_combo.count()):
            if self._whisper_combo.itemData(i) == current_whisper:
                self._whisper_combo.setCurrentIndex(i)
                break
        self._whisper_combo.currentIndexChanged.connect(self._on_whisper_changed)  # 开始监听
        whisper_row.addWidget(whisper_label, 1)
        whisper_row.addWidget(self._whisper_combo, 2)
        mg_layout.addLayout(whisper_row)

        # LLM
        llm_row    = QHBoxLayout()
        llm_label  = QLabel("翻译 LLM")
        self._llm_combo = QComboBox()
        for m in self._config.get("available_llm_models", []):
            label = f"{m['name']}  —  {m['ram_requirement']}"
            self._llm_combo.addItem(label, userData=m["id"])
        # 【修复 1】先 setCurrentIndex，再连接信号
        current_llm = self._config.get("ollama", {}).get("current_model", "gemma3:1b")
        for i in range(self._llm_combo.count()):
            if self._llm_combo.itemData(i) == current_llm:
                self._llm_combo.setCurrentIndex(i)
                break
        self._llm_combo.currentIndexChanged.connect(self._on_llm_changed)  # 开始监听
        llm_row.addWidget(llm_label, 1)
        llm_row.addWidget(self._llm_combo, 2)
        mg_layout.addLayout(llm_row)

        # 下载进度条（默认隐藏）
        self._dl_label = QLabel("正在下载模型...")
        self._dl_label.setStyleSheet(f"color: {ACCENT}; font-size: 12px; background: transparent;")
        self._dl_bar   = QProgressBar()
        self._dl_bar.setRange(0, 100)
        self._dl_bar.setValue(0)
        self._dl_label.hide()
        self._dl_bar.hide()
        mg_layout.addWidget(self._dl_label)
        mg_layout.addWidget(self._dl_bar)

        root.addWidget(model_group)

        # ── 翻译设置 ────────────────────────────────────────
        settings_group = QGroupBox("翻译设置")
        sg_layout      = QVBoxLayout(settings_group)
        sg_layout.setSpacing(10)

        # 学科上下文
        subject_row   = QHBoxLayout()
        subject_label = QLabel("学科上下文")
        self._subject_edit = QLineEdit()
        self._subject_edit.setPlaceholderText("如：深度学习、量子力学 …")
        self._subject_edit.textChanged.connect(self.sig_subject_changed)
        subject_row.addWidget(subject_label, 1)
        subject_row.addWidget(self._subject_edit, 2)
        sg_layout.addLayout(subject_row)

        # RMS 灵敏度滑动条
        rms_row     = QHBoxLayout()
        rms_label   = QLabel("麦克风灵敏度")
        self._rms_slider = QSlider(Qt.Orientation.Horizontal)
        # 内部整数：1–50 对应 0.001–0.050（步长 0.001）
        self._rms_slider.setRange(1, 50)
        self._rms_slider.setValue(10)   # 默认 0.010
        self._rms_slider.setTickInterval(5)
        self._rms_val_label = QLabel("0.010")
        self._rms_val_label.setObjectName("value_label")
        self._rms_slider.valueChanged.connect(self._on_rms_changed)
        rms_row.addWidget(rms_label, 1)
        rms_row.addWidget(self._rms_slider, 3)
        rms_row.addWidget(self._rms_val_label)
        sg_layout.addLayout(rms_row)

        # 英文字幕开关
        self._show_en_cb = QCheckBox("显示英文原文字幕")
        self._show_en_cb.setChecked(True)
        self._show_en_cb.toggled.connect(self.sig_show_english)
        sg_layout.addWidget(self._show_en_cb)

        root.addWidget(settings_group)

        # ── 会话管理 ────────────────────────────────────────
        session_group  = QGroupBox("会话管理")
        sess_layout    = QHBoxLayout(session_group)
        btn_open_hist  = QPushButton("📂  打开 history/ 文件夹")
        btn_open_hist.clicked.connect(self._open_history_folder)
        sess_layout.addWidget(btn_open_hist)
        root.addWidget(session_group)

        # ── 主控按钮 ────────────────────────────────────────
        root.addSpacing(4)
        self._btn_start = QPushButton("▶  开始翻译")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.clicked.connect(self._on_start_clicked)

        self._btn_stop = QPushButton("⏹  停止翻译")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.clicked.connect(self._on_stop_clicked)
        self._btn_stop.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        root.addLayout(btn_row)

        # ── 状态栏 ──────────────────────────────────────────
        self._status_label = QLabel("就绪")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            f"color: {TEXT_SEC}; font-size: 11px; background: transparent;"
        )
        root.addWidget(self._status_label)

        self.setLayout(root)
        self.adjustSize()

    # ── 信号处理 ──────────────────────────────────────────────

    def _on_whisper_changed(self, index: int):
        model_id = self._whisper_combo.itemData(index)
        if model_id:
            self.sig_switch_whisper.emit(model_id)

    def _on_llm_changed(self, index: int):
        model_id = self._llm_combo.itemData(index)
        if model_id:
            self.sig_switch_llm.emit(model_id)

    def _on_rms_changed(self, value: int):
        rms = value / 1000.0
        self._rms_val_label.setText(f"{rms:.3f}")
        self.sig_rms_changed.emit(rms)

    def _on_start_clicked(self):
        self._running = True
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self.set_status("🔴  录音中...")
        self.sig_start_translation.emit()

    def _on_stop_clicked(self):
        self._running = False
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self.set_status("就绪")
        self.sig_stop_translation.emit()

    def _open_history_folder(self):
        history_dir = Path("history")
        history_dir.mkdir(exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(history_dir.resolve())])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(history_dir.resolve())])
        else:
            subprocess.Popen(["xdg-open", str(history_dir.resolve())])

    # ── 公开方法（由 main_gui 调用） ──────────────────────────

    def set_status(self, text: str):
        self._status_label.setText(text)

    @pyqtSlot(float, str)
    def set_download_progress(self, pct: float, speed: str = ""):
        """更新下载进度条（0–100）。pct < 0 时隐藏。安全地接受跨线程信号。"""
        if pct < 0:
            self._dl_label.hide()
            self._dl_bar.hide()
            return
        self._dl_label.show()
        self._dl_bar.show()
        self._dl_bar.setValue(int(pct))
        label_text = f"正在下载模型...  {speed}" if speed else "正在下载模型..."
        self._dl_label.setText(label_text)

    def get_subject(self) -> str:
        return self._subject_edit.text().strip()

    def get_rms(self) -> float:
        return self._rms_slider.value() / 1000.0
