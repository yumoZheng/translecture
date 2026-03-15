"""
LectureFlow - ui/overlay_window.py
字幕悬浮层

特性:
  - 无边框、窗口置顶、背景半透明
  - 鼠标点击穿透（不拦截鼠标事件，点击直达下方窗口）
  - 双行字幕：上英下中
  - 支持动态显示/隐藏英文行
  - 支持拖动重定位（按住 Alt + 左键）
"""

from PyQt6.QtCore    import Qt, QPoint
from PyQt6.QtGui     import QColor, QPainter, QPainterPath, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication


class SubtitleOverlay(QWidget):
    """
    全透明置顶字幕窗口。

    调用方式:
        overlay = SubtitleOverlay()
        overlay.show()
        overlay.update_subtitle(en="Hello", cn="你好")
    """

    # ── 外观常量 ──────────────────────────────────────────────
    BG_COLOR       = QColor(15, 15, 20, 200)    # 深色半透明背景
    EN_COLOR       = "#94a3b8"                  # 英文：蓝灰
    CN_COLOR       = "#f1f5f9"                  # 中文：近白
    ACCENT_COLOR   = "#38bdf8"                  # 强调色（边框）
    BORDER_RADIUS  = 14
    PADDING_H      = 24
    PADDING_V      = 12

    def __init__(self, show_english: bool = True, parent=None):
        super().__init__(parent)
        self.show_english = show_english
        self._drag_pos    = QPoint()
        self._alt_held    = False

        self._setup_window()
        self._setup_ui()
        self._position_window()

    # ── 初始化 ────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool               # 不在任务栏显示
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # 点击穿透：鼠标消息直接传到下层窗口
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setMinimumWidth(500)
        self.setMaximumWidth(900)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self.PADDING_H, self.PADDING_V,
            self.PADDING_H, self.PADDING_V,
        )
        layout.setSpacing(4)

        # 英文字幕
        self._en_label = QLabel("")
        self._en_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._en_label.setWordWrap(True)
        self._en_label.setStyleSheet(f"""
            QLabel {{
                color: {self.EN_COLOR};
                font-size: 15px;
                font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
                font-weight: 400;
                background: transparent;
            }}
        """)

        # 中文字幕
        self._cn_label = QLabel("")
        self._cn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cn_label.setWordWrap(True)
        self._cn_label.setStyleSheet(f"""
            QLabel {{
                color: {self.CN_COLOR};
                font-size: 21px;
                font-family: 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', sans-serif;
                font-weight: 600;
                background: transparent;
            }}
        """)

        layout.addWidget(self._en_label)
        layout.addWidget(self._cn_label)
        self.setLayout(layout)

    def _position_window(self):
        """默认停靠在主屏幕底部居中。"""
        screen_geo = QApplication.primaryScreen().availableGeometry()
        self.adjustSize()
        x = screen_geo.center().x() - self.width() // 2
        y = screen_geo.bottom() - self.height() - 60
        self.move(x, y)

    # ── 公开方法 ──────────────────────────────────────────────

    def update_subtitle(self, en: str = "", cn: str = ""):
        """更新字幕。en/cn 为空字符串时对应行清空。"""
        self._en_label.setText(en)
        self._cn_label.setText(cn)
        self._en_label.setVisible(self.show_english and bool(en))
        self._cn_label.setVisible(bool(cn))
        self.adjustSize()

    def set_show_english(self, visible: bool):
        self.show_english = visible
        self._en_label.setVisible(visible and bool(self._en_label.text()))
        self.adjustSize()

    def enable_drag(self, enable: bool):
        """
        启用/禁用拖动模式。
        启用时临时移除鼠标穿透属性，允许用户拖动重定位。
        """
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not enable)

    # ── 绘制圆角半透明背景 ────────────────────────────────────

    def paintEvent(self, event):
        if not self._cn_label.text() and not self._en_label.text():
            return   # 无内容时不绘制背景

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, self.width(), self.height(),
            self.BORDER_RADIUS, self.BORDER_RADIUS,
        )

        # 背景填充
        painter.fillPath(path, self.BG_COLOR)

        # 顶部细边框（强调色）
        painter.setPen(QColor(self.ACCENT_COLOR))
        painter.drawLine(
            self.BORDER_RADIUS, 0,
            self.width() - self.BORDER_RADIUS, 0,
        )

    # ── 拖动重定位（Alt + 左键） ──────────────────────────────
    # 注意：需先调用 enable_drag(True) 才能接收鼠标事件

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
