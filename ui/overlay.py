"""
LectureFlow - ui/overlay.py
悬浮窗界面：在屏幕一角显示实时翻译文字，
支持透明背景、置顶显示、可拖动。
"""

# TODO Phase 2: 实现 PyQt6 悬浮窗
# from PyQt6.QtWidgets import QWidget, QLabel
# from PyQt6.QtCore import Qt


class OverlayWindow:
    """
    屏幕悬浮字幕窗口。
    Phase 1: 仅包含接口定义（占位）。
    """

    def __init__(self, config: dict):
        self.opacity = config["ui"]["opacity"]
        self.font_size = config["ui"]["font_size"]
        self.position = config["ui"]["position"]

    def show(self):
        """显示悬浮窗。"""
        print("[OverlayWindow] 悬浮窗已显示（占位模式）")

    def hide(self):
        """隐藏悬浮窗。"""
        print("[OverlayWindow] 悬浮窗已隐藏")

    def update_text(self, text: str):
        """更新显示的翻译文本。"""
        print(f"[OverlayWindow] 字幕更新: {text}")
