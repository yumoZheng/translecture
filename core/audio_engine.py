"""
LectureFlow - core/audio_engine.py
音频采集引擎 v3 (时间驱动切片 · 低延迟版)

核心变更 vs v2:
  - 移除"等待静音"逻辑，改为定时强制刷新（FLUSH_INTERVAL = 4.0s）
  - 每个新切片头部包含前一切片末尾 0.5s，避免词语被截断
  - RMS_THRESHOLD 作为公开属性暴露，供未来 UI 滑块控制
  - 保留可选 WAV 录音功能
"""

import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd


# ─────────────────────────────────────────────────────────────
# 默认参数
# ─────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE  = 16000    # Hz — Whisper 要求
DEFAULT_BLOCK_SIZE   = 1024     # 每次 callback 帧数
DEFAULT_FLUSH_SEC    = 4.0      # 强制推送间隔（秒）
DEFAULT_OVERLAP_SEC  = 0.5      # 上下文重叠（秒）
DEFAULT_RMS_THRESH   = 0.01     # 初始静音阈值（可运行时调整）


class AudioEngine:
    """
    基于定时切片的麦克风采集引擎。

    每隔 FLUSH_INTERVAL 秒，无论是否有说话停顿，
    都会将当前缓冲区推送至 audio_queue。
    每个切片头部附带上一切片末尾 overlap_samples 帧。

    公开属性:
        rms_threshold (float): 静音判定阈值，范围 0.0–1.0。
                                可在运行期间动态调整（UI 滑块）。

    用法:
        q = queue.Queue()
        engine = AudioEngine(audio_queue=q)
        engine.start()
        chunk = q.get()   # np.ndarray float32 shape (N,)
        engine.stop()
    """

    def __init__(
        self,
        audio_queue:   queue.Queue,
        device_index:  int | None = None,
        sample_rate:   int        = DEFAULT_SAMPLE_RATE,
        flush_interval: float     = DEFAULT_FLUSH_SEC,
        overlap_sec:   float      = DEFAULT_OVERLAP_SEC,
        save_audio:    bool       = False,
    ):
        self.audio_queue    = audio_queue
        self.device_index   = device_index
        self.sample_rate    = sample_rate
        self.flush_interval = flush_interval
        self.overlap_sec    = overlap_sec
        self.save_audio     = save_audio

        # ── 公开：可被 UI 滑块动态写入 ──────────────────────
        self.rms_threshold: float = DEFAULT_RMS_THRESH

        self._stream:     sd.InputStream | None = None
        self._recording   = False
        self._lock        = threading.Lock()

        # 主采集缓冲（回调线程写入，flush 线程读取）
        self._buf:        list[np.ndarray] = []
        self._buf_lock    = threading.Lock()

        # 上下文重叠缓存（上一切片的末尾）
        self._overlap_samples = int(sample_rate * overlap_sec)
        self._prev_tail:  np.ndarray | None = None

        # 定时 flush 线程
        self._flush_thread: threading.Thread | None = None
        self._stop_flush    = threading.Event()

        # WAV 录音
        self._wav_file: wave.Wave_write | None = None
        self._recordings_dir = Path("recordings")

    # ── 公开接口 ──────────────────────────────────────────────

    def start(self):
        """开始录音和定时推送。"""
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._buf       = []
            self._prev_tail = None
            self._stop_flush.clear()

            if self.save_audio:
                self._open_wav_file()

            self._stream = sd.InputStream(
                samplerate  = self.sample_rate,
                channels    = 1,
                dtype       = "float32",
                device      = self.device_index,
                blocksize   = DEFAULT_BLOCK_SIZE,
                callback    = self._audio_callback,
            )
            self._stream.start()

            self._flush_thread = threading.Thread(
                target  = self._flush_loop,
                daemon  = True,
                name    = "AudioFlushTimer",
            )
            self._flush_thread.start()
            print(f"[AudioEngine] ▶ 开始录音 (flush={self.flush_interval}s, overlap={self.overlap_sec}s)")

    def stop(self):
        """停止录音，将剩余缓冲推送到队列。"""
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._stop_flush.set()

            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # 推送最后一帧残余
            self._flush(label="[最终帧]")

            if self._wav_file:
                self._wav_file.close()
                self._wav_file = None

            print("[AudioEngine] ⏹ 录音已停止")

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ── 内部：sounddevice 回调（独立音频线程） ─────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"[AudioEngine] ⚠️  {status}")

        mono = indata[:, 0].copy()   # (frames,) float32

        # 实时写 WAV
        if self._wav_file:
            pcm16 = (mono * 32767).astype(np.int16)
            self._wav_file.writeframes(pcm16.tobytes())

        with self._buf_lock:
            self._buf.append(mono)

    # ── 内部：定时 flush 线程 ─────────────────────────────────

    def _flush_loop(self):
        """每隔 flush_interval 秒触发一次推送。"""
        while not self._stop_flush.wait(timeout=self.flush_interval):
            self._flush()

    def _flush(self, label: str = ""):
        """
        将当前缓冲拼合（头部接入 overlap），推入队列，
        然后保留末尾 overlap_samples 帧作为下一切片前缀。
        """
        with self._buf_lock:
            if not self._buf:
                return
            raw = np.concatenate(self._buf)
            self._buf = []

        # 检查是否有实质内容（RMS 阈值）
        rms = float(np.sqrt(np.mean(raw ** 2)))
        if rms < self.rms_threshold:
            # 全静音段：更新 overlap 但不推送
            if len(raw) >= self._overlap_samples:
                self._prev_tail = raw[-self._overlap_samples:]
            return

        # 拼接上下文重叠
        if self._prev_tail is not None:
            chunk = np.concatenate([self._prev_tail, raw])
        else:
            chunk = raw

        # 保存本次末尾供下次使用
        if len(raw) >= self._overlap_samples:
            self._prev_tail = raw[-self._overlap_samples:]
        else:
            self._prev_tail = raw.copy()

        duration = len(chunk) / self.sample_rate
        tag = label or f"[定时 {self.flush_interval}s]"
        print(f"[AudioEngine] 🎙 推送音频块 {tag}: {duration:.2f}s (含 overlap {self.overlap_sec}s)")
        self.audio_queue.put(chunk)

    # ── WAV 文件 ──────────────────────────────────────────────

    def _open_wav_file(self):
        self._recordings_dir.mkdir(exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        filepath = self._recordings_dir / filename
        self._wav_file = wave.open(str(filepath), "wb")
        self._wav_file.setnchannels(1)
        self._wav_file.setsampwidth(2)
        self._wav_file.setframerate(self.sample_rate)
        print(f"[AudioEngine] 💾 WAV 录音: {filepath}")
