"""
LectureFlow - core/translator.py
ASR + LLM 翻译引擎 v4 (语境滑动窗口版)

新增能力:
  - context_buffer         : 滑动记忆窗口，缓存前 N 句英文原文
  - llm_translate()        : 自动将前文语境注入 Prompt，改善跨切片连贯性
  - switch_whisper_model() : 内存安全切换 Whisper 模型
  - switch_llm_model()     : Ollama 模型热更新（下次请求生效）
"""

import gc
import requests
import numpy as np
from faster_whisper import WhisperModel


# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_COMPUTE_TYPE  = "float32"
DEFAULT_OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL  = "gemma3:1b"

SYSTEM_PROMPT = (
    "Role: Professional Interpreter.\n"
    "Task: Translate ONLY the [CURRENT] text to Simplified Chinese.\n"
    "Rules:\n"
    "1. NO English in output.\n"
    "2. NO repetition of [PREVIOUS] text.\n"
    "3. NO self-introduction or chatting.\n"
    "4. If [CURRENT] is a fragment, use [PREVIOUS] to ensure the translation makes sense.\n"
    "5. OUTPUT ONLY CHINESE."
)

# 语境窗口大小：保留最近 N 句英文，作为翻译时的前文参考
CONTEXT_WINDOW_SIZE = 3


class Translator:
    """
    语音识别 + 翻译一体化处理器，支持运行时动态切换模型。

    核心公开方法:
        transcribe(audio)                    -> str        (Whisper ASR)
        llm_translate(en, subject)           -> str        (Ollama 翻译)
        translate(audio, subject)            -> dict       (端到端)
        switch_whisper_model(model_name)     -> bool       (内存安全切换)
        switch_llm_model(model_name)         -> bool       (热更新)

    公开属性:
        current_whisper_model (str)   当前 Whisper 模型 ID
        ollama_model          (str)   当前 Ollama 模型 ID
    """

    def __init__(self, config: dict):
        # ── Ollama 配置 ──────────────────────────────────────
        ollama_cfg        = config.get("ollama", {})
        self.ollama_url   = (
            ollama_cfg.get("base_url", "http://localhost:11434")
            + ollama_cfg.get("api_endpoint", "/api/generate")
        )
        self.ollama_model = ollama_cfg.get("current_model", DEFAULT_OLLAMA_MODEL)

        # ── Whisper 配置 ─────────────────────────────────────
        whisper_cfg                = config.get("whisper", {})
        self._compute_type         = whisper_cfg.get("compute_type", DEFAULT_COMPUTE_TYPE)
        self.current_whisper_model = whisper_cfg.get("current_model", DEFAULT_WHISPER_MODEL)

        # ── 语境滑动窗口 ─────────────────────────────────────
        # 存储最近 CONTEXT_WINDOW_SIZE 句英文原文，用于辅助 LLM 理解跨切片语义
        self.context_buffer: list[str] = []

        # ── 加载 Whisper ─────────────────────────────────────
        self._whisper: WhisperModel | None = None
        self._load_whisper(self.current_whisper_model)

    # ═══════════════════════════════════════════════════════════
    # 公开 API — 推理
    # ═══════════════════════════════════════════════════════════

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Whisper ASR：将音频转录为英文字符串。
        在 consumer_loop 中先行调用，可立即打印英文。

        audio_data: float32 ndarray, shape (N,), 16kHz
        """
        if self._whisper is None:
            print("[Translator] ⚠️  Whisper 模型未就绪，跳过转录。")
            return ""

        segments, _ = self._whisper.transcribe(
            audio_data,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def llm_translate(self, english_text: str, subject_context: str = "") -> str:
        # 构造极简 Prompt
        prompt_elements = []
        if subject_context:
            prompt_elements.append(f"Subject: {subject_context}")
        
        if self.context_buffer:
            # 使用简写，减少干扰
            prompt_elements.append(f"PREV: {' '.join(self.context_buffer)}")
            
        prompt_elements.append(f"CURRENT: {english_text}")
        
        full_prompt = "\n".join(prompt_elements)

        payload = {
            "model":  self.ollama_model,
            "system": SYSTEM_PROMPT,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p":       0.9,
            },
        }

        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()

            # ── 翻译成功：更新滑动语境窗口 ──────────────────
            if result and result != "[翻译失败]":
                self.context_buffer.append(english_text)
                if len(self.context_buffer) > CONTEXT_WINDOW_SIZE:
                    self.context_buffer.pop(0)   # 弹出最早的一句

            return result

        except requests.exceptions.RequestException as e:
            print(f"[Translator] ❌ Ollama 请求失败: {e}")
            return "[翻译失败]"

    def translate(
        self,
        audio_data: np.ndarray,
        subject_context: str = "",
    ) -> dict[str, str]:
        """
        端到端：音频 → ASR → 翻译 → {"en": ..., "cn": ...}。
        若需要"先见英文"体验，请在外部分别调用
        transcribe() 与 llm_translate()。
        """
        en = self.transcribe(audio_data)
        if not en:
            return {"en": "", "cn": ""}
        cn = self.llm_translate(en, subject_context)
        return {"en": en, "cn": cn}

    # ═══════════════════════════════════════════════════════════
    # 公开 API — 动态切换
    # ═══════════════════════════════════════════════════════════

    def switch_whisper_model(self, model_name: str) -> bool:
        """
        内存安全地切换 Whisper 模型。

        流程：
          1. 打印提示
          2. del self._whisper + gc.collect() 释放内存
          3. 加载新模型
          4. 更新 self.current_whisper_model

        返回 True 表示成功，False 表示失败（异常已捕获）。
        """
        if model_name == self.current_whisper_model:
            print(f"[Translator] ℹ️  Whisper 已是 {model_name}，无需切换。")
            return True

        print(f"[Translator] 🔄 Whisper 切换: {self.current_whisper_model} → {model_name}")
        print("[Translator]    正在释放旧模型...")

        # ── 释放旧模型内存 ───────────────────────────────────
        if self._whisper is not None:
            del self._whisper
            self._whisper = None
            gc.collect()
            print("[Translator]    ✅ 旧模型已释放")

        # ── 加载新模型 ───────────────────────────────────────
        return self._load_whisper(model_name)

    def switch_llm_model(self, model_name: str) -> bool:
        """
        热更新 Ollama 模型（仅修改字符串，下次请求即生效）。

        返回 True 表示成功，False 表示失败（异常已捕获）。
        """
        if model_name == self.ollama_model:
            print(f"[Translator] ℹ️  LLM 已是 {model_name}，无需切换。")
            return True

        try:
            old = self.ollama_model
            self.ollama_model = model_name
            print(f"[Translator] ✅ LLM 已更新: {old} → {model_name}（下次翻译生效）")
            return True
        except Exception as e:
            print(f"[Translator] ❌ LLM 切换失败: {e}")
            return False

    def ensure_llm_model_exists(self, model_name: str) -> bool:
        """
        通过 GET /api/tags 检查 model_name 是否已安装。

        返回:
            True  — 模型已存在，可直接使用
            False — 模型不存在，或 Ollama 未启动（异常已捕获）

        UI 对接: 可在后台线程调用，结果驱动"下载"按钮的显示/隐藏。
        """
        tags_url = self.ollama_url.replace("/api/generate", "/api/tags")
        try:
            resp = requests.get(tags_url, timeout=5)
            resp.raise_for_status()
            installed = [m["name"] for m in resp.json().get("models", [])]
            # Ollama tag 格式: "gemma3:1b" 或 "gemma3:1b-..."
            exists = any(
                m == model_name or m.startswith(model_name.split(":")[0] + ":")
                and model_name in m
                for m in installed
            )
            if exists:
                print(f"[Translator] ✅ 模型 {model_name} 已安装")
            else:
                print(f"[Translator] ⚠️  模型 {model_name} 未找到")
                print(f"[Translator]    已安装模型: {installed or '(无)'}")
            return exists
        except requests.exceptions.ConnectionError:
            print(f"[Translator] ❌ 无法连接 Ollama，请确认 `ollama serve` 正在运行。")
            return False
        except requests.exceptions.RequestException as e:
            print(f"[Translator] ❌ 检查模型时出错: {e}")
            return False

    def download_llm_model(
        self,
        model_name: str,
        progress_callback=None,
    ) -> bool:
        """
        通过 POST /api/pull 流式下载 Ollama 模型，
        并在控制台渲染实时进度条。

        参数:
            model_name        : 要下载的模型 ID，如 "gemma3:1b"
            progress_callback : 可选回调 fn(pct: float, speed_mb: str)
                                供 Phase 3 UI 进度条绑定使用

        返回:
            True  — 下载并加载完成
            False — 下载失败或 Ollama 未启动

        进度条格式: [████████░░░░░░░] 62%  ↓ 3.4 MB/s
        """
        import json as _json
        import time  as _time

        pull_url = self.ollama_url.replace("/api/generate", "/api/pull")
        print(f"[Translator] ⬇️  开始下载模型: {model_name}")
        print(f"[Translator]    API: {pull_url}\n")

        BAR_WIDTH  = 30
        last_bytes = 0
        last_time  = _time.time()

        def _render(pct: float, speed_str: str):
            filled = int(BAR_WIDTH * pct / 100)
            bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
            line   = f"\r  [{bar}] {pct:5.1f}%  ↓ {speed_str}"
            print(line, end="", flush=True)
            if progress_callback:
                progress_callback(pct, speed_str)

        try:
            with requests.post(
                pull_url,
                json={"model": model_name, "stream": True},
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()

                completed  = 0
                total      = 0
                status_msg = ""

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = _json.loads(raw_line)
                    except _json.JSONDecodeError:
                        continue

                    status_msg = data.get("status", status_msg)
                    completed  = data.get("completed", completed)
                    total      = data.get("total",     total)

                    # ── 计算百分比和速度 ──────────────────────
                    if total and total > 0:
                        pct       = completed / total * 100
                        now       = _time.time()
                        dt        = now - last_time
                        db        = completed - last_bytes

                        if dt > 0.3:          # 每 0.3s 刷新一次速度
                            speed_bps  = db / dt
                            speed_mb   = f"{speed_bps / 1_048_576:.1f} MB/s"
                            last_bytes = completed
                            last_time  = now
                        else:
                            speed_mb   = "..."

                        _render(pct, speed_mb)

                    elif "success" in status_msg.lower() or status_msg == "":
                        pass   # 最后一帧，无 total 字段

                # 下载完成，换行
                print(f"\r  [{'█' * BAR_WIDTH}] 100.0%  ✅ 下载完成！        ")
                print(f"\n[Translator] ✅ 模型 {model_name} 安装成功")
                return True

        except requests.exceptions.ConnectionError:
            print(f"\n[Translator] ❌ Ollama 连接失败，下载中断。")
            return False
        except requests.exceptions.Timeout:
            print(f"\n[Translator] ❌ 下载超时（300s）。")
            return False
        except requests.exceptions.RequestException as e:
            print(f"\n[Translator] ❌ 下载失败: {e}")
            return False

    # ═══════════════════════════════════════════════════════════
    # 内部辅助
    # ═══════════════════════════════════════════════════════════


    def _load_whisper(self, model_name: str) -> bool:
        """加载指定 Whisper 模型，成功返回 True，失败返回 False。"""
        print(f"[Translator]    正在加载 Whisper [{model_name}] ({self._compute_type})...")
        try:
            self._whisper = WhisperModel(
                model_name,
                device="cpu",
                compute_type=self._compute_type,
            )
            self.current_whisper_model = model_name
            print(f"[Translator] ✅ Whisper [{model_name}] 加载完成")
            return True
        except Exception as e:
            print(f"[Translator] ❌ Whisper [{model_name}] 加载失败: {e}")
            self._whisper = None
            return False
