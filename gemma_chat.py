# Gemma 4 chat helper for the Newton viewer.
# Wraps litert_lm.Engine + Conversation in a background thread so the
# OpenGL viewer's UI thread never blocks on inference.

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "gemma-4-E2B-it.litertlm",
)


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    text: str = ""
    image_path: str | None = None


@dataclass
class GemmaChat:
    model_path: str = DEFAULT_MODEL_PATH
    cache_dir: str = field(default_factory=lambda: os.path.expanduser("~/.cache/litert-lm"))
    use_gpu: bool = False

    def __post_init__(self):
        self.messages: list[ChatMessage] = []
        self._engine = None
        self._conversation = None
        self._lock = threading.Lock()
        self._inflight = False
        self._error: str | None = None
        self._status = "idle"  # "idle" | "loading" | "ready" | "thinking" | "error"
        # Thread -> UI updates (chunks of streamed text)
        self._chunk_q: queue.Queue = queue.Queue()
        # Track the assistant message currently being streamed.
        self._streaming_msg: ChatMessage | None = None
        self._worker: threading.Thread | None = None

    # ----- model lifecycle -----
    def load(self):
        """Kick off model loading on a background thread."""
        if self._engine is not None or self._status == "loading":
            return
        if not os.path.exists(self.model_path):
            self._status = "error"
            self._error = f"Model file not found: {self.model_path}"
            return
        self._status = "loading"
        self._error = None
        threading.Thread(target=self._load_engine, daemon=True).start()

    def _load_engine(self):
        try:
            import litert_lm  # noqa: PLC0415

            os.makedirs(self.cache_dir, exist_ok=True)
            backend = litert_lm.Backend.GPU if self.use_gpu else litert_lm.Backend.CPU
            vision = litert_lm.Backend.GPU if self.use_gpu else litert_lm.Backend.CPU

            self._engine = litert_lm.Engine(
                self.model_path,
                backend=backend,
                vision_backend=vision,
                cache_dir=self.cache_dir,
            )
            self._conversation = self._engine.create_conversation(
                system_message=(
                    "You are a vision-capable assistant embedded in a Newton "
                    "physics simulation viewer. Be concise."
                ),
            )
            with self._lock:
                self._status = "ready"
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status = "error"
                self._error = f"{type(exc).__name__}: {exc}"

    def is_ready(self) -> bool:
        return self._status == "ready" and self._conversation is not None

    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def busy(self) -> bool:
        return self._inflight or self._status == "loading"

    # ----- chat -----
    def send(self, text: str, image_path: str | None = None):
        """Queue a user message; return immediately. Inference runs on a worker thread."""
        if not self.is_ready() or self._inflight:
            return
        text = (text or "").strip()
        if not text and not image_path:
            return

        user_msg = ChatMessage(role="user", text=text, image_path=image_path)
        assistant_msg = ChatMessage(role="assistant", text="")
        with self._lock:
            self.messages.append(user_msg)
            self.messages.append(assistant_msg)
            self._streaming_msg = assistant_msg
            self._inflight = True
            self._status = "thinking"

        self._worker = threading.Thread(
            target=self._run_inference, args=(user_msg,), daemon=True
        )
        self._worker.start()

    def _build_payload(self, msg: ChatMessage):
        content: list[dict] = []
        if msg.image_path:
            content.append({"type": "image", "path": msg.image_path})
        if msg.text:
            content.append({"type": "text", "text": msg.text})
        return {"role": "user", "content": content}

    def _run_inference(self, user_msg: ChatMessage):
        try:
            payload = self._build_payload(user_msg)
            for chunk in self._conversation.send_message_async(payload):
                for item in chunk.get("content", []):
                    if item.get("type") == "text":
                        self._chunk_q.put(item.get("text", ""))
            self._chunk_q.put(None)  # sentinel: done
        except Exception as exc:  # noqa: BLE001
            self._chunk_q.put(("__error__", f"{type(exc).__name__}: {exc}"))

    def pump(self):
        """Drain streamed chunks into the current assistant message. Call from the UI thread."""
        if self._streaming_msg is None and self._chunk_q.empty():
            return
        try:
            while True:
                item = self._chunk_q.get_nowait()
                if item is None:
                    with self._lock:
                        self._streaming_msg = None
                        self._inflight = False
                        self._status = "ready"
                elif isinstance(item, tuple) and item and item[0] == "__error__":
                    with self._lock:
                        if self._streaming_msg is not None:
                            self._streaming_msg.text += f"\n[error] {item[1]}"
                        self._streaming_msg = None
                        self._inflight = False
                        self._status = "error"
                        self._error = item[1]
                else:
                    with self._lock:
                        if self._streaming_msg is not None:
                            self._streaming_msg.text += item
        except queue.Empty:
            pass

    def reset(self):
        """Clear chat history and start a fresh conversation."""
        with self._lock:
            self.messages.clear()
            self._streaming_msg = None
        if self._engine is not None:
            try:
                self._conversation = self._engine.create_conversation()
            except Exception as exc:  # noqa: BLE001
                self._error = f"reset failed: {exc}"

    def close(self):
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:  # noqa: BLE001
                pass
            self._engine = None
            self._conversation = None
            self._status = "idle"
