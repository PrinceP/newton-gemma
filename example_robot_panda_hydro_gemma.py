"""Newton example + Gemma 4 multimodal chat panel.

Run (default uses panda_hydro, requires CUDA):
    python example_robot_panda_hydro_gemma.py

CPU-only fallback (no CUDA required):
    python example_robot_panda_hydro_gemma.py --example cloth_franka

The chat panel lives in a floating ImGui window inside the viewer. Type a
prompt, optionally point at an image file, and Gemma 4 (loaded from
models/gemma-4-E2B-it.litertlm via litert-lm) will reply.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import warp as wp


def _patch_pinned_for_cpu_only_host():
    """Newton's GL viewer asks Warp for pinned host memory unconditionally
    (viewer_gl.py:542). Pinned memory only exists when CUDA is present, so on
    a CPU-only host the alloc raises before the viewer even opens. Pinned
    memory is just a DMA optimization for host->GPU transfers; with no GPU to
    transfer to, plain pageable memory is fine. Downgrade pinned=True to
    pinned=False when no CUDA device is available.
    """
    cuda_available = any(d.is_cuda for d in wp.get_devices())
    if cuda_available:
        return
    _orig_empty = wp.empty

    def _empty_no_pinned(*args, **kwargs):
        if kwargs.get("pinned"):
            kwargs["pinned"] = False
        return _orig_empty(*args, **kwargs)

    wp.empty = _empty_no_pinned


_patch_pinned_for_cpu_only_host()


def _patch_wp_mesh_drops_device_kwarg():
    """Newton's raytrace render_context.py:306 still calls
    `wp.Mesh(points, indices, device=...)` but this Warp version dropped the
    `device` kwarg — the mesh inherits its device from its array arguments.
    Wrap the constructor to silently drop the kwarg.
    """
    _orig_init = wp.Mesh.__init__

    def _init_drop_device(self, *args, **kwargs):
        kwargs.pop("device", None)
        return _orig_init(self, *args, **kwargs)

    wp.Mesh.__init__ = _init_drop_device


_patch_wp_mesh_drops_device_kwarg()

import newton.examples  # noqa: E402  (must come after the patches)

from gemma_chat import DEFAULT_MODEL_PATH, ChatMessage, GemmaChat
from snapshot_camera import SnapshotCamera, screenshot_gl_viewer


_DEFAULT_AUTO_PROMPTS = {
    "cloth_franka": "What is the color of the cloth in this image? Answer in one short sentence.",
    "panda_hydro": "What objects do you see in this image? Answer in one short sentence.",
}


_BASE_EXAMPLES = {
    "panda_hydro": "newton.examples.robot.example_robot_panda_hydro",
    "cloth_franka": "newton.examples.cloth.example_cloth_franka",
}


def _resolve_base_example(name: str):
    if name not in _BASE_EXAMPLES:
        raise SystemExit(
            f"Unknown --example '{name}'. Choices: {sorted(_BASE_EXAMPLES)}"
        )
    module = importlib.import_module(_BASE_EXAMPLES[name])
    return module.Example


def _peek_example_arg(argv) -> str:
    # Parse just --example up front so we can use the matching base class.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--example", default="panda_hydro")
    known, _ = pre.parse_known_args(argv)
    return known.example


def _build_example_class(base_cls):
    class Example(base_cls):
        def __init__(self, viewer, args):
            super().__init__(viewer, args)

            self.chat = GemmaChat(
                model_path=args.gemma_model,
                use_gpu=args.gemma_gpu,
            )
            self._auto_prompt_text = (
                args.auto_prompt
                if args.auto_prompt is not None
                else _DEFAULT_AUTO_PROMPTS.get(args.example, "Describe what you see.")
            )
            self._auto_prompt_sent = args.no_auto_prompt
            # State owned by the UI callback (must live across frames).
            self._chat_input = ""
            self._image_path = ""
            self._scroll_to_bottom = False
            # Frames simulated since startup — needed so the cloth/objects
            # have settled before we take the auto snapshot.
            self._frames_since_start = 0
            self._auto_prompt_frame_delay = 30
            # Lazy: built on first snapshot request.
            self._snapshot_cam: SnapshotCamera | None = None
            self._last_snapshot_path: str | None = None

            if hasattr(self.viewer, "register_ui_callback"):
                # "free" places this in its own floating window so it doesn't
                # collide with the example's "side" panel.
                self.viewer.register_ui_callback(self._render_chat_panel, position="free")

            # Kick off model loading immediately so the auto prompt can fire
            # as soon as both Gemma is ready and the scene has settled.
            if not self._auto_prompt_sent:
                self.chat.load()

        def _ensure_snapshot_cam(self) -> SnapshotCamera:
            if self._snapshot_cam is None:
                self._snapshot_cam = SnapshotCamera(self.model)
            return self._snapshot_cam

        def _current_state(self):
            # Newton examples vary on which attribute holds the live state.
            for name in ("state_0", "state", "vis_state"):
                s = getattr(self, name, None)
                if s is not None:
                    return s
            return None

        def _camera_transform(self):
            cam = getattr(self.viewer, "camera", None)
            if cam is None:
                return None
            return SnapshotCamera.from_viewer_camera(cam)

        def take_snapshot(self) -> str | None:
            # Prefer a real screenshot of the GL viewer (what's actually on
            # screen). Fall back to a raytraced snapshot via SensorTiledCamera
            # if the GL framebuffer read fails (e.g. wrong thread, no FBO yet).
            if hasattr(self.viewer, "renderer") and getattr(self.viewer, "renderer", None) is not None:
                try:
                    self._last_snapshot_path = screenshot_gl_viewer(self.viewer)
                    return self._last_snapshot_path
                except Exception as exc:  # noqa: BLE001
                    self.chat.messages.append(
                        ChatMessage(
                            role="system",
                            text=f"GL screenshot failed ({exc}); falling back to raytrace.",
                        )
                    )

            state = self._current_state()
            if state is None:
                return None
            try:
                cam = self._ensure_snapshot_cam()
                self._last_snapshot_path = cam.save_snapshot(state, self._camera_transform())
                return self._last_snapshot_path
            except Exception as exc:  # noqa: BLE001
                self.chat.messages.append(
                    ChatMessage(role="system", text=f"Snapshot failed: {exc}")
                )
                return None

        # Drain streamed chunks once per physics step; the UI callback also
        # pumps so streaming stays smooth even if step() is throttled.
        def step(self):
            super().step()
            self.chat.pump()
            self._frames_since_start += 1
            self._maybe_send_auto_prompt()

        def _maybe_send_auto_prompt(self):
            if self._auto_prompt_sent:
                return
            if not self.chat.is_ready() or self.chat.busy:
                return
            if self._frames_since_start < self._auto_prompt_frame_delay:
                return
            snap = self.take_snapshot()
            self.chat.send(self._auto_prompt_text, image_path=snap)
            self._auto_prompt_sent = True
            self._scroll_to_bottom = True

        def _render_chat_panel(self, imgui):
            self.chat.pump()

            imgui.set_next_window_size(imgui.ImVec2(420, 520), imgui.Cond_.first_use_ever.value)
            imgui.set_next_window_pos(imgui.ImVec2(40, 80), imgui.Cond_.first_use_ever.value)
            opened, _ = imgui.begin("Gemma 4 Chat")
            if not opened:
                imgui.end()
                return

            imgui.text(f"Model: {os.path.basename(self.chat.model_path)}")
            imgui.text(f"Status: {self.chat.status}")
            if self.chat.error:
                imgui.text_wrapped(f"Error: {self.chat.error}")

            if self.chat.status in ("idle", "error"):
                if imgui.button("Load model"):
                    self.chat.load()
            elif self.chat.status == "loading":
                imgui.text("Loading Gemma weights... (first load may take a while)")
            else:
                if imgui.button("Reset chat"):
                    self.chat.reset()
                imgui.same_line()
                if imgui.button("Snapshot view"):
                    snap = self.take_snapshot()
                    if snap:
                        self._image_path = snap

            imgui.separator()

            _, self._image_path = imgui.input_text("Image path", self._image_path)
            if self._image_path:
                exists = os.path.isfile(self._image_path)
                imgui.text(("[ok] " if exists else "[missing] ") + self._image_path)
                imgui.same_line()
                if imgui.button("Clear##img"):
                    self._image_path = ""

            imgui.separator()

            avail = imgui.get_content_region_avail()
            log_height = max(120.0, avail.y - 110.0)
            imgui.begin_child(
                "##chat_log",
                imgui.ImVec2(0, log_height),
                child_flags=0,
                window_flags=imgui.WindowFlags_.horizontal_scrollbar.value,
            )
            for msg in self.chat.messages:
                self._render_message(imgui, msg)
            if self._scroll_to_bottom:
                imgui.set_scroll_here_y(1.0)
                self._scroll_to_bottom = False
            imgui.end_child()

            imgui.separator()

            flags = (
                imgui.InputTextFlags_.enter_returns_true.value
                | imgui.InputTextFlags_.ctrl_enter_for_new_line.value
            )
            submitted, self._chat_input = imgui.input_text_multiline(
                "##chat_input",
                self._chat_input,
                imgui.ImVec2(-1, 60),
                flags=flags,
            )
            send_clicked = imgui.button("Send")
            imgui.same_line()
            if self.chat.busy:
                imgui.text("Gemma is thinking...")

            if (submitted or send_clicked) and not self.chat.busy and self.chat.is_ready():
                text = self._chat_input.strip()
                img = self._image_path.strip() or None
                if img and not os.path.isfile(img):
                    self.chat.messages.append(
                        ChatMessage(role="system", text=f"Image not found: {img}")
                    )
                    img = None
                if text or img:
                    self.chat.send(text, image_path=img)
                    self._chat_input = ""
                    self._scroll_to_bottom = True

            imgui.end()

        @staticmethod
        def _render_message(imgui, msg: ChatMessage):
            if msg.role == "user":
                imgui.text_disabled("You")
            elif msg.role == "assistant":
                imgui.text_disabled("Gemma")
            else:
                imgui.text_disabled(msg.role)
            if msg.image_path:
                imgui.text_wrapped(f"[image] {msg.image_path}")
            if msg.text:
                imgui.text_wrapped(msg.text)
            imgui.spacing()

        def __del__(self):
            try:
                self.chat.close()
            except Exception:
                pass

        @staticmethod
        def create_parser():
            # Some Newton examples (e.g. cloth_franka) don't ship a
            # create_parser; fall back to the default examples parser.
            if hasattr(base_cls, "create_parser"):
                parser = base_cls.create_parser()
            else:
                parser = newton.examples.create_parser()
            parser.add_argument(
                "--example",
                default="panda_hydro",
                choices=sorted(_BASE_EXAMPLES),
                help="Which Newton example to wrap with the Gemma chat panel.",
            )
            parser.add_argument(
                "--gemma-model",
                type=str,
                default=DEFAULT_MODEL_PATH,
                help="Path to gemma-4-*.litertlm",
            )
            parser.add_argument(
                "--gemma-gpu",
                action="store_true",
                help="Use GPU backend for Gemma (text + vision). Default: CPU.",
            )
            parser.add_argument(
                "--auto-prompt",
                type=str,
                default=None,
                help="First prompt to auto-send with a scene snapshot (default: example-specific).",
            )
            parser.add_argument(
                "--no-auto-prompt",
                action="store_true",
                help="Disable the auto-prompt and just open the chat panel.",
            )
            return parser

    return Example


if __name__ == "__main__":
    base_name = _peek_example_arg(sys.argv[1:])
    base_cls = _resolve_base_example(base_name)
    Example = _build_example_class(base_cls)

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
