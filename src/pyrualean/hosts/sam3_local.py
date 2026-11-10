# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""SAM3 in this process, behind the interface the RoboDojo tools segment through.

RPent runs SAM3 as a server of its own and hands its tools a client.  A host without RPent's
runtime (:mod:`.xpolicylab_robodojo`) loads the same model here: the official ``sam3``
package and its interactive point head, with the calls and settings RPent's server makes
(every candidate kept, bfloat16 autocast, the best-scoring instance returned), so one picture
and one prompt give one mask whichever of the two runs it.  ``segment`` takes exactly one of
a text prompt and a ``[row, col]`` pixel and answers like
``rpent.robots.components.sam3_client.Sam3Client``.
"""

from __future__ import annotations

import hashlib
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Segmented:
    """One mask, or none: what RPent's client returns."""

    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask: np.ndarray | None = None
    reason: str | None = None


def _array(value: Any) -> np.ndarray:
    return value if isinstance(value, np.ndarray) else value.detach().float().cpu().numpy()


def select_top(masks: Any, scores: Any, boxes: Any, min_score: float) -> Segmented:
    """The best-scoring candidate, if it reaches ``min_score`` and its mask is not empty."""

    if masks is None or scores is None:
        return Segmented(False, reason="SAM3 returned no candidate")
    masks, scores = _array(masks), _array(scores).reshape(-1)
    if masks.size == 0 or scores.size == 0:
        return Segmented(False, reason="SAM3 returned no candidate")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[0] != scores.shape[0]:
        raise RuntimeError(
            f"unexpected SAM3 candidate shapes: masks={masks.shape}, scores={scores.shape}"
        )
    index = int(np.argmax(scores))
    score = float(scores[index])
    box: list[float] | None = None
    if boxes is not None:
        boxes = _array(boxes)
        if boxes.ndim >= 2 and index < boxes.shape[0]:
            box = [float(value) for value in boxes[index].reshape(-1)[:4]]
    if score < min_score:
        return Segmented(
            False,
            score=score,
            box=box,
            reason=f"top score {score:.3f} is below min_score {min_score:.3f}",
        )
    mask = np.asarray(masks[index]) > 0
    if mask.ndim != 2 or not mask.any():
        return Segmented(False, score=score, box=box, reason="SAM3 returned an empty mask")
    return Segmented(True, score=score, box=box, mask=mask)


class LocalSam3:
    """SAM3 loaded on one GPU of this process."""

    def __init__(self, checkpoint: str | None = None, cuda_device: int | None = None) -> None:
        checkpoint = checkpoint or os.environ.get("SAM3_CHECKPOINT_PATH")
        if not checkpoint:
            raise RuntimeError("SAM3_CHECKPOINT_PATH is not set")
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {resolved}")
        if cuda_device is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            # the model takes device 0 of what is visible; nothing has touched CUDA yet
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "SAM3 in this process needs torch and the sam3 package "
                "(https://github.com/facebookresearch/sam3)"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 in this process needs a CUDA-capable GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = build_sam3_image_model(
                device="cuda",
                checkpoint_path=str(resolved),
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
        self._torch = torch
        self._model = model
        # every candidate is kept: the caller's min_score decides
        self._processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        self._lock = threading.Lock()
        self._digest: str | None = None
        self._state: dict[str, Any] | None = None

    def _autocast(self) -> Any:
        return self._torch.autocast("cuda", dtype=self._torch.bfloat16)

    def _state_for(self, pixels: np.ndarray) -> dict[str, Any]:
        """The image features, computed once for as many prompts as are put to one picture."""

        from PIL import Image

        digest = hashlib.sha256(repr(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if digest != self._digest or self._state is None:
            with self._autocast():
                self._state = self._processor.set_image(Image.fromarray(pixels, mode="RGB"))
            self._digest = digest
        return self._state

    def segment(
        self,
        image: Any,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Segmented:
        has_text = isinstance(text_prompt, str) and bool(text_prompt.strip())
        if has_text == (point is not None):
            raise ValueError("provide exactly one of text_prompt or point")
        if point is not None and (not isinstance(point, (list, tuple)) or len(point) != 2):
            raise ValueError("point must be [row, col]")
        try:
            min_score = float(min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_score must be a number, got {min_score!r}") from exc
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        pixels = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
        with self._lock, warnings.catch_warnings():
            # the model runs inside the program's own process here: what it warns about (a
            # skipped post-processing kernel on a new GPU) is not the program's to read
            warnings.simplefilter("ignore")
            state = self._state_for(pixels)
            if has_text:
                assert text_prompt is not None
                with self._autocast():
                    out = self._processor.set_text_prompt(prompt=text_prompt.strip(), state=state)
                return select_top(out.get("masks"), out.get("scores"), out.get("boxes"), min_score)
            assert point is not None
            row, col = int(point[0]), int(point[1])
            height, width = int(state["original_height"]), int(state["original_width"])
            if row < 0 or col < 0 or row >= height or col >= width:
                raise ValueError(
                    f"point [row, col] {list(point)} is outside image shape [{height}, {width}]"
                )
            if getattr(self._model, "inst_interactive_predictor", None) is None:
                raise RuntimeError("SAM3 instance interactivity is not enabled")
            with self._autocast():
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=np.asarray([[col, row]], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int64),
                    multimask_output=True,
                )
            return select_top(masks, scores, None, min_score)


class RemoteSam3:
    """SAM3 over RPent's HTTP RPC service (``rpent.robots.components.sam3_server``).

    Same ``segment`` contract as :class:`LocalSam3`, so a host can serve the
    model from a separate process: the 3.3 GB checkpoint is loaded once and
    the policy's interpreter does not need torch or the sam3 package.
    """

    def __init__(self, endpoint: str) -> None:
        self._base = endpoint.rstrip("/")
        if not self._base.startswith(("http://", "https://")):
            self._base = f"http://{self._base}"
        self._lock = threading.Lock()

    def segment(
        self,
        image: Any,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Segmented:
        import base64
        import io
        import json
        import urllib.request

        from PIL import Image

        has_text = isinstance(text_prompt, str) and bool(text_prompt.strip())
        if has_text == (point is not None):
            raise ValueError("provide exactly one of text_prompt or point")
        if point is not None and (not isinstance(point, (list, tuple)) or len(point) != 2):
            raise ValueError("point must be [row, col]")
        min_score = float(min_score)
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        pixels = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
        buffer = io.BytesIO()
        Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
        kwargs: dict[str, Any] = {
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "min_score": min_score,
        }
        if has_text:
            kwargs["text_prompt"] = str(text_prompt).strip()
        else:
            assert point is not None
            kwargs["point"] = [int(point[0]), int(point[1])]
        body = json.dumps(
            {"method": "sam3.segment", "args": [], "kwargs": kwargs, "session_id": None}
        ).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - operator-supplied endpoint
            self._base + "/call",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._lock:
            with urllib.request.urlopen(request, timeout=180) as reply:  # noqa: S310
                payload = json.loads(reply.read().decode("utf-8"))
        if not isinstance(payload, dict) or not payload.get("ok"):
            error = payload.get("error") if isinstance(payload, dict) else payload
            raise RuntimeError(f"SAM3 service refused the request: {error}")
        result = payload.get("result") or {}
        if not isinstance(result, dict) or not isinstance(result.get("found"), bool):
            raise RuntimeError(f"invalid SAM3 segment response: {result!r}")
        score = result.get("score")
        box = result.get("box")
        if not result["found"]:
            return Segmented(
                False,
                score=None if score is None else float(score),
                box=box,
                reason=str(result.get("reason") or "SAM3 found no mask"),
            )
        encoded = result.get("mask_png_base64")
        shape = result.get("mask_shape")
        if not isinstance(encoded, str) or not encoded:
            raise RuntimeError("SAM3 response marked found but omitted mask_png_base64")
        mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")) > 0
        if not mask.any():
            return Segmented(False, score=score, box=box, reason="SAM3 returned an empty mask")
        if isinstance(shape, list) and len(shape) == 2:
            mask = mask.reshape(int(shape[0]), int(shape[1]))
        return Segmented(
            True, score=None if score is None else float(score), box=box, mask=mask
        )


_SHARED: dict[tuple[str, str], Any] = {}
_SHARED_LOCK = threading.Lock()


def shared(checkpoint: str | None = None, cuda_device: int | None = None) -> Any:
    """The process's one SAM3 client: remote when ``SAM3_ENDPOINT`` is set, else local.

    A policy server plays one episode after another: the model is loaded for the first and
    serves the rest (a second copy would cost its load time and the card its memory).
    """

    endpoint = os.environ.get("SAM3_ENDPOINT") or ""
    if endpoint:
        key = ("remote", endpoint)
        with _SHARED_LOCK:
            if key not in _SHARED:
                _SHARED[key] = RemoteSam3(endpoint)
            return _SHARED[key]
    path = checkpoint or os.environ.get("SAM3_CHECKPOINT_PATH") or ""
    key = (path, os.environ.get("CUDA_VISIBLE_DEVICES") or str(cuda_device))
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = LocalSam3(checkpoint, cuda_device)
        return _SHARED[key]


__all__ = ["LocalSam3", "RemoteSam3", "Segmented", "select_top", "shared"]
