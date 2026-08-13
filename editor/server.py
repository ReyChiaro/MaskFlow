from __future__ import annotations

import argparse
import base64
import gc
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = ROOT / "outputs" / "editor"


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    stage: str = "等待处理"
    phase: str = "loading"
    progress: int = 0
    error: str | None = None
    output_path: Path | None = None
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "phase": self.phase,
            "progress": self.progress,
            "error": self.error,
        }
        if self.output_path:
            data["result_url"] = f"/api/jobs/{self.id}/result"
        return data


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
MODEL_CACHE: dict[str, Any] = {"key": None, "pipeline": None, "label": None}


def _set_job(
    job: Job,
    *,
    status: str | None = None,
    stage: str | None = None,
    phase: str | None = None,
    progress: int | None = None,
):
    with JOBS_LOCK:
        if status is not None:
            job.status = status
        if stage is not None:
            job.stage = stage
        if phase is not None:
            job.phase = phase
        if progress is not None:
            job.progress = progress


def _decode_image(data_url: str, destination: Path) -> None:
    _, encoded = data_url.split(",", 1)
    destination.write_bytes(base64.b64decode(encoded))


def _model_settings(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "pretrained_model": raw.get("pretrained_model") or "Qwen/Qwen-Image-Edit-2511",
        "device": raw.get("device") or "cuda",
        "dtype": raw.get("dtype") or "bfloat16",
        "sft_path": raw.get("sft_path") or "ReyChiaro/MaskFlow",
        "sft_weight_name": raw.get("sft_weight_name") or "maskflow-S.safetensors",
        "dmd_path": raw.get("dmd_path") or None,
        "dmd_weight_name": raw.get("dmd_weight_name") or None,
        "mask_dilation_kernel": int(raw.get("mask_dilation_kernel", 25)),
        "mask_blur_kernel": int(raw.get("mask_blur_kernel", 25)),
        "mask_blur_sigma": float(raw.get("mask_blur_sigma", 25.0)),
        "mask_edge_width": int(raw.get("mask_edge_width", 50)),
        "enable_local_denoise_infer": bool(raw.get("enable_local_denoise_infer", False)),
        "enable_pixel_blend": bool(raw.get("enable_pixel_blend", True)),
        "enable_poisson_infer": bool(raw.get("enable_poisson_infer", True)),
        "poisson_num_iter": int(raw.get("poisson_num_iter", 50)),
    }


def _build_config(settings: dict[str, Any]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="inference")

    cfg.runtime.device = settings["device"]
    cfg.runtime.dtype = settings["dtype"]
    cfg.checkpoint.sft_path = settings["sft_path"]
    cfg.checkpoint.sft_weight_name = settings["sft_weight_name"]
    cfg.checkpoint.dmd_path = settings["dmd_path"]
    cfg.checkpoint.dmd_weight_name = settings["dmd_weight_name"]
    cfg.pipeline.pretrained_model = settings["pretrained_model"]
    for name in (
        "mask_dilation_kernel",
        "mask_blur_kernel",
        "mask_blur_sigma",
        "mask_edge_width",
        "enable_local_denoise_infer",
        "enable_pixel_blend",
        "enable_poisson_infer",
        "poisson_num_iter",
    ):
        cfg.pipeline[name] = settings[name]
    return cfg


def _load_pipeline(settings: dict[str, Any], job: Job):
    import torch

    from inference import build_pipeline

    if settings["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA GPU。你仍可制作和保存 mask，推理请在 GPU 环境中启动。")

    cache_key = json.dumps(settings, sort_keys=True, ensure_ascii=False)
    if MODEL_CACHE["key"] == cache_key:
        _set_job(job, stage="复用已加载模型")
        return MODEL_CACHE["pipeline"]

    _set_job(job, stage="加载基础模型与 LoRA")
    cfg = _build_config(settings)
    device = torch.device(cfg.runtime.device)
    dtype = getattr(torch, cfg.runtime.dtype)

    if MODEL_CACHE["pipeline"] is not None:
        MODEL_CACHE.update(key=None, pipeline=None, label=None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pipeline = build_pipeline(cfg, device, dtype)
    MODEL_CACHE.update(
        key=cache_key,
        pipeline=pipeline,
        label=f"{settings['pretrained_model']} · {settings['dtype']}",
    )
    _set_job(job, stage="模型已就绪")
    return pipeline


def _run_preload(job: Job, payload: dict[str, Any]) -> None:
    try:
        _set_job(job, status="running", stage="准备模型", phase="loading", progress=0)
        with MODEL_LOCK:
            _load_pipeline(_model_settings(payload.get("settings", {})), job)
        _set_job(job, status="done", stage="模型已加载", phase="done", progress=100)
    except Exception as exc:
        job.error = str(exc)
        _set_job(job, status="error", stage="加载失败", phase="error")


def _run_inference(job: Job, payload: dict[str, Any]) -> None:
    try:
        import torch
        import torchvision.transforms.functional as TF

        from inference import load_image

        settings = _model_settings(payload.get("settings", {}))
        task_dir = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{job.id[:6]}"
        task_dir.mkdir(parents=True, exist_ok=True)
        source_path = task_dir / "source.png"
        mask_path = task_dir / "mask.png"
        output_path = task_dir / "result.png"
        _decode_image(payload["source"], source_path)
        _decode_image(payload["mask"], mask_path)

        _set_job(job, status="running", stage="准备输入", phase="loading", progress=0)
        with MODEL_LOCK:
            pipeline = _load_pipeline(settings, job)
            runtime = payload.get("runtime", {})
            seed = int(runtime.get("seed", 42))
            pipeline.generator.manual_seed(seed)

            source = load_image(str(source_path))
            mask = load_image(str(mask_path))
            batch = {
                "prompt": [payload["prompt"].strip()],
                "negative_prompt": [payload.get("negative_prompt", "").strip()],
                "target": source,
                "conditions": {"source": source, "mask": mask},
            }

            total_steps = int(runtime.get("num_inference_steps", 50))
            _set_job(
                job,
                stage=f"MaskFlow 正在生成 · 0/{total_steps}",
                phase="generating",
                progress=0,
            )
            result = pipeline.eval_step(
                batch=batch,
                num_inference_steps=total_steps,
                text_cfg_scale=float(runtime.get("text_cfg_scale", 4.0)),
                mask_cfg_scale=float(runtime.get("mask_cfg_scale", 1.0)),
                progress_callback=lambda step, total: _set_job(
                    job,
                    stage=f"MaskFlow 正在生成 · {step}/{total}",
                    phase="generating",
                    progress=int(100 * step / total),
                ),
            )
            _set_job(job, stage="保存结果", phase="saving", progress=100)
            output = result["output"][0].float().cpu().clamp(0, 1)
            TF.to_pil_image(output).save(output_path)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        job.output_path = output_path
        _set_job(job, status="done", stage="生成完成", phase="done", progress=100)
    except Exception as exc:
        job.error = str(exc)
        _set_job(job, status="error", stage="生成失败", phase="error")


class EditorHandler(BaseHTTPRequestHandler):
    server_version = "MaskFlowEditor/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length))

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            try:
                import torch

                cuda = torch.cuda.is_available()
                gpu_name = torch.cuda.get_device_name(0) if cuda else None
            except ImportError:
                cuda, gpu_name = False, None
            self._json(
                {
                    "cuda_available": cuda,
                    "gpu_name": gpu_name,
                    "model_loaded": MODEL_CACHE["pipeline"] is not None,
                    "model_label": MODEL_CACHE["label"],
                }
            )
            return

        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            job = JOBS.get(parts[2]) if len(parts) >= 3 else None
            if not job:
                self._json({"error": "任务不存在"}, HTTPStatus.NOT_FOUND)
            elif len(parts) == 4 and parts[3] == "result" and job.output_path:
                self._serve_file(job.output_path, "image/png")
            else:
                self._json(job.public())
            return

        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        asset = assets.get(path)
        if asset:
            self._serve_file(STATIC_DIR / asset[0], asset[1])
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/jobs", "/api/model/load"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._payload()
            kind = "inference" if path == "/api/jobs" else "model"
            if kind == "inference" and not all(payload.get(name) for name in ("source", "mask", "prompt")):
                self._json({"error": "请提供原图、mask 和 prompt"}, HTTPStatus.BAD_REQUEST)
                return
            job = Job(id=uuid.uuid4().hex, kind=kind)
            with JOBS_LOCK:
                JOBS[job.id] = job
            target = _run_inference if kind == "inference" else _run_preload
            threading.Thread(target=target, args=(job, payload), daemon=True).start()
            self._json(job.public(), HTTPStatus.ACCEPTED)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="MaskFlow local image editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), EditorHandler)
    print(f"MaskFlow Editor: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
