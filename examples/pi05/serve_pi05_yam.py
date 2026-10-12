"""Serve robocurve/pi0.5-yam behind the inspect-robots-yam /act contract.

Self-contained: run inside a stock openpi environment
(github.com/Physical-Intelligence/openpi @ 15a9616, `uv sync` + `uv pip install json_numpy`);
no openpi patches needed. The YAM transforms and the yam_pi05 inference config
are inlined below, matching what the checkpoint was trained with.

    uv run --no-sync python serve_pi05_yam.py --ckpt <dir with params/ + assets/> --port 8204

Wire contract (inspect_robots_yam.policy.ActServerPolicy): POST /act, json_numpy
body {top_cam, left_cam, right_cam: HxWx3 uint8, instruction: str,
state: (14,) float, num_steps: int} -> {"actions": (16, 14), "dt_ms": 33.3}.
Actions are ABSOLUTE joint positions, [left 6 joints + gripper, right ditto],
grippers in [0, 1], 30 fps. The per-request num_steps field (flow-matching
denoising steps) is ignored; it is fixed at load time by --num-steps.
"""

import argparse
import dataclasses
import http.server
import time

import einops
import json_numpy
import numpy as np
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.training import config as _config

DT_MS = 1000.0 / 30.0  # training data is 30 fps


# ---- policy transforms: identical to the yam_policy.py the checkpoint was trained with ----


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": _parse_image(data["images"]["top"]),
                "left_wrist_0_rgb": _parse_image(data["images"]["left"]),
                "right_wrist_0_rgb": _parse_image(data["images"]["right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Model action dim is 32; the robot uses the first 14.
        return {"actions": np.asarray(data["actions"][:, :14])}


@dataclasses.dataclass(frozen=True)
class YamInferenceConfig(_config.DataConfigFactory):
    def create(self, assets_dirs, model_config) -> _config.DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=_transforms.Group(
                inputs=[YamInputs(model_type=model_config.model_type)],
                outputs=[YamOutputs()],
            ),
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def load_policy(ckpt_dir: str, num_steps: int):
    train_config = _config.TrainConfig(
        name="yam_pi05",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=16),
        # asset_id must match the norm-stats dir shipped in the checkpoint:
        # <ckpt>/assets/yam-bimanual-merged/norm_stats.json
        data=YamInferenceConfig(assets=_config.AssetsConfig(asset_id="yam-bimanual-merged")),
    )
    return policy_config.create_trained_policy(
        train_config, ckpt_dir, sample_kwargs={"num_steps": num_steps}
    )


class ActHandler(http.server.BaseHTTPRequestHandler):
    policy = None  # set in main()

    def do_POST(self):
        if self.path.rstrip("/") != "/act":
            self.send_error(404, "POST /act only")
            return
        try:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json_numpy.loads(body)
            obs = {
                "images": {
                    "top": np.asarray(payload["top_cam"]),
                    "left": np.asarray(payload["left_cam"]),
                    "right": np.asarray(payload["right_cam"]),
                },
                "state": np.asarray(payload["state"], dtype=np.float32),
                "prompt": str(payload.get("instruction") or ""),
            }
            t0 = time.perf_counter()
            actions = np.asarray(self.policy.infer(obs)["actions"])[:, :14]
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # surface the reason to the client, not a bare 500
            self.send_error(500, f"{type(exc).__name__}: {exc}")
            return
        resp = json_numpy.dumps({"actions": actions, "dt_ms": DT_MS}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
        print(f"/act: chunk {actions.shape} in {elapsed_ms:.0f} ms", flush=True)

    def log_message(self, *args):  # default per-request stderr noise
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing params/ and assets/")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8204)
    ap.add_argument("--num-steps", type=int, default=10, help="flow-matching denoising steps")
    args = ap.parse_args()

    print(f"loading policy from {args.ckpt} ...", flush=True)
    ActHandler.policy = load_policy(args.ckpt, args.num_steps)

    # Warm up: the first infer JIT-compiles (can take minutes), which would
    # blow through the client's 30 s request timeout if paid on a live request.
    print("warming up (JIT compile) ...", flush=True)
    t0 = time.perf_counter()
    ActHandler.policy.infer(
        {
            "images": {
                cam: np.zeros((360, 640, 3), dtype=np.uint8) for cam in ("top", "left", "right")
            },
            "state": np.zeros(14, dtype=np.float32),
            "prompt": "warmup",
        }
    )
    print(f"warmup done in {time.perf_counter() - t0:.1f} s", flush=True)

    server = http.server.HTTPServer((args.host, args.port), ActHandler)
    print(f"serving pi05-yam on http://{args.host}:{args.port}/act", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
