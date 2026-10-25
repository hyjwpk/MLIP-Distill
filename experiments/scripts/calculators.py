"""Load archived student weights without their old training-data paths."""

import copy
from pathlib import Path

import torch
from fairchem.core import OCPCalculator
from fairchem.core.common.registry import registry
from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage
from fairchem.core.trainers.ocp_trainer import OCPTrainer


@registry.register_trainer("archived_inference")
class ArchivedInferenceTrainer(OCPTrainer):
    def load_checkpoint(self, checkpoint_path, checkpoint=None, inference_only=True):
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # This FAIR-Chem version skips EMA initialization in inference-only mode.
        self.ema = None
        if checkpoint.get("ema") is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), checkpoint["ema"]["decay"])
        return super().load_checkpoint(checkpoint_path, checkpoint, inference_only=True)


def student_calculator(checkpoint, weights, seed=0, device="cuda"):
    if weights not in ("raw", "ema"):
        raise ValueError("weights must be raw or ema")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if weights == "ema" and saved.get("ema") is None:
        raise ValueError(f"Checkpoint has no EMA parameters: {checkpoint}")
    config = copy.deepcopy(saved["config"])
    if "model_attributes" in config:
        config["model_attributes"]["name"] = config["model"]
        config["model"] = config.pop("model_attributes")
    if config["model"].get("scale_file"):
        scale = "painn.pt" if config["model"]["name"] == "painn" else "gemnet.json"
        config["model"]["scale_file"] = str(Path(__file__).resolve().parents[1] / "runs/configs/scales" / scale)
    config["dataset"] = {"train": {"src": "unused-in-inference"}}
    config["is_debug"] = True
    config.pop("logger", None)
    config["outputs"] = {k: v for k, v in config["outputs"].items() if k in ("energy", "forces")}
    config["loss_functions"] = [v for v in config["loss_functions"] if next(iter(v)) in ("energy", "forces")]
    calculator = OCPCalculator(config_yml=config, checkpoint_path=str(checkpoint),
                               trainer="archived_inference", cpu=device == "cpu", seed=seed)
    if weights == "raw":
        calculator.trainer.ema = None
    return calculator
