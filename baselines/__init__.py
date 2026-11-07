"""Baseline models. build(config) takes one file of baselines/configs and returns an untrained model.

    model = baselines.build(yaml.safe_load(open("baselines/configs/enwik8/large/gru.yaml")))
    logits = model(tokens)      # int64 (B, L) -> float32 (B, L, classes)
"""
import importlib

MODULES = {"gru": "gru", "lstm": "lstm", "mamba2": "mamba2", "transformer": "transformer",
           "gated_deltanet": "gated_deltanet", "deltaproduct2": "deltaproduct", "deltaproduct4": "deltaproduct",
           "hyenadna": "hyenadna"}


def module(name):
    """The module of a baseline name; holds build and AMP."""
    return importlib.import_module(f"{__name__}.{MODULES[name]}")


def build(config):
    """config: model, task and the shape keys of the model (optimiser keys are ignored)."""
    return module(config["model"]).build(config)
