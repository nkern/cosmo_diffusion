"""Regression tests for opt-in, fresh-only DiT zero initialization."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from diffusers import DiTTransformer2DModel, UNet2DModel

from cosmodiff.initialization import initialize_dit_adaln_zero
from cosmodiff.utils import parse_config_model


def _dit_kwargs(patch_size=4, num_layers=2):
    return dict(
        sample_size=8, patch_size=patch_size, in_channels=1, out_channels=1,
        num_layers=num_layers, num_attention_heads=2, attention_head_dim=8,
        num_embeds_ada_norm=10, norm_type="ada_norm_zero",
    )


def _state(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


@pytest.mark.parametrize("patch_size,num_layers", [(4, 2), (8, 16)])
def test_only_modulation_and_output_parameters_are_zeroed(patch_size, num_layers):
    model = DiTTransformer2DModel(**_dit_kwargs(patch_size, num_layers))
    before = _state(model)
    report = initialize_dit_adaln_zero(model, fresh=True)
    targets = {
        f"{name}.{parameter}"
        for name in report["linear_modules"] for parameter in ("weight", "bias")
    }
    assert report["zeroed_tensors"] == 2 * (num_layers + 2)
    assert len(targets) == report["zeroed_tensors"]
    assert all(parameter.requires_grad for parameter in model.parameters())
    for name, value in model.state_dict().items():
        if name in targets:
            assert torch.count_nonzero(value) == 0, name
        else:
            assert torch.equal(value, before[name]), name


def test_zero_prediction_and_output_head_can_learn():
    torch.manual_seed(123)
    model = DiTTransformer2DModel(**_dit_kwargs())
    initialize_dit_adaln_zero(model, fresh=True)
    images = torch.randn(2, 1, 8, 8)
    timesteps = torch.tensor([1, 2])
    labels = torch.zeros(2, dtype=torch.long)
    output = model(images, timesteps, class_labels=labels).sample
    assert torch.count_nonzero(output) == 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss = (output - torch.ones_like(output)).square().mean()
    loss.backward()
    assert torch.isfinite(model.proj_out_2.weight.grad).all()
    assert torch.count_nonzero(model.proj_out_2.weight.grad) > 0
    optimizer.step()
    learned = model(images, timesteps, class_labels=labels).sample
    assert torch.isfinite(learned).all()
    assert torch.count_nonzero(learned) > 0


def test_initialized_blocks_are_identity_mappings():
    model = DiTTransformer2DModel(**_dit_kwargs())
    initialize_dit_adaln_zero(model, fresh=True)
    features = torch.randn(2, 4, 16)
    timesteps = torch.tensor([1, 2])
    labels = torch.zeros(2, dtype=torch.long)
    for block in model.transformer_blocks:
        output = block(features, timestep=timesteps, class_labels=labels)
        assert torch.equal(output, features)


@pytest.mark.parametrize("fresh", [False, True])
def test_reinitialization_is_rejected_without_mutation(fresh):
    model = DiTTransformer2DModel(**_dit_kwargs())
    if fresh:
        initialize_dit_adaln_zero(model, fresh=True)
    before = _state(model)
    with pytest.raises(RuntimeError, match="fresh"):
        initialize_dit_adaln_zero(model, fresh=fresh)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_incompatible_projection_does_not_partially_zero_model():
    model = DiTTransformer2DModel(**_dit_kwargs())
    model.proj_out_1 = torch.nn.Linear(16, 1)
    before = _state(model)
    with pytest.raises(ValueError, match="proj_out_1"):
        initialize_dit_adaln_zero(model, fresh=True)
    assert not getattr(model, "_cosmodiff_adaln_zero_initialized", False)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_wrong_normalization_is_rejected():
    model = DiTTransformer2DModel(**_dit_kwargs())
    model.transformer_blocks[-1].norm1 = torch.nn.LayerNorm(16)
    before = _state(model)
    with pytest.raises(TypeError, match="ada_norm_zero"):
        initialize_dit_adaln_zero(model, fresh=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_wrong_model_class_is_rejected():
    with pytest.raises(TypeError, match="DiTTransformer2DModel"):
        initialize_dit_adaln_zero(torch.nn.Linear(2, 2), fresh=True)


@pytest.mark.parametrize("explicit_native", [False, True])
def test_default_dit_initialization_is_unchanged(explicit_native):
    config = {"model": {"class": "DiTTransformer2DModel", "kwargs": _dit_kwargs()}}
    if explicit_native:
        config["model"]["initialization"] = "native"
    torch.manual_seed(123)
    expected = DiTTransformer2DModel(**_dit_kwargs())
    torch.manual_seed(123)
    actual = parse_config_model(config)["model"]
    for name, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[name])


def test_config_opt_in_initializes_before_optimizer_without_changing_config():
    config = {
        "model": {"class": "DiTTransformer2DModel", "initialization": "adaln_zero",
                  "kwargs": _dit_kwargs()},
        "optimizer": {"class": "AdamW", "kwargs": {"lr": 1e-4}},
    }
    original = copy.deepcopy(config)
    result = parse_config_model(config)
    model = result["model"]
    assert model._cosmodiff_adaln_zero_initialized
    assert torch.count_nonzero(model.proj_out_2.weight) == 0
    assert config == original
    parameters = result["optimizer"].param_groups[0]["params"]
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in model.parameters()
    }


@pytest.mark.parametrize("initialization", ["unknown", None])
def test_unknown_initialization_policy_is_rejected(initialization):
    config = {"model": {"class": "DiTTransformer2DModel", "initialization": initialization,
                        "kwargs": _dit_kwargs()}}
    with pytest.raises(ValueError, match="model.initialization"):
        parse_config_model(config)


def test_config_rejects_zero_init_for_unet():
    config = {"model": {"class": "UNet2DModel", "initialization": "adaln_zero"}}
    with pytest.raises(ValueError, match="requires DiTTransformer2DModel"):
        parse_config_model(config)


def test_unet_native_initialization_is_unchanged():
    kwargs = dict(sample_size=8, in_channels=1, out_channels=1, layers_per_block=1,
                  block_out_channels=(8,), down_block_types=("DownBlock2D",),
                  up_block_types=("UpBlock2D",), norm_num_groups=4)
    torch.manual_seed(123)
    expected = UNet2DModel(**kwargs)
    torch.manual_seed(123)
    actual = parse_config_model({"model": {"class": "UNet2DModel", "kwargs": kwargs}})["model"]
    for name, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[name])


def test_save_load_preserves_learned_nonzero_projections(tmp_path):
    model = parse_config_model({"model": {
        "class": "DiTTransformer2DModel", "initialization": "adaln_zero",
        "kwargs": _dit_kwargs(),
    }})["model"]
    with torch.no_grad():
        model.proj_out_2.weight.fill_(0.125)
        model.proj_out_2.bias.fill_(0.25)
        model.transformer_blocks[0].norm1.linear.weight.fill_(0.375)
    model.save_pretrained(tmp_path)
    loaded = DiTTransformer2DModel.from_pretrained(tmp_path)
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    with pytest.raises(RuntimeError, match="fresh"):
        initialize_dit_adaln_zero(loaded, fresh=False)
    assert torch.all(loaded.proj_out_2.weight == 0.125)


def test_checkpoint_loader_preserves_learned_nonzero_projections(tmp_path):
    import yaml
    from diffusers import DDPMScheduler
    from cosmodiff.utils import load_checkpoint

    model = DiTTransformer2DModel(**_dit_kwargs())
    initialize_dit_adaln_zero(model, fresh=True)
    with torch.no_grad():
        model.proj_out_2.weight.fill_(0.125)
        model.transformer_blocks[0].norm1.linear.bias.fill_(0.25)
    model.save_pretrained(tmp_path)
    DDPMScheduler(num_train_timesteps=10).save_pretrained(tmp_path)
    checkpoint_config = {
        "model": {"initialization": "adaln_zero"},
        "noise_scheduler": {"class": "diffusers.DDPMScheduler"},
        "optimizer": {"class": "torch.optim.AdamW"},
        "lr_scheduler": {"class": "torch.optim.lr_scheduler.ConstantLR",
                         "kwargs": {"factor": 1.0, "total_iters": 0}},
    }
    (tmp_path / "checkpoint_config.yaml").write_text(yaml.safe_dump(checkpoint_config))
    loaded, _, optimizer, _, _ = load_checkpoint(str(tmp_path))
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter) for parameter in loaded.parameters()
    }


def test_training_script_resume_bypasses_fresh_model_factory(tmp_path, monkeypatch):
    import yaml
    from cosmodiff import optim, utils

    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump({"io": {"output_dir": str(tmp_path / "output")},
                                         "train": {}, "data": {}}))
    checkpoint = str(tmp_path / "checkpoint")
    monkeypatch.setattr(utils, "find_latest_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(utils, "parse_config_data", lambda _: {"data": object()})

    def unexpected_fresh_model(_):
        raise AssertionError("resume must not reconstruct/reinitialize the model")

    monkeypatch.setattr(utils, "parse_config_model", unexpected_fresh_model)
    calls = []

    def fake_train(dataset, **kwargs):
        calls.append(kwargs)
        return {"metrics": {"epoch_loss": [0.1], "epoch_times": [1.0]}}

    monkeypatch.setattr(optim, "train", fake_train)
    script = Path(__file__).resolve().parents[2] / "scripts" / "cosmodiff_train.py"
    spec = importlib.util.spec_from_file_location("cosmodiff_train_zero_init_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script), "--config", str(config_path)])
    module.main()
    assert len(calls) == 1
    assert calls[0]["resume_from_checkpoint"] == checkpoint
