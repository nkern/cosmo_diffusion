import copy
import numpy as np
import torch
from cosmodiff.transform import (
    minmax_norm,
    center_max_norm,
    tanh_norm,
    Normalization,
    Transform,
)


# ---------------------------------------------------------------------------
# minmax_norm / center_max_norm / tanh_norm
# ---------------------------------------------------------------------------

def test_minmax_norm():
    # use non-negative data so the formula x *= 2/xmax is well-defined
    x = torch.rand(4, 8, 8)
    out, params = minmax_norm(x)
    assert out.min().item() >= -1.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6
    assert 'xmin' in params and 'xmax' in params


def test_minmax_norm_inverse():
    x = torch.rand(4, 8, 8)
    out, params = minmax_norm(x)
    recovered, _ = minmax_norm(out, inverse=True, **params)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_minmax_norm_inplace():
    x = torch.rand(4, 8, 8)
    x_clone = x.clone()
    out, _ = minmax_norm(x, inplace=True)
    assert out.data_ptr() == x.data_ptr()
    _, params2 = minmax_norm(x_clone)
    assert torch.allclose(out, x_clone.sub(params2['xmin']).mul(2 / params2['xmax']).sub(1), atol=1e-5)


def test_center_max_norm():
    x = torch.randn(100)
    out, params = center_max_norm(x.clone())
    assert abs(out.mean().item()) < 0.1
    assert out.abs().max().item() <= 1.0 + 1e-6
    assert 'center' in params and 'xmax' in params


def test_center_max_norm_inverse():
    x = torch.randn(100)
    out, params = center_max_norm(x.clone())
    recovered, _ = center_max_norm(out, inverse=True, **params)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_tanh_norm():
    x = torch.randn(100)
    out, params = tanh_norm(x)
    assert out.shape == x.shape
    assert set(params.keys()) == {'mu', 'alpha', 'beta', 'delta', 'gamma', 'sigma'}
    # tanh output is strictly bounded by sigma * (-beta, alpha)
    assert out.min().item() > -params['beta'] * params['sigma']
    assert out.max().item() < params['alpha'] * params['sigma']


def test_tanh_norm_sigma():
    x = torch.randn(100)
    out_default, _ = tanh_norm(x, sigma=1.0)
    out_scaled, params = tanh_norm(x, sigma=2.0)
    assert torch.allclose(out_scaled, out_default * 2.0, atol=1e-6)
    assert params['sigma'] == 2.0


def test_tanh_norm_inverse():
    x = torch.randn(100) * 0.5  # keep within tanh saturation limits
    out, params = tanh_norm(x)
    recovered, _ = tanh_norm(out, inverse=True, **params)
    assert torch.allclose(recovered, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Normalization class
# ---------------------------------------------------------------------------

def test_normalization_minmax():
    x = torch.rand(10, 1, 8, 8)
    norm = Normalization('min-max', inplace=False)
    out = norm(x)
    assert out is not None
    assert out.shape == x.shape
    assert out.min().item() >= -1.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_normalization_centermax():
    x = torch.randn(10, 1, 8, 8)
    norm = Normalization('center-max', inplace=False)
    out = norm(x)
    assert out is not None
    assert out.shape == x.shape
    assert out.abs().max().item() <= 1.0 + 1e-6


def test_normalization_inverse():
    x = torch.rand(10, 1, 8, 8)
    norm = Normalization('min-max', inplace=False)
    out = norm(x)
    recovered = norm.inverse(out)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_normalization_tanh():
    # tanh branch: minmax first, then tanh — output is strictly within sigma*(-beta, alpha)
    x = torch.rand(10, 1, 8, 8)
    norm = Normalization('tanh', inplace=False)
    out = norm(x)
    assert out.shape == x.shape
    assert out.min().item() > -norm.kwargs['beta'] * norm.kwargs['sigma']
    assert out.max().item() < norm.kwargs['alpha'] * norm.kwargs['sigma']
    # params from both stages must be stored after forward
    assert all(k in norm.kwargs for k in ('center', 'xmax', 'mu', 'alpha', 'beta', 'gamma', 'delta', 'sigma'))


def test_normalization_tanh_inverse():
    x = torch.rand(10, 1, 8, 8)
    norm = Normalization('tanh', inplace=False)
    out = norm(x)
    recovered = norm.inverse(out)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_normalization_params_fixed_when_given():
    """Params supplied at init must not be overwritten by the forward pass."""
    fixed_xmin = torch.tensor(0.0)
    fixed_xmax = torch.tensor(2.0)
    norm = Normalization('min-max', inplace=False, xmin=fixed_xmin, xmax=fixed_xmax)

    # forward on data whose natural min/max differ from the fixed values
    x = torch.rand(10, 1, 8, 8) * 10 + 5
    norm(x)

    assert torch.equal(norm.kwargs['xmin'], fixed_xmin), "xmin was overwritten"
    assert torch.equal(norm.kwargs['xmax'], fixed_xmax), "xmax was overwritten"


def test_normalization_params_inferred_on_first_forward():
    """Params not supplied at init must be populated after the first forward pass."""
    norm = Normalization('min-max', inplace=False)
    assert 'xmin' not in norm.kwargs
    assert 'xmax' not in norm.kwargs

    x = torch.rand(10, 1, 8, 8)
    norm(x)

    assert 'xmin' in norm.kwargs, "xmin not set after first forward"
    assert 'xmax' in norm.kwargs, "xmax not set after first forward"
    assert np.allclose(norm.kwargs['xmin'], x.min().item())
    assert np.allclose(norm.kwargs['xmax'], x.max().item())

    # second forward on different data must reuse the inferred params, not recompute
    y = torch.rand(10, 1, 8, 8) * 10 + 5
    xmin_after_first = copy.copy(norm.kwargs['xmin'])
    xmax_after_first = copy.copy(norm.kwargs['xmax'])
    norm(y)

    assert np.allclose(norm.kwargs['xmin'], xmin_after_first), "xmin changed on second forward"
    assert np.allclose(norm.kwargs['xmax'], xmax_after_first), "xmax changed on second forward"


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------

def test_transform_log_roundtrip():
    """log forward + inverse recovers the original tensor."""
    t = Transform(ops=[], log=True)
    x = torch.rand(2, 1, 8, 8) + 0.5  # strictly positive
    y = t(x)
    assert torch.allclose(y, x.log(), atol=1e-6)
    x_back = t.inverse(y)
    assert torch.allclose(x_back, x, atol=1e-5)


def test_transform_rfft2_shape_and_roundtrip():
    """rfft2 op doubles the channel dim (real|imag) and round-trips back."""
    t = Transform(ops=['rfft2'], log=False)
    x = torch.randn(3, 2, 8, 8)
    y = t(x)
    # real and imaginary parts concatenated along channel dim
    assert y.shape == (3, 4, 8, 8), f"expected channel doubling, got {y.shape}"
    x_back = t.inverse(y)
    assert x_back.shape == x.shape
    assert torch.allclose(x_back, x, atol=1e-5)


def test_transform_log_then_rfft2_roundtrip():
    """log → rfft2 forward; ifft2 → exp inverse round-trips."""
    t = Transform(ops=['rfft2'], log=True)
    x = torch.rand(2, 1, 8, 8) + 0.5  # strictly positive so log is well-defined
    y = t(x)
    assert y.shape == (2, 2, 8, 8)
    x_back = t.inverse(y)
    assert torch.allclose(x_back, x, atol=1e-5)


def test_transform_inplace_false_does_not_mutate_input():
    """Default inplace=False leaves the input tensor untouched."""
    t = Transform(ops=[], log=True)
    x = torch.rand(2, 1, 8, 8) + 0.5
    x_orig = x.clone()
    _ = t(x)
    assert torch.equal(x, x_orig), "input was mutated despite inplace=False"


def test_transform_unknown_op_raises():
    """Unknown op names raise a clear ValueError."""
    t = Transform(ops=['not-a-real-op'])
    x = torch.randn(2, 1, 8, 8)
    try:
        t(x)
    except ValueError as e:
        assert "not-a-real-op" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown op")
