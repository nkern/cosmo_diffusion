"""Data normalization transforms for cosmodiff.

Provides three pointwise normalization functions and a :class:`Normalization`
``nn.Module`` that wraps them with a uniform forward / inverse interface.
"""

import torch


def minmax_norm(
    x: torch.Tensor,
    xmin: float | None = None,
    xmax: float | None = None,
    inverse: bool = False,
    inplace: bool = False,
    **kwargs
) -> torch.Tensor:
    """Normalize a tensor to ``[-1, 1]`` via min-max scaling.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        xmin (float): normalize by min, default is x.min()
        xmax (float): normalize by max, default is x.max()
        inverse (bool): if True, apply the inverse mapping (requires xmin, xmax)
        inplace (bool): edit tensor inplace, default is False

    Returns:
        torch.Tensor: Normalized tensor with values in ``[-1, 1]``.
        dict: normalization parameters
    """
    if not inplace:
        x = x.clone()
    if not inverse:
        if xmin is None:
            xmin = x.min().item()
        if xmax is None:
            xmax = x.max().item()
        x -= xmin
        x *= 2 / xmax
        x -= 1

    else:
        assert xmin is not None
        assert xmax is not None
        x = (x + 1) / 2 * xmax + xmin

    return x, {'xmin': xmin, 'xmax': xmax}


def center_max_norm(
    x: torch.Tensor,
    center: float | None = None,
    xmax: float | None = None,
    inverse: bool | None = False,
    inplace: bool = False,
    **kwargs
):
    """Center a tensor based on its average, and normalize by its absolute deviation.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        center (float): average centering to subtract off
        xmax (float): abs-max normalization to divide by
        inverse (bool): apply inverse operation, requires center and xmax
        inplace (bool): If True, edit inplace.

    Returns:
        torch.Tensor: scaled tensor
        dict: normalization parameters
    """
    if not inplace:
        x = x.clone()

    if not inverse:
        # center
        if center is None:
            center = x.mean().item()
        x -= center

        # scale by max-abs
        if xmax is None:
            xmax = x.abs().max().item()
        x /= xmax

    else:
        assert xmax is not None
        assert center is not None
        x *= xmax
        x += center

    return x, {'center': center, 'xmax': xmax}


def tanh_norm(x, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, sigma=1.0, mu=0.0, inverse=False, **kwargs):
    """
    tanh normalization

        f(x) = sigma * alpha * tanh((gamma * (x-mu)) / alpha) if (x-mu) >= 0
        f(x) = sigma * beta  * tanh((delta * (x-mu)) / beta)  if (x-mu) < 0

    Args:
        x (tensor): Input tensor
        alpha (float): Positive saturation limit (upper bound).
        beta (float): Negative saturation limit (lower bound).
        gamma (float): Multiplicative gain for the positive side.
        delta (float): Multiplicative gain for the negative side.
        sigma (float): Final scaling
        mu (float): Mean shift / center point.
        inverse (bool): Toggle between forward and inverse operations.

    Returns:
        torch.Tensor: normalized data
        dict: normalization parameters
    """
    if not inverse:
        # Forward: x -> y
        x_shifted = x - mu
        pos = alpha * torch.tanh((gamma * x_shifted) / alpha)
        neg = beta * torch.tanh((delta * x_shifted) / beta)
        y = torch.where(x_shifted >= 0, pos, neg) * sigma
    else:
        # Inverse: y -> x
        # Note: data (y) should be clamped within (-beta, alpha) for stability
        y = torch.clamp(x / sigma, -beta + 1e-9, alpha - 1e-9)
        pos_inv = (alpha * torch.atanh(y / alpha)) / gamma
        neg_inv = (beta * torch.atanh(y / beta)) / delta
        y =  torch.where(y >= 0, pos_inv, neg_inv) + mu

    params = {'mu': mu, 'alpha': alpha, 'beta': beta, 'delta': delta, 'gamma': gamma, 'sigma': sigma}
    return y, params


class Normalization(torch.nn.Module):
    """Invertible pointwise data normalization.

    Wraps :func:`minmax_norm`, :func:`center_max_norm`, and :func:`tanh_norm`
    behind a single object with consistent ``forward`` / ``inverse`` calls.
    On the first forward pass, any normalization parameters not supplied at
    construction (e.g. ``xmin``, ``xmax``, ``center``) are estimated from the
    input tensor and stored on ``self.kwargs``; subsequent forward and inverse
    calls reuse those fitted parameters so the mapping stays consistent
    across batches.

    Args:
        method (str): which normalization to apply. One of:

            * ``'minmax'`` / ``'min-max'``: linear rescale to ``[-1, 1]``.
            * ``'centermax'`` / ``'center-max'``: subtract mean, divide by
              max-abs.
            * ``'tanh'``: ``center-max`` followed by :func:`tanh_norm` for
              soft saturation of heavy tails.

        inplace (bool): edit the tensor in place when supported. Defaults to
            ``False`` (returns a new tensor).
        **kwargs: parameters forwarded to the underlying normalization
            function (e.g. ``xmin``, ``xmax``, ``center``, ``alpha``,
            ``beta``, ``gamma``, ``delta``, ``sigma``, ``mu``).  Any value
            left as ``None`` is estimated from data on the first forward pass
            and persisted on ``self.kwargs``.

    Example::

        norm = Normalization('center-max')
        y = norm(x)              # fit on x, return normalized
        x_back = norm.inverse(y) # exactly recovers x
    """

    def __init__(self, method: str, inplace: bool = False, **kwargs):
        super().__init__()
        self.method = method
        self.inplace = inplace
        self.kwargs = kwargs

    def forward(self, x):
        if self.method in ['minmax', 'min-max']:
            x, kw = minmax_norm(x, inplace=self.inplace, **self.kwargs)
            self.kwargs.update(kw)

        elif self.method in ['centermax', 'center-max']:
            x, kw = center_max_norm(x, inplace=self.inplace, **self.kwargs)
            self.kwargs.update(kw)

        elif self.method in ['tanh']:
            x, kw = center_max_norm(x, inplace=self.inplace, **self.kwargs)
            self.kwargs.update(kw)
            x, kw = tanh_norm(x, inplace=self.inplace, **self.kwargs)
            self.kwargs.update(kw)

        return x

    def inverse(self, x):
        if self.method in ['minmax', 'min-max']:
            x, _ = minmax_norm(x, inplace=self.inplace, inverse=True, **self.kwargs)
        elif self.method in ['centermax', 'center-max']:
            x, _ = center_max_norm(x, inplace=self.inplace, inverse=True, **self.kwargs)
        elif self.method in ['tanh']:
            x, _ = tanh_norm(x, inplace=self.inplace, inverse=True, **self.kwargs)
            x, _ = center_max_norm(x, inplace=self.inplace, inverse=True, **self.kwargs)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(method={self.method})"


class MultiNormalization(torch.nn.Module):
    """
    Multiple normalizations applied to batch dimension
    of data for a set of discrete class labels.
    WIP
    """

    def __init__(self, classes, norms):
        self.classes = classes
        self.norms = norms

    def forward(self, x, labels):
        for lbl, norm in zip(self.classes, self.norms):
            idx = labels == lbl
            x[idx] = norm(x[idx])

        return x

    def inverse(self, x, labels):
        for lbl, norm in zip(self.classes, self.norms):
            idx = labels == lbl
            x[idx] = norm.inverse(x[idx])

        return x


class Transform(torch.nn.Module):
    """Composable, invertible data transformation pipeline.

    Applies an optional ``log`` transform first, then iterates through a list
    of named operations in ``ops``.  Each operation must have a defined
    inverse; calling :meth:`inverse` runs the operations in reverse order.

    Args:
        ops (list of str, optional): ordered list of operation names to apply
            after the log step.  Currently supported:

            * ``'fft2'``: 2D FFT; real and imaginary parts are concatenated
              along the channel dimension to keep the tensor real-valued.
              Actually uses fft2 so that we don't change size of image.
              But negative frequencies are discarded upon ifft2.

            Defaults to an empty pipeline if ``None``.
        log (bool): if ``True``, take ``log()`` before any ``ops`` (and
            ``exp()`` last on inverse).  Useful for log-normal data such as
            cosmological matter density fields.  Defaults to ``False``.
        inplace (bool): edit the tensor in place when supported. Defaults
            to ``False``.
    """
    def __init__(
        self,
        ops: list[str] | None = None,
        log: bool = False,
        inplace: bool = False,
        **kwargs,
        ):
        super().__init__()
        self.ops = ops
        self.log = log
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            x = x.clone()

        # first operation is log
        if self.log:
            x.log_()

        # now perform other operations
        for op in self.ops:
            if op == 'fft2':
                x = torch.fft.fft2(x)
                x = torch.cat([x.real, x.imag], dim=1)

            else:
                raise ValueError(f"didn't recognize '{op}'")

        return x

    def inverse(self, x):
        if not self.inplace:
            x = x.clone()

        # go through operations in reverse order
        for op in self.ops[::-1]:
            if op == 'fft2':
                N = x.shape[1]//2
                x = torch.complex(x[:, :N], x[:, N:])
                x = torch.fft.ifft2(x).real

            else:
                raise ValueError(f"didn't recognize '{op}'")

        # last operation is unlog
        if self.log:
            x.exp_()

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(ops={self.ops})"


class MultiTransform(torch.nn.Module):
    """
    Multiple transformations applied to batch dimension
    of data for a set of discrete class labels.
    WIP
    """
    def __init__(self, classes, tforms):
        self.classes = classes
        self.tforms = tforms

    def forward(self, x, labels):
        for lbl, tform in zip(self.classes, self.tforms):
            idx = labels == lbl
            x[idx] = tform(x[idx])

        return x

    def inverse(self, x):
        for lbl, tform in zip(self.classes, self.tforms):
            idx = labels == lbl
            x[idx] = tform.inverse(x[idx])

        return x

