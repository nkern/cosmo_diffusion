from importlib.metadata import version as _pkg_version

from . import utils
from . import augment
from . import optim

from .optim import train, generate

__version__ = _pkg_version("cosmodiff")
