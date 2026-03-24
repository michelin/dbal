"""Main module of dbal."""

try:
    from .version import __version__, __version_date__
except ImportError:
    __version__ = None
    __version_date__ = ""
