import os


_ROOT = os.path.dirname(os.path.dirname(__file__))
_SRC_PACKAGE = os.path.join(_ROOT, "src", "matchmaking_data")

__path__ = [_SRC_PACKAGE]
