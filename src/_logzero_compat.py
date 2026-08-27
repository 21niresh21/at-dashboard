"""Compatibility shim for logzero on Python 3.11+.

logzero uses inspect.getargspec() which was removed in Python 3.11.
This module patches the missing function before smartapi-python imports logzero.
"""

from __future__ import annotations

import inspect
import sys

# Only patch if getargspec is missing (Python 3.11+)
if not hasattr(inspect, "getargspec"):

    def _getargspec(func):
        """Backward-compatible replacement for inspect.getargspec."""
        sig = inspect.signature(func)
        args = []
        varargs = None
        keywords = None
        defaults = []

        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                varargs = name
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                keywords = name
            elif param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            ):
                args.append(name)
                if param.default is not inspect.Parameter.empty:
                    defaults.append(param.default)

        # Build a named tuple-like object matching getargspec signature
        class Argspec:
            def __init__(self, args, varargs, keywords, defaults):
                self.args = args
                self.varargs = varargs
                self.keywords = keywords
                self.defaults = tuple(defaults) if defaults else None

        return Argspec(args, varargs, keywords, defaults)

    inspect.getargspec = _getargspec  # type: ignore[attr-defined]
