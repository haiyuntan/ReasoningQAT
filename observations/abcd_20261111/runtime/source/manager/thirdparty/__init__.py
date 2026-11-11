
"""Load only the requested backend, including during Lean/retrieval checks."""
from importlib import import_module

_MODULES = {'Interactive': 'interactive', 'LeanSearch': 'lean_search',
            'Claude': 'claude', 'TacticGenerator': 'generator', 'Critic': 'critic'}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module('.' + _MODULES[name], __name__), name)
    globals()[name] = value
    return value
