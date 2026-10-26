"""Agent nodes for the aidevteam pipeline.

Naming convention (one name everywhere — graph node, container/service, file,
function): ``agents/<name>.py`` must define a function ``<name>(state) -> dict``.
``load_node`` is the single authority that enforces the convention.
"""

import importlib
from typing import Callable


def load_node(name: str) -> Callable[[dict], dict]:
    """Load the node function for `name` following the naming convention.

    Raises a descriptive error when the convention is violated, so adding a
    new agent is just: create ``agents/<name>.py`` with ``def <name>(state)``
    and wire it into ``build_graph``.
    """
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"Unknown agent {name!r}: expected the module "
            f"{__name__}.{name} to exist (naming convention: "
            f"agents/<name>.py defines <name>())."
        ) from exc
    if not hasattr(module, name):
        raise AttributeError(
            f"Naming convention violated: agents/{name}.py must define "
            f"a function {name}(state) -> dict."
        )
    return getattr(module, name)