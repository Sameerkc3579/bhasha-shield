"""Build any defense by name -- the one place the grid looks them up.

    from defenses.registry import build, DEFENSES
    d = build("camel_lite", planner=client)

Class A defenses take `threshold=`/`scorer=`/`device=`; the two planner-based
Class C defenses need `planner=` (a ModelClient, normally the agent's own).

Phase 6 normalized classifiers are named `<mode>+<classifier>`, e.g.
`translit+protectai_deberta` (see defenses/normalize.py). Keyword arguments go
to the inner classifier; `translate+...` uses `planner=` as its translator, the
same free local model.
"""

from __future__ import annotations

from typing import Any, Callable

from defenses.base import Defense, NullDefense

# name -> (class, needs a planner). Imports are deferred so building one defense
# never imports another's heavy dependencies.
_SPECS: dict[str, tuple[str, bool]] = {
    "undefended": ("defenses.base:NullDefense", False),
    # A -- classifiers
    "promptguard2": ("defenses.classifiers.promptguard2:PromptGuard2", False),
    "protectai_deberta": ("defenses.classifiers.protectai_deberta:ProtectAIDeberta", False),
    "xlmr_finetuned": ("defenses.classifiers.xlmr_finetuned:XLMRFinetuned", False),
    # B -- prompting
    "spotlight": ("defenses.prompting.spotlight:Spotlight", False),
    "sandwich": ("defenses.prompting.sandwich:Sandwich", False),
    "hierarchy": ("defenses.prompting.hierarchy:Hierarchy", False),
    # C -- architectural
    "allowlist": ("defenses.architectural.allowlist:Allowlist", True),
    "camel_lite": ("defenses.architectural.camel_lite:CamelLite", True),
    "honeytool": ("defenses.architectural.honeytool:Honeytool", False),
}

DEFENSES: tuple[str, ...] = tuple(n for n in _SPECS if n != "undefended")
CLASSIFIERS: tuple[str, ...] = ("promptguard2", "protectai_deberta", "xlmr_finetuned")


def is_known(name: str) -> bool:
    """A base defense, or `<mode>+<classifier>` with a valid mode."""
    from defenses.normalize import MODES

    if "+" in name:
        mode, inner = name.split("+", 1)
        return mode in MODES and inner in CLASSIFIERS
    return name in _SPECS


def needs_planner(name: str) -> bool:
    return name.startswith("translate+") or _SPECS.get(name, ("", False))[1]


def _load(path: str) -> Callable[..., Defense]:
    import importlib

    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)


def build(name: str, *, planner: Any = None, **kwargs: Any) -> Defense:
    if "+" in name and is_known(name):
        from defenses.normalize import NormalizedClassifier, Normalizer, llm_translator

        mode, inner = name.split("+", 1)
        if mode == "translate" and planner is None:
            raise ValueError(f"{name} needs planner= (a ModelClient used as the translator)")
        translator = llm_translator(planner) if mode == "translate" else None
        return NormalizedClassifier(build(inner, **kwargs), Normalizer(mode, translator))
    if name not in _SPECS:
        raise KeyError(f"unknown defense {name!r}; known: {', '.join(_SPECS)}")
    path, needs_planner = _SPECS[name]
    cls = _load(path)
    if needs_planner:
        if planner is None:
            raise ValueError(f"{name} needs planner= (a ModelClient)")
        return cls(planner, **kwargs)
    return cls(**kwargs)


__all__ = ["CLASSIFIERS", "DEFENSES", "NullDefense", "build", "is_known", "needs_planner"]
