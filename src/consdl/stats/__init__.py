"""Per-checkpoint firing statistics: mean activation, sparsity, top
indices, L0.

``SAEInterpreter`` is re-exported lazily so that ``import consdl.stats``
does not import transformers. The stats CLIs call
``load_dotenv(override=True)`` after argument parsing, and transformers
reads its cache location at import time.
"""

__all__ = ["SAEInterpreter"]


def __getattr__(name):
    if name == "SAEInterpreter":
        from consdl.stats.interpreter import SAEInterpreter  # noqa: PLC0415
        return SAEInterpreter
    raise AttributeError(name)
