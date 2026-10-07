"""SAE architectures and their architecture-specific aux trainers.

  * :mod:`consdl.models.fista`      -- ``FistaSAE``: ConSDL and its ablations
    (sparse coding, + refitting); co-active similarity regularization
    lives in the trainer (:mod:`consdl.training.trainer`).
  * :mod:`consdl.models.topk`       -- ``TopKSAE`` + ``TopKAuxLossTrainer``.
  * :mod:`consdl.models.batchtopk`  -- ``BatchTopKSAE``, ``MatryoshkaBatchTopKSAE``.
  * :mod:`consdl.models.mp`         -- ``MPSAE`` + ``MPRevivalLossTrainer`` and
    the eval-time MP inference helpers.
  * :mod:`consdl.models.isae`       -- ``IsaeSAE`` / ``IsaeMESAE`` + ``IsaeAuxRipTrainer``.
  * :mod:`consdl.models.resampling` -- ReLU SAE neuron resampling trainer (LM).

ReLU SAEs are fastsae's ``VanillaSAE``. Importing this package installs
the load-time hook in :mod:`consdl.models._resize` (exact ``d_sae``,
``model.safetensors``); loading any checkpoint imports its SAE class,
which triggers this init, so every load path is covered.
"""

from consdl.models import _resize  # noqa: F401
