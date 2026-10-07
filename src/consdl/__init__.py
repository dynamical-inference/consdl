"""ConSDL: consistent sparse dictionary learning.

Subpackages:

  * ``models``   -- SAE architectures (ConSDL = ``models.fista`` + refitting
    + co-active similarity regularization, and the six baselines)
  * ``training`` -- the trainer (with the co-active similarity term) and resume
  * ``data``     -- vision / LM backbones and data, the synthetic benchmark
  * ``stats``    -- per-latent firing statistics
  * ``metrics``  -- dictionary and code consistency metrics
  * ``io``       -- checkpoints, paths, Hugging Face Hub download
  * ``cli``      -- every entry point (``python -m consdl.cli.<name>``)

This init is import-light (no torch).
"""

__version__ = "0.1.0"
