"""Cross-seed consistency metrics.

  * ``kernels``      -- Hungarian matching of decoder atoms (DCS)
  * ``dict_metrics`` -- dictionary metrics: DCS, #dead, #dup (from which
    the usable atom fraction follows)
  * ``code_metrics`` -- code metrics: MCC of matched atoms, input R^2
  * ``clustering``, ``activations``, ``quality``, ``latent_consistency``
    -- helpers used by the above

Import the submodules directly; this package root re-exports nothing.
"""
