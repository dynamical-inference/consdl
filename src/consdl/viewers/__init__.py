"""Interactive viewers for trained dictionaries.

* ``explorer``   one backbone's server: the latent explorer (``/explore``:
                 one input, its latents on every panel's UMAP) and the
                 cluster-consistency view (``/view``, ``umap_view``).
* ``hub``        one URL in front of several backbone servers.
* ``precompute`` decoder k-means and joint UMAP views; ``tops`` the top
                 inputs per latent. Both run through
                 ``python -m consdl.cli.viewer_precompute``.

Launch with ``python -m consdl.cli.viewer``.
"""
