"""
Reusable neural-network blocks, grouped by kind:

- ``convolution``: ConvNormAct, residual / ConvNeXt / MBConv blocks, LRN, SE.
- ``transformer``: attention, patch embedding, positional embeddings, TransformerBlock.
- ``readout``: pooling and MLP readouts.
- ``geometry``: box utilities used by the detection components.

Shared primitives that blocks build on (LayerNorm2d, StochasticDepth) live in
``optastra.nn.layers``.
"""
