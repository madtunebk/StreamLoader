"""
streamloader_engine: Rust weight-streaming engine for DiT transformers
(FLUX.2-klein today), plus one small Python convenience function
(`attach_engine`) that wires it into a diffusers meta-model. Everything
else in this package is the compiled Rust extension.
"""

from .streamloader_engine import *

__doc__ = streamloader_engine.__doc__
if hasattr(streamloader_engine, "__all__"):
    __all__ = streamloader_engine.__all__


def attach_engine(transformer, engine, compute_stream_ptr, prefetch_ahead=1):
    """
    Wire `engine` into `transformer` (a diffusers DiT model built with
    accelerate.init_empty_weights(), i.e. all-meta weights). For every
    block id in engine.block_ids(), registers a forward pre-hook that
    streams that block's real weights in (load_state_dict(assign=True))
    right before it runs and prefetches `prefetch_ahead` blocks ahead,
    and a forward hook that marks the block done right after. Also loads
    the engine's shared/unassigned tensors and issues the first
    prefetch. Returns the hook handles.

    Block ids are matched to submodules by `family.index` -- e.g.
    "transformer_blocks.3" -> transformer.transformer_blocks[3] -- which
    is how diffusers names every nn.ModuleList of DiT blocks; this is
    not FLUX-specific.
    """
    block_ids = engine.block_ids()
    pos_of = {bid: i for i, bid in enumerate(block_ids)}

    def block_module(block_id):
        family, idx = block_id.rsplit(".", 1)
        return getattr(transformer, family)[int(idx)]

    def make_pre_hook(block_id):
        def hook(module, args, kwargs):
            tensors = engine.get_block(block_id, compute_stream_ptr)
            transformer.load_state_dict(tensors, strict=False, assign=True)
            pos = pos_of[block_id] + prefetch_ahead
            if pos < len(block_ids):
                engine.prefetch(block_ids[pos])
            return args, kwargs

        return hook

    def make_post_hook(block_id):
        def hook(module, args, output):
            engine.mark_block_done(block_id, compute_stream_ptr)
            return output

        return hook

    handles = []
    for bid in block_ids:
        mod = block_module(bid)
        handles.append(mod.register_forward_pre_hook(make_pre_hook(bid), with_kwargs=True))
        handles.append(mod.register_forward_hook(make_post_hook(bid)))

    transformer.load_state_dict(engine.get_shared(), strict=False, assign=True)
    engine.prefetch(block_ids[0])
    return handles
