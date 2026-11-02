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
    if prefetch_ahead < 1:
        raise ValueError(f"prefetch_ahead must be >= 1, got {prefetch_ahead}")

    # Ring-slot safety invariant. The engine protects a VRAM slot from
    # being overwritten while a compute kernel may still be reading it via
    # a per-slot CUDA `free_event` armed by `mark_block_done`. That event
    # is only recorded once a block's forward pass has returned. With a
    # ring of `S` streaming slots and `prefetch_ahead = P`, at most P
    # transfers are ever in flight ahead of the block currently computing,
    # so P + 1 distinct blocks (the one running + the P prefetched) must
    # each own a distinct slot at once. If S <= P, a prefetch wraps the
    # ring back onto a slot whose block has NOT yet been marked done, and
    # the engine would overwrite it mid-read -- silent corruption, not a
    # crash. Require strictly more streaming slots than the prefetch depth
    # and fail loudly here rather than produce wrong pixels.
    #
    # Resident blocks never occupy a ring slot, so they don't count toward
    # S; `vram_slots` passed to the engine is exactly the streaming ring
    # size. We can't read it back from the handle, so this check lives at
    # the call site where both numbers are known -- see generate_*.py,
    # which pass matching VRAM_SLOTS / PREFETCH_AHEAD constants.
    vram_slots = getattr(engine, "vram_slots", None)
    if callable(vram_slots):
        vram_slots = vram_slots()
    if vram_slots is not None and vram_slots <= prefetch_ahead:
        raise ValueError(
            f"vram_slots ({vram_slots}) must be > prefetch_ahead ({prefetch_ahead}): "
            f"a {vram_slots}-slot ring cannot hold the running block plus "
            f"{prefetch_ahead} prefetched blocks without overwriting one still in use"
        )

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
