"""Match shared C2 time weights without changing the frozen teacher or RNG state."""
import torch


POLICY = 'match_time_only_shared_projection_v1'


def build_matched(legacy_builder, module, decoder, posterior, runtime, config):
    """Use the existing builder; align only the shared C2 time projection.

    C0, C1 and time-only C2 are unchanged. For time-plus-change C2, construct
    the existing time-only model in an isolated CPU RNG context, and copy its
    time projection. The new delta module keeps its legacy initialization.
    """
    model = legacy_builder(module, decoder, posterior, runtime, config)
    if config['variant'] != 'C2' or not config['include_change']:
        return model
    if not config['include_time'] or next(model.parameters()).device.type != 'cpu':
        raise RuntimeError('Matched C2 construction requires the existing CPU time-enabled model')
    before = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[], enabled=True):
        # Seed CPU only. The enclosing context restores the caller's CPU RNG;
        # unlike torch.manual_seed this does not reseed an initialized GPU.
        torch.random.default_generator.manual_seed(config['model_seed'])
        reference = module.ContinuousPosterior(decoder.base, posterior, 'C2',
            include_time=True, include_change=False, checkpoint_prefixes=True)
        if (runtime.digest(reference.posterior) != runtime.digest(model.posterior)
                or runtime.digest(reference.encoder) != runtime.digest(model.encoder)
                or reference.time_projection.weight.shape != model.time_projection.weight.shape):
            raise RuntimeError('Shared Q-Former, encoder or time projection differs before matching')
        model.time_projection.load_state_dict(reference.time_projection.state_dict(), strict=True)
    if not torch.equal(before, torch.random.get_rng_state()):
        raise RuntimeError('Matching changed the caller CPU random state')
    return model
