"""Host-only encoder stage comparison using the pinned model's block code."""
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess


def compare(a, model, deployed, cfg, whistle, pcm, quant):
    import jax
    import jax.numpy as jnp
    import numpy as np
    native = a.output / 'native-stages'
    native.mkdir()
    audio = a.output / 'stage-input.f32'
    pcm.tofile(audio)
    command = [str(a.stage_binary.resolve()), str(a.model.resolve()), str(audio.resolve()), str(native.resolve())]
    if not quant:
        command.append('--float')
    subprocess.run(command, check=True, timeout=180)
    def stem_method(m, wav, lengths):
        x, lengths = whistle.features(cfg, wav, lengths)
        return m.stem(x.astype(cfg.jax_dtype), lengths, quant)
    x, frames = model.apply({'params': deployed}, jnp.asarray(pcm[None, :]), jnp.asarray([len(pcm)]), method=stem_method)
    valid_frames = int(frames[0])
    results = []
    def record(name, expected, native_name=None):
        expected = np.asarray(expected)[0, :valid_frames]
        actual = np.fromfile(native / ((native_name or name) + '.f32'), np.float32).reshape(expected.shape)
        relative = float(np.linalg.norm(actual - expected) / max(float(np.linalg.norm(expected)), 1e-12))
        result = dict(stage=name, relative_l2_error=relative, maximum_absolute_error=float(np.max(np.abs(actual - expected))),
                      all_finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all()), within_tolerance=bool(relative < 2e-4))
        expected.tofile(a.output / (name + '-jax.f32'))
        results.append(result)
        print(json.dumps(result), flush=True)
    record('stem', x)
    feature_shape = whistle.features(cfg, jnp.asarray(pcm[None, :]), jnp.asarray([len(pcm)]))[0].shape
    native_features = np.fromfile(native / 'features.f32', np.float32).reshape(1, -1, cfg.mels)
    shared_features = jnp.pad(jnp.asarray(native_features), ((0, 0), (0, feature_shape[1] - native_features.shape[1]), (0, 0)))
    shared_stem, _ = model.apply({'params': deployed}, shared_features, jnp.asarray([native_features.shape[1]]),
                                method=lambda m, feat, lengths: m.stem(feat, lengths, quant))
    record('stem-native-features', shared_stem, 'stem')
    x = jnp.pad(x, ((0, 0), (0, x.shape[1] % 2), (0, 0)))
    seen = jnp.arange(x.shape[1]) < frames[:, None]
    mask = jnp.broadcast_to(seen[:, None, None, :], (1, 1, x.shape[1], x.shape[1]))
    kwargs = dict(mask=mask, valid=seen[..., None], rope=whistle.precompute_rope_freqs(cfg.qk_dim, x.shape[1], cfg.rope))
    lanes = cfg.enc_lanes
    x = jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], lanes, cfg.dim))
    body = whistle._LaneBody(partial(whistle.EncoderBlock, cfg, cfg.jax_dtype), cfg.jax_dtype)
    execute = jax.jit(lambda params, value, hc: body.apply({'params': {'block': params}}, value, (None, hc), quant, kwargs)[0])
    enc = deployed['encoder']
    for index in range(cfg.enc_layers):
        block = jax.tree.map(lambda value: value[index], enc['layers']['block'])
        hc = {name: enc['mhc_' + name][index] for name in ('phi_pre', 'phi_post', 'phi_res', 'a_pre', 'a_post', 'a_res', 'b_pre', 'b_post', 'b_res')}
        lane = jnp.eye(lanes)[index % lanes]
        hc.update(pre_off=8 * lane - 4, post_off=-4 * (1 - lane))
        x = execute(block, x, hc)
        record('lanes-' + str(index), x)
        if index == 0:
            shared_input = np.fromfile(native / 'stem.f32', np.float32).reshape(1, valid_frames, 1, cfg.dim)
            shared_input = np.repeat(shared_input, lanes, axis=2)
        else:
            shared_input = np.fromfile(native / ('lanes-' + str(index - 1) + '.f32'), np.float32).reshape(1, valid_frames, lanes, cfg.dim)
        shared_input = jnp.pad(jnp.asarray(shared_input), ((0, 0), (0, x.shape[1] - valid_frames), (0, 0), (0, 0)))
        isolated = execute(block, shared_input, hc)
        record('isolated-lanes-' + str(index), isolated, 'lanes-' + str(index))
    pooled = jnp.mean(x, axis=2)
    final = (1 + enc['final_norm']['scale']) * pooled / jnp.sqrt(jnp.mean(pooled ** 2, axis=-1, keepdims=True) + 1e-6)
    record('encoder', final)
    report = dict(schema=1, scope='Stage localization on first catalog clip; no decoder, training, speech or target acceptance',
                  native_binary_sha256=hashlib.sha256(a.stage_binary.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  input_pcm_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), stages=results,
                  all_passed=all(v['within_tolerance'] and v['all_finite'] for v in results), relative_tolerance=2e-4,
                  deployment_permitted=False)
    (a.output / 'stages.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_passed']:
        raise ValueError('Encoder stages differ; inspect the first divergent stage')
    (a.output / 'INCOMPLETE').unlink()
