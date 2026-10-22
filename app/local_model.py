"""Serialized in-process MLX generation, without remote fallback."""
import hashlib
import json
import os
from pathlib import Path
import threading
from .config import ROOT, settings
from .local_resource import model_resource

_lock = threading.RLock()
_loaded = None
_outlines_model = None


def model_path():
    path = Path(settings().get('LOCAL_MODEL_PATH', 'models/qwen3-4b-instruct-2507-4bit')).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def available():
    path = model_path()
    return path.is_dir() and all((path / n).is_file() for n in ('config.json', 'tokenizer.json', 'local_manifest.json')) and any(path.glob('*.safetensors'))


def adapter_path():
    value = settings().get('LOCAL_ADAPTER_PATH', '').strip()
    if not value: return None
    path = Path(value).expanduser()
    return str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def verify_model_files():
    if not available(): raise ValueError('本地模型尚未下载并校验，请先运行 scripts/download_local_model.py。')
    directory = model_path()
    manifest = json.loads((directory / 'local_manifest.json').read_text())
    for name, meta in manifest['files'].items():
        if Path(name).name != name: raise ValueError('本地模型清单路径无效。')
        path = directory / name; h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
        if path.stat().st_size != meta['bytes'] or h.hexdigest() != meta['sha256']:
            raise ValueError('本地模型文件校验失败。')
    return manifest


def _load():
    global _loaded
    adapter = adapter_path()
    weights = Path(adapter) / 'adapters.safetensors' if adapter else None
    signature = (str(model_path()), adapter,
                 (weights.stat().st_mtime_ns, weights.stat().st_size) if weights and weights.exists() else None)
    if _loaded and _loaded[0] == signature: return _loaded[1:]
    verify_model_files()
    directory = model_path()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['OUTLINES_CACHE_DIR'] = str(ROOT / 'runtime/outlines-cache')
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx
    except (ImportError, RuntimeError) as exc:
        raise ValueError('MLX环境或Metal GPU不可用，请在Mac终端启动离线服务。') from exc
    mx.set_memory_limit(17 * 1024**3)
    model, tokenizer = load(str(directory), adapter_path=signature[1], lazy=False)
    _loaded = (signature, model, tokenizer, generate, make_sampler, mx)
    return _loaded[1:]


def complete(messages, max_tokens=4096, schema=None):
    global _outlines_model
    with _lock, model_resource('inference'):
        model, tokenizer, generate, make_sampler, mx = _load()
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(prompt))
        limit = int(settings().get('LOCAL_CONTEXT_TOKENS', '16384'))
        if count + max_tokens > limit:
            raise ValueError(f'材料超出本地上下文上限（{limit} tokens），请拆分材料；未截断证据。')
        if schema is not None:
            import outlines
            from outlines.inputs import Chat
            if _outlines_model is None or _outlines_model[0] is not model:
                _outlines_model = (model, outlines.from_mlxlm(model, tokenizer))
            text = _outlines_model[1](Chat(messages), output_type=schema,
                max_tokens=max_tokens, sampler=make_sampler(temp=0.0), verbose=False)
        else:
            text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                            sampler=make_sampler(temp=0.0), verbose=False)
        mx.clear_cache()
        return text


def complete_batch(requests, max_tokens=3000, progress=None):
    """Independent constrained processors per prompt, using MLX BatchGenerator."""
    with _lock, model_resource('annotation_batch'):
        model, tokenizer, _, make_sampler, mx = _load()
        import outlines
        from mlx_lm.generate import BatchGenerator
        wrapped = outlines.from_mlxlm(model, tokenizer)
        prompts = []; processors = []
        limit = int(settings().get('LOCAL_CONTEXT_TOKENS', '16384'))
        for index,(messages, schema) in enumerate(requests):
            if progress: progress({'stage':'compiling_constraints','item':index+1,'items':len(requests)})
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if len(prompt) + max_tokens > limit: raise ValueError('Batch context limit exceeded; nothing truncated')
            prompts.append(prompt)
            generator = outlines.Generator(wrapped, schema)
            generator.logits_processor.reset()
            processors.append([generator.logits_processor])
        generator = BatchGenerator(model, sampler=make_sampler(temp=0.0), stop_tokens=tokenizer.eos_token_ids,
            prefill_batch_size=1, completion_batch_size=len(prompts), prefill_step_size=512,
            prompt_progress_callback=(lambda values: progress({'stage':'prefill','progress':values})) if progress else None)
        ids = generator.insert(prompts, max_tokens=max_tokens, logits_processors=processors)
        tokens = {uid: [] for uid in ids}
        try:
            while responses := generator.next():
                for result in responses:
                    if result.finish_reason != 'stop': tokens[result.uid].append(result.token)
                if progress: progress({'stage':'generating','generated_tokens':sum(map(len,tokens.values())),
                    'active_memory_gb':round(mx.get_active_memory()/1e9,3)})
            stats = vars(generator.stats())
        finally:
            generator.close()
        texts = [tokenizer.decode(tokens[uid]) for uid in ids]
        mx.clear_cache()
        return texts, stats
