"""pyhotpants 统一入口：通过 backend 参数路由到不同计算引擎"""
import importlib

_BACKENDS = {
    'numba':      '.numba',
    'float32':    '.float32',
    'metal':      '.metal',
    'cython':     '.cython',
    'cuda64':     '.cuda64',
    'agx_orin':   '.agx_orin_cuda',
}

_cache = {}

def hotpants(*, backend='numba', **kwargs):
    if backend not in _cache:
        mod_name = _BACKENDS.get(backend)
        if mod_name is None:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available: {', '.join(_BACKENDS.keys())}"
            )
        try:
            mod = importlib.import_module(mod_name, package=__package__)
            _cache[backend] = mod.hotpants
        except ImportError as e:
            raise ImportError(
                f"Failed to load backend '{backend}': {e}"
            ) from e
    return _cache[backend](**kwargs)
