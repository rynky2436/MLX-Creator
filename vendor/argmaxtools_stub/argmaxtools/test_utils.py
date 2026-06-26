class AppleSiliconContextMixin:           # no-op context helper
    def __enter__(self): return self
    def __exit__(self, *a): return False
class InferenceContextSpec:               # accepts/ignores any config
    def __init__(self, *a, **k): pass
    def spec_dict(self, *a, **k): return {}
