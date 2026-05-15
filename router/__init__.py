from router.engine import RouterEngine

_engine: RouterEngine | None = None

def get_engine() -> RouterEngine:
    global _engine
    if _engine is None:
        _engine = RouterEngine()
    return _engine