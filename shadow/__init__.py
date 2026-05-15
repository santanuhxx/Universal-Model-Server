from shadow.manager import ShadowManager, ShadowResult

_manager = ShadowManager(shadow_ratio=0.1)

def get_shadow_manager() -> ShadowManager:
    return _manager