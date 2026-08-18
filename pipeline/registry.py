from pipeline.dag import PipelineDAG

_pipelines: dict[str, PipelineDAG] = {}


def register_pipeline(pipeline: PipelineDAG) -> None:
    _pipelines[pipeline.name] = pipeline
    print(f"🔗 Registered pipeline: '{pipeline.name}'")


def get_pipeline(name: str) -> PipelineDAG:
    if name not in _pipelines:
        raise KeyError(
            f"Pipeline '{name}' not found. "
            f"Available: {list(_pipelines.keys())}"
        )
    return _pipelines[name]


def list_pipelines() -> list[str]:
    return list(_pipelines.keys())