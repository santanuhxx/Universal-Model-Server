import asyncio
import time
from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class PipelineStage:
    name: str
    handler: Callable            # async function
    depends_on: list[str] = field(default_factory=list)


class PipelineDAG: 
    def __init__(self, name: str):
        self.name = name
        self._stages: dict[str, PipelineStage] = {}

    def add_stage(
        self,
        name: str,
        handler: Callable,
        depends_on: list[str] | None = None,
    ) -> "PipelineDAG":
        self._stages[name] = PipelineStage(
            name=name,
            handler=handler,
            depends_on=depends_on or [],
        )
        return self  

    async def execute(self, initial_inputs: dict[str, Any]) -> dict[str, Any]:   
        results: dict[str, Any] = {"__input__": initial_inputs}
        completed: set[str] = set()
        start = time.perf_counter()

        while len(completed) < len(self._stages):
            ready = [
                stage
                for name, stage in self._stages.items()
                if name not in completed
                and all(dep in completed for dep in stage.depends_on)
            ]

            if not ready:
                raise RuntimeError(
                    f"Pipeline '{self.name}' deadlock! "
                    f"Completed: {completed}, "
                    f"Remaining: {set(self._stages) - completed}"
                )

            tasks = [
                self._run_stage(stage, results)
                for stage in ready
            ]
            stage_results = await asyncio.gather(*tasks)

            for stage, result in zip(ready, stage_results):
                results[stage.name] = result
                completed.add(stage.name)

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        results["__pipeline_ms__"] = elapsed
        return results

    async def _run_stage(
        self,
        stage: PipelineStage,
        all_results: dict[str, Any],
    ) -> Any:
        if not stage.depends_on:
            inputs = all_results["__input__"]
        else:
            inputs = {
                dep: all_results[dep]
                for dep in stage.depends_on
                if dep in all_results
            }
            if len(stage.depends_on) == 1:
                inputs = all_results[stage.depends_on[0]]

        return await stage.handler(inputs)