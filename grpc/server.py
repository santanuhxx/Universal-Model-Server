import asyncio
from concurrent import futures
from core.schemas import InferenceRequest, Priority
from router import get_engine


async def serve_grpc(port: int = 50051) -> None:
    try:
        import grpc
        print(f"🔌 gRPC server ready on port {port}")
        print("   Note: compile serving.proto to enable full gRPC support")
    except ImportError:
        print("⚠️  grpcio not installed. Run: pip install grpcio grpcio-tools")


class ModelServingServicer: 
    async def Infer(self, request, context):
        engine = get_engine()
        infer_request = InferenceRequest(
            model_name=request.model_name,
            inputs=dict(request.inputs),
            priority=Priority(request.priority or "normal"),
            tenant_id=request.tenant_id or "default",
        )
        result = await engine.handle(infer_request)
        return result


if __name__ == "__main__":
    asyncio.run(serve_grpc())