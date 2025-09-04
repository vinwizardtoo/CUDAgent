from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from cuda_agent_core.llm import ChatClient, ChatMessage
except Exception as e:  # pragma: no cover - allow app to start without core installed
    ChatClient = None  # type: ignore
    ChatMessage = None  # type: ignore


class ChatMessageModel(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class PlanRequest(BaseModel):
    messages: List[ChatMessageModel]
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None


class ExecuteRequest(BaseModel):
    action: str = "run"
    kernel: Optional[str] = None
    config: Optional[Dict] = None


app = FastAPI(title="CUDAgent API", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def _sse_event(data: str) -> bytes:
    # SSE data frame; must end with double newline
    return f"data: {data}\n\n".encode("utf-8")


@app.post("/v1/coach/plan")
async def coach_plan(req: PlanRequest):
    if ChatClient is None or ChatMessage is None:
        raise HTTPException(status_code=500, detail="Core LLM client not available")

    # Initialize client (reads env OPENAI_API_KEY)
    try:
        client = ChatClient(model=req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM init error: {e}")

    async def streamer() -> AsyncGenerator[bytes, None]:
        # Yield a small preamble event
        yield _sse_event(json.dumps({"status": "starting"}))
        # Bridge sync generator to async streaming
        try:
            for chunk in client.stream_chat([ChatMessage(**m.model_dump()) for m in req.messages] ):
                if chunk.content_delta:
                    yield _sse_event(chunk.content_delta)
                await asyncio.sleep(0)  # cooperative yield
            yield _sse_event("[DONE]")
        except Exception as e:  # surface error to stream
            yield _sse_event(json.dumps({"error": str(e)}))

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.post("/v1/coach/execute")
async def coach_execute(req: ExecuteRequest):
    async def streamer() -> AsyncGenerator[bytes, None]:
        steps = [
            "Compile kernel...",
            "Validate vs PyTorch baseline...",
            "Benchmark: warming up...",
            "Benchmark: measuring...",
            "Summarize results...",
        ]
        for s in steps:
            yield _sse_event(s)
            await asyncio.sleep(0.4)
        result = {
            "latency_us": 12.34,
            "throughput_gbps": 89.0,
            "status": "ok",
        }
        yield _sse_event(json.dumps(result))
        yield _sse_event("[DONE]")

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.post("/v1/run")
async def run_kernel(payload: Dict) -> JSONResponse:
    # TODO: integrate with runtime package. For now, echo payload.
    return JSONResponse({"received": payload, "status": "stub"})

