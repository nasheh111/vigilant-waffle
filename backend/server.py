# -*- coding: utf-8 -*-
r"""
CV-Agent 服务入口（FastAPI + SSE）
启动：D:\python\envs\edu_agent\python.exe -m backend.server
访问：http://localhost:8000
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

try:
    from .chat import chat_stream, route
    from .kb import CHUNKS
except ImportError:  # 兼容在 backend 目录内直接运行脚本
    from chat import chat_stream, route
    from kb import CHUNKS

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="CV-Agent 简历智能问答", version="0.1.0")


class ChatBody(BaseModel):
    question: str
    history: list[dict] = []


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "chat.html")


@app.get("/api/health")
def health():
    return {"ok": True, "kb_chunks": len(CHUNKS)}


@app.post("/api/chat")
async def chat(body: ChatBody):
    question = body.question.strip()
    if not question:
        return StreamingResponse(iter([_ev({"type": "error", "text": "问题为空。"})]), media_type="text/event-stream")

    async def gen():
        async for item in chat_stream(question, body.history or []):
            yield _ev(item)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _ev(item: dict) -> str:
    return f"data: {json.dumps(item, ensure_ascii=False)}\n\n"


if __name__ == "__main__":
    import uvicorn
    try:
        from .config import CFG
    except ImportError:
        from config import CFG

    uvicorn.run(app, host="0.0.0.0", port=CFG.app_port, log_level=CFG.log_level.lower())
