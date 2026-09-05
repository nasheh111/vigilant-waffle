# -*- coding: utf-8 -*-
r"""
CV-Agent 服务入口（FastAPI + SSE）
启动：D:\python\envs\edu_agent\python.exe -m backend.server
访问：http://localhost:8000
"""
import json
import hmac
import time
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

try:
    from .config import CFG
    from .chat import chat_stream, route
    from .kb import CHUNKS
    from .question_store import count_questions, list_questions, save_question
except ImportError:  # 兼容在 backend 目录内直接运行脚本
    from config import CFG
    from chat import chat_stream, route
    from kb import CHUNKS
    from question_store import count_questions, list_questions, save_question

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="CV-Agent 简历智能问答", version="0.1.0")


class ChatBody(BaseModel):
    question: str
    history: list[dict] = []


class LoginBody(BaseModel):
    password: str


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "chat.html")


@app.get("/api/health")
def health():
    return {"ok": True, "kb_chunks": len(CHUNKS)}


@app.post("/api/chat")
async def chat(body: ChatBody, request: Request):
    question = body.question.strip()
    if not question:
        return StreamingResponse(iter([_ev({"type": "error", "text": "问题为空。"})]), media_type="text/event-stream")
    try:
        save_question(
            question,
            request.client.host if request.client else "",
            request.headers.get("user-agent", ""),
        )
    except Exception:
        pass

    async def gen():
        async for item in chat_stream(question, body.history or []):
            yield _ev(item)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _ev(item: dict) -> str:
    return f"data: {json.dumps(item, ensure_ascii=False)}\n\n"


def _session_token() -> str:
    return hmac.new(CFG.admin_session_secret.encode(), b"admin", "sha256").hexdigest()


def _is_admin(request: Request) -> bool:
    return hmac.compare_digest(request.cookies.get("cv_admin", ""), _session_token())


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    if not CFG.admin_password:
        return HTMLResponse(_admin_login("后台未配置 ADMIN_PASSWORD，请先在 Render 环境变量中设置。"), status_code=503)
    if not _is_admin(request):
        return HTMLResponse(_admin_login())

    rows = list_questions()
    items = "\n".join(
        f"<tr><td>{r['id']}</td><td>{_esc(r['created_at'])}</td><td>{_esc(r['question'])}</td><td>{_esc(r['ip'])}</td></tr>"
        for r in rows
    )
    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>刘城问答 Agent 后台</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f6f7fb;color:#111827;margin:0;padding:28px}}
.wrap{{max-width:1100px;margin:0 auto}} h1{{font-size:22px}} .bar{{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}}
a,button{{font:inherit}} table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e5e7eb}}
th,td{{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top;font-size:14px}}
th{{background:#f9fafb}} td:nth-child(3){{white-space:pre-wrap;line-height:1.6}}
.muted{{color:#6b7280;font-size:13px}}
</style></head><body><div class="wrap">
<div class="bar"><div><h1>刘城问答 Agent 后台</h1><div class="muted">共 {count_questions()} 条用户提问</div></div>
<form method="post" action="/admin/logout"><button>退出</button></form></div>
<table><thead><tr><th>ID</th><th>时间 UTC</th><th>用户提问</th><th>IP</th></tr></thead><tbody>{items}</tbody></table>
</div></body></html>"""
    )


@app.post("/admin/login")
async def admin_login(body: LoginBody):
    if CFG.admin_password and hmac.compare_digest(body.password, CFG.admin_password):
        resp = Response(json.dumps({"ok": True}), media_type="application/json")
        resp.set_cookie("cv_admin", _session_token(), httponly=True, samesite="lax", max_age=86400)
        return resp
    return Response(json.dumps({"ok": False, "error": "密码错误"}), media_type="application/json", status_code=401)


@app.post("/admin/logout")
def admin_logout():
    resp = RedirectResponse("/admin", status_code=303)
    resp.delete_cookie("cv_admin")
    return resp


def _admin_login(error: str = "") -> str:
    err = f"<div class='err'>{_esc(error)}</div>" if error else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>管理员登录</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f6f7fb;color:#111827;display:grid;place-items:center;min-height:100vh;margin:0}}
.box{{width:min(420px,calc(100vw - 32px));background:#fff;border:1px solid #e5e7eb;padding:24px;border-radius:10px}}
h1{{font-size:20px;margin:0 0 16px}} input{{width:100%;padding:11px;border:1px solid #d1d5db;border-radius:8px;font:inherit}}
button{{margin-top:12px;width:100%;padding:11px;border:0;border-radius:8px;background:#2563eb;color:#fff;font:inherit;cursor:pointer}}
.err{{background:#fef2f2;color:#b91c1c;padding:9px;border-radius:8px;margin-bottom:12px;font-size:14px}}
</style></head><body><div class="box"><h1>刘城问答 Agent 后台</h1>{err}
<input id="pwd" type="password" placeholder="输入管理员密码" autofocus>
<button onclick="login()">登录</button>
<script>
async function login(){{
  const password = document.getElementById('pwd').value;
  const r = await fetch('/admin/login', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{password}})}});
  if(r.ok) location.reload(); else alert('密码错误');
}}
document.getElementById('pwd').addEventListener('keydown', e=>{{if(e.key==='Enter') login();}});
</script></div></body></html>"""


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    import uvicorn
    try:
        from .config import CFG
    except ImportError:
        from config import CFG

    uvicorn.run(app, host="0.0.0.0", port=CFG.app_port, log_level=CFG.log_level.lower())
