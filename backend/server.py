# -*- coding: utf-8 -*-
r"""
CV-Agent 服务入口（FastAPI + SSE）
启动：D:\python\envs\edu_agent\python.exe -m backend.server
访问：http://localhost:8000
"""
import json
import hmac
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

try:
    from .config import CFG
    from .chat import chat_stream, route
    from .kb import CHUNKS
    from .question_store import count_questions, list_questions, save_answer, save_question
except ImportError:  # 兼容在 backend 目录内直接运行脚本
    from config import CFG
    from chat import chat_stream, route
    from kb import CHUNKS
    from question_store import count_questions, list_questions, save_answer, save_question

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
    question_id = None
    try:
        question_id = save_question(
            question,
            request.client.host if request.client else "",
            request.headers.get("user-agent", ""),
        )
    except Exception:
        pass

    async def gen():
        answer_parts = []
        try:
            async for item in chat_stream(question, body.history or []):
                if item.get("type") in {"token", "error"}:
                    answer_parts.append(str(item.get("text", "")))
                yield _ev(item)
        finally:
            try:
                save_answer(question_id, "".join(answer_parts))
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _ev(item: dict) -> str:
    return f"data: {json.dumps(item, ensure_ascii=False)}\n\n"


def _session_token() -> str:
    return hmac.new(CFG.admin_session_secret.encode(), b"admin", "sha256").hexdigest()


def _is_admin(request: Request) -> bool:
    return hmac.compare_digest(request.cookies.get("cv_admin", ""), _session_token())


def _has_admin_api_access(request: Request) -> bool:
    header_password = request.headers.get("x-admin-password", "")
    return _is_admin(request) or (
        bool(CFG.admin_password) and hmac.compare_digest(header_password, CFG.admin_password)
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    if not CFG.admin_password:
        return HTMLResponse(_admin_login("后台未配置 ADMIN_PASSWORD，请先在 Render 环境变量中设置。"), status_code=503)
    if not _is_admin(request):
        return HTMLResponse(_admin_login())

    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>刘城问答 Agent 后台</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f6f7fb;color:#111827;margin:0;padding:28px}}
.wrap{{max-width:1100px;margin:0 auto}} h1{{font-size:22px;margin:0 0 8px}} .bar{{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;gap:16px}}
a,button{{font:inherit}} button{{padding:8px 12px;border:1px solid #d1d5db;background:#fff;border-radius:8px;cursor:pointer}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e5e7eb}}
th,td{{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top;font-size:14px}}
th{{background:#f9fafb}} td:nth-child(3),td:nth-child(4){{white-space:pre-wrap;line-height:1.6}}
.muted{{color:#6b7280;font-size:13px;line-height:1.6}} .status{{font-size:12px;color:#16a34a;margin-top:4px}}
.empty{{text-align:center;color:#6b7280;padding:28px}} .pending{{color:#9ca3af}}
</style></head><body><div class="wrap">
<div class="bar"><div><h1>刘城问答 Agent 后台</h1><div class="muted">共 <span id="total">{count_questions()}</span> 条记录，时间为北京时间（Asia/Shanghai），最新提问置顶，问题和模型回复都会归档。</div><div class="status" id="status">正在读取最新记录...</div></div>
<form method="post" action="/admin/logout"><button>退出</button></form></div>
<table><thead><tr><th>ID</th><th>提问时间</th><th>用户提问</th><th>模型回复</th><th>回复时间</th><th>IP</th></tr></thead><tbody id="rows"><tr><td class="empty" colspan="6">正在加载...</td></tr></tbody></table>
<script>
const rowsEl = document.getElementById('rows');
const totalEl = document.getElementById('total');
const statusEl = document.getElementById('status');
function esc(s){{return String(s ?? '').replace(/[&<>"']/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function renderRows(items){{
  if(!items.length){{
    rowsEl.innerHTML = '<tr><td class="empty" colspan="6">还没有用户提问</td></tr>';
    return;
  }}
  rowsEl.innerHTML = items.map(r =>
    `<tr><td>${{esc(r.id)}}</td><td>${{esc(r.created_at)}}</td><td>${{esc(r.question)}}</td><td>${{r.answer ? esc(r.answer) : '<span class="pending">回复生成中...</span>'}}</td><td>${{esc(r.answered_at || '')}}</td><td>${{esc(r.ip)}}</td></tr>`
  ).join('');
}}
async function loadQuestions(){{
  try {{
    const r = await fetch('/admin/api/questions', {{cache:'no-store'}});
    if(!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    totalEl.textContent = data.total;
    renderRows(data.items || []);
    statusEl.textContent = '已自动更新：' + data.server_time + '，每 3 秒刷新一次';
  }} catch(e) {{
    statusEl.textContent = '自动更新失败，请确认仍处于登录状态';
  }}
}}
loadQuestions();
setInterval(loadQuestions, 3000);
</script>
</div></body></html>"""
    )


@app.get("/admin/api/questions")
def admin_questions(request: Request):
    if not CFG.admin_password:
        return Response(json.dumps({"ok": False, "error": "ADMIN_PASSWORD 未配置"}, ensure_ascii=False), media_type="application/json", status_code=503)
    if not _has_admin_api_access(request):
        return Response(json.dumps({"ok": False, "error": "未授权"}, ensure_ascii=False), media_type="application/json", status_code=401)
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return Response(
        json.dumps(
            {
                "ok": True,
                "timezone": "Asia/Shanghai",
                "server_time": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
                "total": count_questions(),
                "items": list_questions(),
            },
            ensure_ascii=False,
        ),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
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
