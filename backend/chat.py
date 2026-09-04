# -*- coding: utf-8 -*-
"""
对话核心：证据检索 → 门槛路由 → DeepSeek 流式生成
链路映射自简历方法论：混合检索 → 证据校验(阈值) → 生成/兜底
"""
import asyncio
from typing import AsyncGenerator

from openai import AsyncOpenAI

try:
    from .config import CFG
    from .kb import Chunk, retrieve
except ImportError:  # 兼容在 backend 目录内直接运行脚本
    from config import CFG
    from kb import Chunk, retrieve

_client = AsyncOpenAI(api_key=CFG.deepseek_api_key, base_url=CFG.deepseek_base_url)

SYSTEM_PROMPT = """你是"刘城简历问答助手"，服务 HR/面试官/求职同行。你只能依据【证据】回答关于刘城（大模型应用工程师）的问题。

铁律：
1. 证据中没有的事实——尤其是数字、时间、公司名、技能——一律不得出现；不推测、不补写、不"举例"。
2. 证据同时含简历(L0)与口述话术(L1)且冲突时，采信 L0（简历原文）。
3. 涉及"是否掌握/是否做过某技能"：证据未正面支持时，不要说"不会"、"不知道"、"没做过"；改用边界表达："现有材料没有把这项作为核心经历展开，刘城的公开简历重点是……，准确参与深度建议联系刘城本人确认。"
4. 证据完全不相关或不足：输出一段简短边界兜底，不扩展编造，不直接拒答，不使用"不知道/不会"这类生硬措辞。
5. 回答用中文；称呼本人为"刘城"。HR 类问题分点简洁；技术类问题可按口述话术展开，允许口语化。寒暄可简短回应后引导提问。
6. 不要在回答里自己编造来源清单（服务端会附加真实引用）。"""

FALLBACK = CFG.fallback_msg + "。你也可以换个问法，围绕项目经历、RAG 链路、Agent 编排、指标结果继续追问。"

WEAK_NOTE = "\n\n（当前检索资料覆盖有限，细节口径建议联系刘城本人确认）"


def _build_context(top: list[tuple[Chunk, float, float]]) -> str:
    blocks, total = [], 0
    for c, _, _ in top:
        t = c.text[: CFG.max_block_chars]
        if total + len(t) > CFG.ctx_total_chars:
            break
        blocks.append(f"[来源: {c.file} §{c.section}]\n{t}")
        total += len(t)
        if len(blocks) >= CFG.max_ctx_blocks:
            break
    return "\n\n---\n\n".join(blocks)


def _denied(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in CFG.deny_intents)


def route(query: str):
    """返回 ('fallback'|'weak'|'strong', evidence_chunks, 理由)"""
    if _denied(query):
        return "fallback", [], "命中简历未重点展开的技能主题"
    top = [x for x in retrieve(query) if x[0].level != "META"]
    if not top:
        return "fallback", [], "无任何召回"
    best = top[0][2]
    if best < CFG.hit_weak:
        return "fallback", [], f"最佳召回率 {best:.2f} < 弱阈值 {CFG.hit_weak}"
    if best < CFG.hit_strong:
        return "weak", top[: CFG.max_ctx_blocks], f"召回率 {best:.2f} ∈ [弱, 强)"
    return "strong", top[: CFG.max_ctx_blocks], f"召回率 {best:.2f}"


async def chat_stream(question: str, history: list[dict]) -> AsyncGenerator[dict, None]:
    """yield: {"type": "token"|"fallback"|"weak"|"error"|"sources", ...}"""
    mode, evid, reason = route(question)
    if mode == "fallback":
        # 兜底模板直返（不调用 LLM），避免在低证据场景编造经历
        for ch in FALLBACK:
            yield {"type": "token", "text": ch}
            await asyncio.sleep(0)
        yield {"type": "sources", "items": [], "mode": "fallback", "reason": reason}
        return

    context = _build_context(evid)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for m in history[-4:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": f"【证据】\n{context}\n\n【问题】{question}"})

    try:
        stream = await _client.chat.completions.create(
            model=CFG.model_chat,
            messages=msgs,
            stream=True,
            temperature=0.3,
        )
        async for part in stream:
            delta = part.choices[0].delta.content if part.choices else None
            if delta:
                yield {"type": "token", "text": delta}
    except Exception as e:  # 网络/API 故障也要优雅降级，不裸奔
        yield {"type": "error", "text": f"服务暂时不可用（{type(e).__name__}），请稍后重试。"}
        return

    if mode == "weak":
        for ch in WEAK_NOTE:
            yield {"type": "token", "text": ch}
            await asyncio.sleep(0)
    yield {
        "type": "sources",
        "items": [{"file": c.file, "section": c.section} for c, _, _ in evid],
        "mode": mode,
        "reason": reason,
    }
