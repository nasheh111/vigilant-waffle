# -*- coding: utf-8 -*-
"""
知识库加载 + 轻量检索（纯 Python，无外部向量依赖）
- 分块：按 Markdown 标题层级切块，每块 = {标题链, 文本, 文件, level(L0/L1/L3)}
- 打分：字符 bigram 词重叠 + idf + L0 加权；MVP 启发式，M3 用测试集校准
"""
import math
import re
from dataclasses import dataclass, field

try:
    from .config import CFG
except ImportError:  # 兼容在 backend 目录内直接运行脚本
    from config import CFG

_HEADING = re.compile(r"^(#{1,4})\s+(.+)$", re.M)

FILE_LEVEL = {"01": "L0", "02": "L1", "03": "L3", "04": "META"}


@dataclass
class Chunk:
    file: str          # 文件名（用于引用展示）
    level: str         # L0/L1/L3
    section: str       # 标题链，如 "项目一 > 个人职责"
    text: str
    _ngrams: set = field(default_factory=set, repr=False)


def _ngrams(text: str) -> set:
    """中文按字符 bigram，英文/数字按 token 小写。"""
    out = set()
    for block in re.findall(r"[一-鿿]+", text):
        for i in range(len(block) - 1):
            out.add(block[i : i + 2])
    normalized = text.lower().replace("-", " ").replace("/", " ")
    for token in re.findall(r"[a-z][a-z0-9_+#.]*|\d+(?:\.\d+)?%?", normalized):
        clean = token.strip("._+#")
        if len(clean) >= 2:
            out.add(clean)
    return out


def _section_path(headings: list[tuple[int, str]]) -> str:
    return " > ".join(h[1] for h in headings[-3:])


def load_kb() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in CFG.kb_files:
        fname = path.name
        prefix = fname.split("_", 1)[0]
        level = FILE_LEVEL.get(prefix, "L1")
        if level == "META":
            continue  # 校对清单等过程文档不进知识库
        # 引用来源展示完整文件名，避免知识库编号（如 008/009）和前端来源不一致。
        show_name = fname
        text = path.read_text(encoding="utf-8")
        # 按标题切块：每个标题(及紧随的正文)自成一块
        parts = []
        headings: list[tuple[int, str]] = []
        cur = []
        for line in text.splitlines():
            m = _HEADING.match(line)
            if m:
                if cur and headings:
                    parts.append((_section_path(headings), "\n".join(cur).strip()))
                # 标题作为新块起点
                if cur and not headings:
                    cur = [line]  # 前置文字（文件头）忽略
                else:
                    cur = []
                headings = [h for h in headings if h[0] < len(m.group(1))]
                headings.append((len(m.group(1)), m.group(2).strip()))
                cur.append(line)
            elif line.strip():
                cur.append(line)
            elif cur and headings:
                cur.append("")
        if cur and headings:
            parts.append((_section_path(headings), "\n".join(cur).strip()))
        for section, body in parts:
            body = body.strip()
            if len(body) < 8:
                continue
            chunks.append(Chunk(file=show_name, level=level,
                                section=section, text=body, _ngrams=_ngrams(body)))
    # idf
    n = max(1, len(chunks))
    docfreq: dict[str, int] = {}
    for c in chunks:
        for g in c._ngrams:
            docfreq[g] = docfreq.get(g, 0) + 1
    idf = {g: math.log((n + 1) / (df + 0.5)) for g, df in docfreq.items()}
    _IDF = idf
    return chunks, _IDF


CHUNKS, IDF = load_kb()
INVERTED: dict[str, list[int]] = {}
for i, c in enumerate(CHUNKS):
    for g in c._ngrams:
        INVERTED.setdefault(g, []).append(i)


def _query_ngrams(q: str) -> set:
    return _ngrams(q)


def retrieve(query: str, top_k: int | None = None) -> list[tuple[Chunk, float, float]]:
    """返回 [(chunk, 原始分, 召回率)]，召回率 = 得分 / 查询词最大可达分。"""
    qg = _query_ngrams(query)
    if not qg:
        return []
    # 命中候选（在任一查询 ngram 的倒排中）
    cand: dict[int, float] = {}
    for g in qg:
        w = IDF.get(g, 0.0)
        for i in INVERTED.get(g, []):
            cand[i] = cand.get(i, 0.0) + w
    q_max = sum(IDF.get(g, 0.0) for g in qg) or 1.0
    scored = []
    for i, s in cand.items():
        c = CHUNKS[i]
        w = 1.0 if c.level != "L0" else CFG.l0_weight
        # 精确短语命中额外加权（提升"指标数字/故障码"类硬匹配的表达力）
        exact = 0.0
        ql = query.lower()
        for pat in re.findall(r"[一-鿿]{2,}|\d+(?:\.\d+)?%?", ql):
            if pat in c.text.lower():
                exact += min(3.0, len(pat) / 4)
        scored.append((c, (s + exact) * w, s / q_max))
    scored.sort(key=lambda x: -x[1])
    return scored[: top_k or CFG.retrieve_top_k]


if __name__ == "__main__":  # 自检
    print(f"chunks={len(CHUNKS)}")
    for q in ["检索链路怎么设计", "停机时长降幅", "质量分析报告助手做了什么", "做过前端吗"]:
        print(f"\nQ: {q}")
        for c, s, r in retrieve(q, 3):
            print(f"  [{c.level} {c.file} §{c.section}] 分={s:.1f} 召回率={r:.2f} | {c.text[:40]}…")
