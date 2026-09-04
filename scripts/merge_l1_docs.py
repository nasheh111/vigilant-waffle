# -*- coding: utf-8 -*-
"""
M0: 将 raw 下的话术原始文档合并为 L1 知识库文档（原文不动，只做章节包装与目录索引）。
用法: python scripts/merge_l1_docs.py
输出: data/kb/02_话术库_L1_合并版.md
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "kb" / "02_话术库_L1_合并版.md"

PART1_SRC = RAW / "话术_自我介绍与项目详解.md"          # 自我介绍 + 项目逐节点口述 + MES/QMS 关系
MAIN_SRC = RAW / "话术_RAG+Agent模块面试题.md"            # 主文件：RAG 模块(1.x) + Agent 模块(2.x)
FLOW_SRC = RAW / "项目流程图梳理与面试讲法.md"             # 3 项目 mermaid 流程图 + 面试讲法

def split_main(text: str):
    """主文件按顶层标题切成 RAG 模块 / Agent 模块 两段，标题行归各自段落。"""
    marker = "\n# Agent 模块面试题"
    idx = text.find(marker)
    if idx == -1:
        return text, ""  # 找不到标记就整段归 RAG
    head = text[: idx + 1]          # 保留 RAG 段自己的 # 标题行
    tail = text[idx + 1 :]          # Agent 段含其 # 标题行
    return head, tail

def dedent(s: str) -> str:
    """去掉首尾多余空行，保留正文。"""
    return s.strip("\n")

def main():
    p1 = dedent((PART1_SRC).read_text(encoding="utf-8"))
    main_txt = dedent((MAIN_SRC).read_text(encoding="utf-8"))
    rag_mod, agent_mod = split_main(main_txt)
    flow = dedent((FLOW_SRC).read_text(encoding="utf-8"))

    header = """# 话术库 L1（合并版）— 刘城求职口述话术

> - 本文件 = CV-Agent 知识库 L1 层（口述展开答案），内容来自以下原始文档的**逐字合并**（未改写）：
>   1. `话术_自我介绍与项目详解.md`（自我介绍 + 设备运维 RAG / 质量分析报告助手逐节点口述 + MES/QMS/PLC 关系 + 优化与兜底问答）
>   2. `话术_RAG+Agent模块面试题.md` 的 RAG 模块（1.1~1.31，60+ 题含 ★ 难度）
>   3. 同上文档的 Agent 模块（2.1~2.27）
>   4. `项目流程图梳理与面试讲法.md`（3 项目 Mermaid 流程图 + 面试讲法 + 简历可压缩表达）
> - 与 `01_简历_L0_事实源.md` 冲突处以 L0 为准（指标口径差异见 `04_校对清单.md`）
> - 重复内容不在此处去重：M1 入库阶段按 chunk 内容 hash 去重，保证同语料双检索召回率
> - **重建方法**：`python scripts/merge_l1_docs.py`（任何原始文档改动后重跑即可）
"""

    parts = [
        header,
        "",
        "# 第一部分 自我介绍与项目逐节点口述（源自：话术_自我介绍与项目详解.md）",
        "",
        p1,
        "",
        "# 第二部分 RAG 模块面试题（源自：话术_RAG+Agent模块面试题.md 第 1.x 节）",
        "",
        rag_mod,
        "",
        "# 第三部分 Agent 模块面试题（源自：话术_RAG+Agent模块面试题.md 第 2.x 节）",
        "",
        agent_mod,
        "",
        "# 第四部分 项目流程图梳理与面试讲法（源自：项目流程图梳理与面试讲法.md）",
        "",
        flow,
        "",
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"OK -> {OUT}  ({len(''.join(parts))} chars)")

if __name__ == "__main__":
    main()
