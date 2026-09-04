# -*- coding: utf-8 -*-
"""
CV-Agent 前端（Streamlit）· 刘城简历智能问答

运行：.venv/Scripts/python -m streamlit run app.py
模式：
  - 演示模式（默认）：内置基于真实简历的示例问答 + 引用展示 + 边界兜底话术演示，无需联网/API
  - 实时模式：M1 检索链路接入后启用（LangGraph/FastAPI + DeepSeek API）

演示模式文案为「示例口径」，正式内容以 data/kb/ 校对后的 L0/L1 为准。
"""
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
KB = ROOT / "data" / "kb"

st.set_page_config(page_title="刘城 · CV-Agent 简历智能问答", page_icon="📄", layout="centered")
st.markdown(
    """
<style>
    .block-container {padding-top: 1.2rem;}
    .stApp {background: #fafbfc;}
    div[data-testid="stChatMessage"] {border-radius: 12px;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- 演示知识（示例口径）
DEMO_QUESTIONS = [
    "简单介绍下刘城",
    "设备运维 RAG 系统的检索链路怎么设计的？",
    "质量分析报告助手解决什么问题？",
    "停机时长从 3000 降到 600 是怎么做到的？",
    "商品宣传图自动出图 Agent 你具体怎么做的？",
    "智能视频生成平台为什么要用多 Agent？",
    "他做过前端吗？",
    "为什么从 Dify 原型迁到自研？",
]

CITE_L0 = "《01_简历_L0_事实源.md》"
CITE_L1 = "《02_话术库_L1_合并版.md》"
CITE_L3 = "《03_技能边界表_待确认.md》"

DEMO = {
    DEMO_QUESTIONS[0]: dict(
        answer=(
            "刘城，大模型应用工程师。重庆航天职业技术学院汽车电子专业（2020.09 - 2024.06）毕业，"
            "2024.10 起任职东尼电子股份有限公司大模型应用工程师，至今近 2 年，专注 RAG 与 Agent 方向，"
            "能独立完成需求分析 → 方案设计 → 技术交付的完整闭环。\n\n"
            "主导 / 参与 3 个项目：\n"
            "① 工业设备智能运维 RAG 系统（2025.08 - 2026.06，自研，从 Dify 原型升级而来）：2 万+ 页工业文档知识化，"
            "混合检索 + 证据校验 + 边界兜底，Recall@5 72%→89%，试点产线月均停机时长降 80%；\n"
            "② 质量分析报告助手（2024.12 - 2025.09）：RAG+Agent 质量问题诊断与报告自动生成，根因定位 4 小时→30 分钟内；\n"
            "③ 设备运维助手 Dify 原型（2024.11 - 2025.01）：低代码快速验证 RAG 业务可行性。\n\n"
            "技术栈：LangChain/LangGraph、BM25 + BGE-M3 混合检索、BGE-Reranker 精排、Qwen2.5 微调（LoRA/QLoRA）、"
            "INT8/FP16 量化、FastAPI/Docker/Redis 工程化。\n"
            "（以上为简历内容概述，可继续追问任一项目细节，回答均带引用来源）"
        ),
        cites=[(CITE_L0, "基本信息 / 工作经历 / 项目经历"), (CITE_L1, "第一部分 · 自我介绍与开场白")],
    ),
    DEMO_QUESTIONS[1]: dict(
        answer=(
            "整个链路按 LangGraph 编排为：**问题理解 → 查询路由 → 混合检索 → 精排 → 证据校验 → 生成/边界兜底**。\n\n"
            "1. **问题理解**：意图识别判断是查故障码、查设备型号，还是描述故障现象求解；\n"
            "2. **查询路由**：精确故障码走 BM25 关键词检索；模糊现象描述走 BGE-M3 稠密向量语义检索；两者兼备走 BM25+稠密/稀疏双向量三路混合召回；\n"
            "3. **融合精排**：多路召回结果 RRF 融合去重后，交给 BGE-Reranker-Large 按相关性二次打分排序，取 Top-K；\n"
            "4. **证据校验**：先做关键词/正则硬性字段校验（故障码、设备型号、维修措施），再做交叉编码器软打分——"
            "置信度 >85% 正常生成；60%~85% 生成但末尾追加“检索资料有限，建议确认”；<60% 不硬答，走边界兜底；\n"
            "5. **生成**：证据充分的上下文交给 Qwen2.5-14B 生成排查建议，**输出带引用来源**（哪份文档哪一节），可溯源。\n\n"
            "工程侧：高频查询走 Redis 缓存，FastAPI 封装服务，Docker Compose 部署，Prometheus + Grafana 监控，"
            "平均响应 < 10s，试点期可用性 95.5%+。（对应用户方案里的自我介绍口径）"
        ),
        cites=[(CITE_L1, "§1.1 RAG 的整体流程 / 链路"), (CITE_L1, "§1.16 你的检索链路怎么设计的"), (CITE_L0, "项目一 · 职责 2/3")],
    ),
    DEMO_QUESTIONS[2]: dict(
        answer=(
            "面向制造业质量分析场景——质量工程师处理产线异常时要同时查 MES、QMS、PLC 日志、SPC 数据，再翻历史 8D/FMEA 报告，"
            "痛点：数据来源分散、缺陷描述不统一、历史案例难复用、报告编写周期长。\n\n"
            "系统做的事（RAG + Agent）：\n"
            "1. **缺陷标准化**：定义 QualityIssueRecord 15+ 核心字段，把“表面有划痕”“外观不良”统一归类为“外观缺陷-划伤”，消除统计失真；\n"
            "2. **双路信息汇聚**：一路检索历史相似案例（8D/FMEA，BGE-M3 混合检索）；一路实时调 MES/QMS/SPC 拿当前批次工艺参数与质量记录；\n"
            "3. **根因分析与报告生成**：Qwen2.5-14B 综合两路信息给根因分析建议，再按工厂 8D/质量月报模板自动填充章节，工程师只需审核修改；\n"
            "4. **AI 审核提示**：对比历史同类报告，提醒“是否漏了某种异常因素”。\n\n"
            "效果：案例召回 Recall@5 68%→87%，缺陷归类准确率 85%+；根因定位 4 小时→30 分钟内；8D 报告 5 天→2 天、质量月报 5 天→半天；"
            "重复质量问题占比 35%→15%；2 条试点产线运行 3 个月，累计生成报告 15+ 份。"
        ),
        cites=[(CITE_L1, "第二部分 · 质量分析报告助手（逐节点口述）"), (CITE_L0, "项目二 · 质量分析报告助手")],
    ),
    DEMO_QUESTIONS[3]: dict(
        answer=(
            "口径先说清楚：这是**业务统计口径**——3 条试点产线稳定运行 3 个月，产线**月均**故障停机时长从 3000 分钟降至 600 分钟，降幅 80%。\n\n"
            "支撑这个结果的三层动作：\n"
            "1. **知识底子**：2 万+ 页设备手册/故障码表/维修记录结构化入库（PDF/扫描件/Word/Excel 5 种格式），MinerU+Qwen2.5-14B 抽取，核心字段准确率 85%+；\n"
            "2. **检索准**：200+ 条场景化测试集驱动调优，BM25+BGE-M3 混合检索 + 精排，Recall@5 从 72%→89%，故障码精确匹配率 91%+——维修人员问得准、答得准，不用再人工翻一堆资料；\n"
            "3. **工程稳**：证据校验 + 低置信度边界兜底防误导，Redis 缓存高频查询，服务平均响应 <10s、可用性 95.5%+，一线才愿意天天用。"
        ),
        cites=[(CITE_L0, "项目一 · 业务成果 / 技术成果"), (CITE_L1, "§1.15 召回率低怎么提升 / 你做过哪些优化")],
    ),
    DEMO_QUESTIONS[4]: dict(
        answer=(
            "我把商品宣传图自动出图拆成 6 个节点：输入预处理、卖点提炼、创意策划、画面生成、文案排版、质检自愈。\n\n"
            "具体做法是：先把商品名称、品类、卖点、价格、目标人群、平台尺寸结构化；再用 LLM 输出核心卖点、适用场景、主标题候选等 JSON；"
            "然后按类目、节日、品牌调性从模板库/案例库检索相似风格，生成构图方案。画面层先做商品主体分割和抠图，再生成或检索背景素材，"
            "最后做商品与背景融合。排版层根据平台安全区把标题、卖点标签和价格渲染到图上。\n\n"
            "最后由质检节点检查广告合规、文案事实一致性、视觉质量、尺寸规范；不通过时按错误类型回到对应节点重做，比如文案违规回文案节点，"
            "排版越界回排版节点，主体融合不自然回融合节点。这样它不是单纯调文生图模型，而是一个可控、可返工、可交付的电商出图工作流。"
        ),
        cites=[("《05_AIGC_商品宣传图自动出图Agent_面试问答.md》", "面试官：你具体是怎么做的？")],
    ),
    DEMO_QUESTIONS[5]: dict(
        answer=(
            "因为视频生成链路太长，一个 Agent 全部做容易出现三个问题：上下文太大、职责不清、无法并行。\n\n"
            "我把它设计成“主编排 Agent + 专业执行 Agent”。主编排 Agent 类似导演，负责任务规划、预算控制、状态调度和冲突仲裁；"
            "编剧 Agent 生成结构化剧本，分镜 Agent 输出分镜 JSON，美术 Agent 生成画面素材，配音 Agent 生成旁白/BGM，剪辑 Agent 用 FFmpeg 合成，"
            "质检 Agent 检查音画同步、角色一致性、事实一致性和安全合规。\n\n"
            "这样每个节点输入输出清楚，画面生成和配音可以并行，失败时也能定位责任节点并定向返工，而不是整条视频链路重跑。"
        ),
        cites=[("《06_AIGC_智能视频生成平台_面试问答.md》", "面试官：为什么要用多 Agent，而不是一个 Agent 全部做完？")],
    ),
    DEMO_QUESTIONS[6]: dict(
        answer=(
            "现有简历材料没有把前端作为刘城的核心经历展开。\n\n"
            "从已整理的项目材料看，刘城在这些项目里的重点职责是 **RAG 检索链路、Agent 编排、模型调用与后端工程化**；"
            "界面相关内容更适合按协作或模板支持来表述。若需要确认具体参与深度，建议直接联系刘城本人补充。"
        ),
        cites=[(CITE_L3, "技能边界表 · 前端相关边界")],
    ),
    DEMO_QUESTIONS[7]: dict(
        answer=(
            "背景：2024.11 前后，新工厂建设导致资深设备工程师抽离、新运维人员占比上升，一线依赖老师傅经验、翻资料慢。"
            "当时先用 **Dify 快速搭原型**验证“大模型 + RAG 做设备手册问答”的可行性，两周左右打通了 自然语言提问→召回→生成 的链路，"
            "还完成了企业微信集成验证。\n\n"
            "为什么迁自研：原型跑通后，把试用中记录的问题清单化——Dify 在**故障码精确匹配、复杂检索路由、证据校验、低置信度兜底、监控与部署可控性**"
            "五个方面无法满足工业场景要求（工业容错比炫技重要）。于是 2025.08 起基于 LangGraph + FastAPI 自研，"
            "把混合检索、证据校验、边界兜底、Redis 缓存、监控全链路做成可控的。\n\n"
            "这段经历的价值：**先用低代码验证业务，再为工程边界自研**——原型阶段的技术方案、知识库、Prompt 资产全部复用，没走弯路。"
        ),
        cites=[(CITE_L0, "项目一 · 项目背景 / 项目三 · 成果"), (CITE_L1, "第四部分 · 原型验证到自研升级流程图")],
    ),
}

FALLBACK = (
    "这个问题不在当前简历知识库的重点覆盖范围内。结合现有资料，刘城的核心经历主要集中在 RAG 检索链路、Agent 编排、"
    "模型微调与后端工程化。\n\n"
    "如果需要确认这个问题的准确细节，建议直接联系刘城本人补充。你也可以换个问法，围绕项目经历、RAG 链路、Agent 编排、指标结果继续追问。"
)


# ---------------------------------------------------------------- 检索响应（演示/实时 可切换）
def respond(question: str) -> tuple[str, list]:
    """返回 (回答文本, 引用列表)。M1 接入后：本函数换成 LangGraph 检索链路调用即可。"""
    for q in DEMO_QUESTIONS:
        if question.strip() == q or q in question or question in q:
            d = DEMO[q]
            return d["answer"], d["cites"]
    return FALLBACK, []


# ---------------------------------------------------------------- 页面
st.title("📄 刘城 · CV-Agent")
st.caption("把简历做成知识库 —— 混合检索 / 证据校验 / 边界兜底，全部来自本人的生产方法论")

with st.sidebar:
    st.subheader("⚙️ 模式")
    mode = st.radio("问答模式", ["演示模式（内置示例，无需联网）", "实时模式（M1 检索链路接入后启用）"], label_visibility="collapsed")
    if mode.startswith("实时"):
        st.info("⏳ 实时模式待 M1：LangGraph 检索链路 + DeepSeek API 接入后启用。演示模式可先行体验交互与回答形态。")
    st.divider()
    st.subheader("🗂️ 知识库")
    for f in sorted(KB.glob("*.md")):
        n = len(f.read_text(encoding="utf-8"))
        st.markdown(f"- {f.name.split('_', 1)[-1]} · {n//1000}KB")
    st.caption("⚠️ 当前为演示口径，L0 事实校对中（见 04_校对清单）")
    st.divider()
    st.caption("联系方式\n电话 15330535227 · liucheng1912@gmail.com")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 历史消息
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("cites"):
            with st.expander("📎 引用来源"):
                for c in m["cites"]:
                    st.markdown(f"- {c[0]} ｜ {c[1]}")
            st.caption("（以上来源链接为演示标注；实时模式将提供原文片段可点开）")

# 示例 chips
st.markdown("**试试这样问：**")
chosen = st.pills("示例问题", options=DEMO_QUESTIONS, label_visibility="collapsed", selection_mode="single")
if chosen and chosen != st.session_state.get("_last_chip"):
    st.session_state["_last_chip"] = chosen
    st.session_state.messages.append({"role": "user", "content": chosen, "cites": None})
    answer, cites = respond(chosen)
    st.session_state.messages.append({"role": "assistant", "content": answer, "cites": cites})
    st.rerun()

# 输入框
if prompt := st.chat_input("输入你的问题，例如：他做过哪几个项目？"):
    st.session_state.messages.append({"role": "user", "content": prompt, "cites": None})
    answer, cites = respond(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer, "cites": cites})
    st.rerun()

# 首次进入给个开场引导
if len(st.session_state.messages) == 0:
    st.info(
        "我是刘城问答 Agent。每次回答尽量带真实引用来源；材料覆盖不足的问题，我会给出边界口径，"
        "并建议联系刘城确认细节。试试上方示例问题，或随便提问。"
    )
else:
    st.button(
        "🧹 清空对话",
        on_click=lambda: (st.session_state.pop("messages", None), st.session_state.pop("_last_chip", None)),
    )
