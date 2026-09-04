# -*- coding: utf-8 -*-
"""CV-Agent 配置：解析项目根目录 .env（结构对齐 EduAgent 模板）"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class Config:
    # ---- 应用 ----
    app_debug = _env("APP_DEBUG", "true").lower() == "true"
    app_port = int(_env("PORT", _env("APP_PORT", "8000")))
    log_level = _env("LOG_LEVEL", "INFO")

    # ---- DeepSeek（用户提供，直连 API）----
    deepseek_api_key = _env("DEEPSEEK_API_KEY")
    deepseek_base_url = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_chat = _env("DEEPSEEK_MODEL_CHAT", "deepseek-chat")

    # ---- 检索与证据校验 ----
    retrieve_top_k = int(_env("RETRIEVE_TOP_K", "8"))
    max_block_chars = 900          # 单块送入 LLM 的截断长度
    max_ctx_blocks = 5             # 实际送入 LLM 的证据块数
    ctx_total_chars = 6000         # 证据总长度上限
    conf_high = float(_env("CONF_HIGH", "0.85"))
    conf_mid = float(_env("CONF_MID", "0.60"))
    fallback_msg = _env(
        "CONF_LOW_FALLBACK",
        "这个问题不在当前简历知识库的重点覆盖范围内。结合现有资料，刘城的核心经历主要集中在 RAG 检索链路、Agent 编排、模型微调与后端工程化；如果需要确认该问题的准确细节，建议直接联系刘城本人补充",
    )

    # ---- 知识库 ----
    kb_dir = ROOT / "data" / "kb"
    l0_weight = 1.3                # L0(简历) 权重高于 L1(话术)，口径冲突时以 L0 为准的检索侧实现
    # 演示模式匹配阈值（MVP 启发式；M3 用测试集校准）
    hit_strong = 0.12              # 召回率 >= 此值：正常生成
    hit_weak = 0.05                # 之间：生成但附"依据有限"提示
    # 简历未正面展开的高风险主题，即使命中也走边界兜底
    deny_intents = [
        "前端", "vue", "react", "小程序", "java", "c++", "go语言", "android", "ios", "flutter",
        "离职", "离开现公司", "离开东尼", "为什么离开", "期望薪资", "薪资", "到岗", "期望城市", "学历提升", "提升计划",
    ]

    @property
    def kb_files(self):
        return sorted(self.kb_dir.glob("*.md"))


CFG = Config()
