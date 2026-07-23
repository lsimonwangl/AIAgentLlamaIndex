"""
Router RAG - 模型與連線 Client
=============================
clients.py 集中管理需要對外連線的 client 物件：
    - LLM（NVIDIA NIM，透過 OpenAI-compatible 端點）
    - Embedding Model（NVIDIA）
    - Milvus VectorStore（向量資料庫連線）

集中在此檔案的原因：API 端點、金鑰、URI 這類連線設定是最常需要調整的
部分，統一放在一起維護；indexes.py 的索引建構邏輯只需要拿到建好
的 client 物件使用，不需要知道連線細節。
"""

import os

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore

# ── LLM 連線設定 ──
LLM_API_BASE = "https://integrate.api.nvidia.com/v1"  # NVIDIA NIM 的 OpenAI 相容端點
LLM_CONTEXT_WINDOW = 128000
LLM_TIMEOUT = 300.0

# ── Milvus 連線設定 ──
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "travel_preferences"
EMBED_DIM = 2048  # 需與 embedding 模型輸出維度一致


def build_llm(model=None):
    """建立 LLM 實例（透過 OpenAI-compatible 端點呼叫 NVIDIA NIM）。

    model 不傳時用 .env 的 CHAT_MODEL；傳入時改用指定模型
    （例如 DocumentSummaryIndex 建摘要用的便宜快速模型）。
    """
    return OpenAILike(
        api_base=LLM_API_BASE,
        api_key=os.getenv("NVIDIA_API_KEY"),
        model=model or os.getenv("CHAT_MODEL"),
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=LLM_CONTEXT_WINDOW,
        timeout=LLM_TIMEOUT,
    )


def build_summary_llm():
    """建立 DocumentSummaryIndex 專用 LLM：讀 .env 的 SUMMARY_MODEL（便宜快速的模型）。

    DocumentSummaryIndex 建索引時每篇各打一次 LLM 生摘要，用小模型省時省額度；
    未設 SUMMARY_MODEL 時 fallback 回 CHAT_MODEL。
    """
    return build_llm(os.getenv("SUMMARY_MODEL"))


def build_embed_model():
    """建立 Embedding Model 實例。"""
    return NVIDIAEmbedding(model=os.getenv("EMBEDDING_MODEL"))


def build_milvus_vector_store():
    """建立 Milvus VectorStore 連線（供 VectorStoreIndex 使用）。"""
    return MilvusVectorStore(
        uri=MILVUS_URI,
        collection_name=MILVUS_COLLECTION,
        dim=EMBED_DIM,
        overwrite=True,  # 每次啟動覆寫，確保資料與 ./data 同步
    )
