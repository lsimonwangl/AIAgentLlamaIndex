"""
Router RAG - 模型與連線 Client
=============================
rag_clients.py 集中管理需要對外連線的 client 物件：
    - LLM（NVIDIA NIM，透過 OpenAI-compatible 端點）
    - Embedding Model（NVIDIA）
    - Milvus VectorStore（向量資料庫連線）

集中在此檔案的原因：API 端點、金鑰、URI 這類連線設定是最常需要調整的
部分，統一放在一起維護；rag_indexes.py 的索引建構邏輯只需要拿到建好
的 client 物件使用，不需要知道連線細節。
"""

import asyncio
import os

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore

# ── LLM 連線設定 ──
LLM_API_BASE = "https://integrate.api.nvidia.com/v1"  # NVIDIA NIM 的 OpenAI 相容端點
LLM_CONTEXT_WINDOW = 128000
LLM_TIMEOUT = 300.0

# ── 建圖抽取 LLM 設定 ──
GRAPH_EXTRACT_MAX_RETRIES = 8      # 免費端點對請求量敏感，加大重試靠指數退避慢慢送完
GRAPH_THROTTLE_SECONDS = 3.0       # 請求間隔：免費端點有每分鐘請求數上限，連發會 429

# ── Milvus 連線設定 ──
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "travel_preferences"
EMBED_DIM = 2048  # 需與 embedding 模型輸出維度一致


def build_llm():
    """建立 LLM 實例（透過 OpenAI-compatible 端點呼叫 NVIDIA NIM）。"""
    return OpenAILike(
        api_base=LLM_API_BASE,
        api_key=os.getenv("NVIDIA_API_KEY"),
        model=os.getenv("CHAT_MODEL"),
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=LLM_CONTEXT_WINDOW,
        timeout=LLM_TIMEOUT,
    )


class ThrottledOpenAILike(OpenAILike):
    """每次請求前固定等待，避免建圖的連續抽取請求打到免費端點的每分鐘上限（429）。"""

    async def achat(self, messages, **kwargs):
        await asyncio.sleep(GRAPH_THROTTLE_SECONDS)
        return await super().achat(messages, **kwargs)


def build_graph_llm():
    """建立建圖抽取用的 LLM。

    可用 GRAPH_CHAT_MODEL 指定與主 LLM 不同的模型（worker 配額隔離，
    建圖不會跟選路/合成/Agent 搶額度），未設定則沿用 CHAT_MODEL；
    指定的模型必須支援 function calling（抽取走結構化 tool call）。
    """
    return ThrottledOpenAILike(
        api_base=LLM_API_BASE,
        api_key=os.getenv("NVIDIA_API_KEY"),
        model=os.getenv("GRAPH_CHAT_MODEL") or os.getenv("CHAT_MODEL"),
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=LLM_CONTEXT_WINDOW,
        timeout=LLM_TIMEOUT,
        max_retries=GRAPH_EXTRACT_MAX_RETRIES,
    )


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
