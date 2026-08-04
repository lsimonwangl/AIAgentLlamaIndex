"""
Travel Agent - 模型與連線 Client
================================
clients.py 集中管理需要對外連線的 client 物件：
    - LLM（CHAT_MODEL）：主模型，負責 Agent 回答、Router 選路與最終回應合成
    - Summary LLM（SUMMARY_MODEL）：便宜快速模型，接高頻 LLM 呼叫
    - Embedding Model：把文字轉成向量，供向量相似度檢索使用
    - Milvus VectorStore：向量資料庫連線，所有向量都存這裡，本機不落地

集中在此檔案的原因：API 端點、金鑰、URI 這類連線設定是最常需要調整的部分，
統一放在一起維護；agent.py 與 rag 套件只需要拿到建好的 client 物件使用，
不需要知道連線細節。連線並非 RAG 專屬（Agent 用的 LLM 也共用同一份），
所以放在專案根目錄而不是併進 rag 套件。

此模組提供 build_llm()、build_summary_llm()、build_embed_model() 與
build_milvus_vector_store() 函式供 agent.py 與 rag 套件呼叫。
"""

import os

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore

# ── LLM 連線設定 ─────────────────────────────────────
LLM_API_BASE = "https://integrate.api.nvidia.com/v1"  # NVIDIA NIM 的 OpenAI 相容端點
LLM_CONTEXT_WINDOW = 128000  # 模型單次請求最多能吃的 token 數
LLM_TIMEOUT = 300.0          # 單次請求逾時秒數；免費端點較慢，放寬避免中途中斷

# ── Milvus 連線設定 ──────────────────────────────────
MILVUS_URI = "http://localhost:19530"  # 由 docker-compose.yml 起的本機 Milvus
MILVUS_VECTOR_STORE_INDEX_COLLECTION = "travel_preferences"  # VectorStoreIndex：chunk 向量
MILVUS_DOCUMENT_SUMMARY_INDEX_COLLECTION = (  # DocumentSummaryIndex：每篇摘要的向量
    "travel_preferences_summaries"
)
EMBED_DIM = 4096  # 需與 embedding 模型輸出維度一致（nv-embed-v1 為 4096），不一致會寫入失敗


# ── 建立 LLM 連線 ────────────────────────────────────
def build_llm(model=None):
    """建立 LLM 實例（透過 OpenAI-compatible 端點呼叫 NVIDIA NIM）。

    NVIDIA NIM 提供 OpenAI 相容的 API，所以不需要專用 SDK，用 OpenAILike 指到
    它的端點就能接。全專案的 LLM 都從這裡產生，換模型或換端點只改這一處。

    model 不傳時用 .env 的 CHAT_MODEL；傳入時改用指定模型
    （build_summary_llm() 就是傳入 SUMMARY_MODEL 的薄包裝）。
    """
    return OpenAILike(
        # NVIDIA NIM 的 OpenAI 相容端點
        api_base=LLM_API_BASE,
        # 從環境變數讀取 API 金鑰，不寫死在程式碼裡
        api_key=os.getenv("NVIDIA_API_KEY"),
        # 未指定就用 .env 的 CHAT_MODEL
        model=model or os.getenv("CHAT_MODEL"),
        # 走 chat completion 介面而非舊的 completion
        is_chat_model=True,
        # 宣告支援 function calling，Agent 才能呼叫工具、selector 才能輸出結構化選路結果
        is_function_calling_model=True,
        context_window=LLM_CONTEXT_WINDOW,
        timeout=LLM_TIMEOUT,
    )


# ── 建立便宜快速模型的 LLM 連線 ────────────────────────
def build_summary_llm():
    """建立便宜快速模型的 LLM 實例：讀 .env 的 SUMMARY_MODEL。

    三處高頻或全量的 LLM 呼叫都用它，避免用主模型拖慢回應、燒掉額度：
        - SummaryIndex 查詢：tree_summarize 會掃過全部 chunk
        - DocumentSummaryIndex 建索引：每篇文件各生一段摘要
        - KeywordTableIndex 建索引：每個 chunk 各抽一次關鍵字

    未設 SUMMARY_MODEL 時會傳入 None，build_llm() 會 fallback 回 CHAT_MODEL。
    """
    return build_llm(os.getenv("SUMMARY_MODEL"))


# ── 建立 Embedding Model 連線 ──────────────────────────
def build_embed_model():
    """建立 Embedding Model 實例：讀 .env 的 EMBEDDING_MODEL。

    VectorStoreIndex 把 chunk 嵌入成向量、DocumentSummaryIndex 把每篇摘要嵌入成
    向量，都用這個模型；它不是 chat LLM，呼叫便宜快速，不佔 chat 額度。

    注意：模型的輸出維度必須與 EMBED_DIM 一致，換模型時兩邊要一起改。
    """
    return NVIDIAEmbedding(model=os.getenv("EMBEDDING_MODEL"))


# ── 建立 Milvus VectorStore 連線 ───────────────────────
def build_milvus_vector_store(collection_name: str, overwrite: bool = False):
    """建立 Milvus VectorStore 連線，接上指定的 collection。

    collection_name 刻意不設預設值，因為專案用了兩個語意不同的 collection：
    VectorStoreIndex 存 chunk 向量（比的是「片段像不像」），DocumentSummaryIndex
    存整篇摘要的向量（比的是「整趟旅行像不像」）。兩者混在同一個 collection
    查詢會撈到不相關的向量，所以每次呼叫都要明確寫出要接哪一個。

    overwrite 決定是清空重建還是接上既有資料：建索引時要 True 清掉舊向量，
    查詢時必須是 False，否則會把離線建好的向量洗掉。
    """
    return MilvusVectorStore(
        # 本機 Milvus 服務位址（docker-compose.yml 起的那一組）
        uri=MILVUS_URI,
        collection_name=collection_name,
        # 向量維度，需與 embedding 模型一致
        dim=EMBED_DIM,
        overwrite=overwrite,
    )
