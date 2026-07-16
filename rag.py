"""
Router RAG - 旅遊偏好檢索器
===========================
rag.py 負責將 ./data 中的旅遊紀錄建立兩種索引，
並透過 RouterQueryEngine 依問題類型自動選擇檢索方式。

相較於 Lab2 只使用單一 VectorStoreIndex，Lab3 新增 SummaryIndex，
讓系統能針對不同類型的問題選擇最適合的檢索策略：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節

執行流程：
    0. 載入套件與環境變數
    1. 從 ./data 讀取旅遊紀錄文字檔
    2. 設定全域 LLM 與 Embedding Model
    3. 建立 SummaryIndex（聚合型問題）
    4. 建立 VectorStoreIndex + Milvus（細節型問題）
    5. 將兩個索引包成 QueryEngineTool，寫明各自適合的問題類型
    6. 透過 RouterQueryEngine + EmbeddingSingleSelector 自動選路

此模組提供 build_router_query_engine() 函式供 main.py 呼叫。
"""

import os

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import EmbeddingSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore


def load_data_docs():
    """讀取 ./data 資料夾中的文字檔，轉成 LlamaIndex Document 物件列表。"""
    reader = SimpleDirectoryReader(
        input_dir="./data",
        required_exts=[".txt"],
    )
    return reader.load_data()


def build_router_query_engine():
    """建立 RouterQueryEngine，依問題類型自動在 SummaryIndex 與 VectorStoreIndex 之間選路。"""

    # 設定全域 LLM 與 Embedding Model
    Settings.llm = OpenAILike(
        model=os.getenv("OPENAI_COMPATIBLE_MODEL", "deepseek-ai/deepseek-v4-flash"),
        api_base=os.getenv(
            "OPENAI_COMPATIBLE_API_BASE",
            "https://integrate.api.nvidia.com/v1",
        ),
        api_key=(
            os.getenv("OPENAI_COMPATIBLE_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ),
        timeout=240.0,
        max_retries=2,
        max_tokens=800,
        is_chat_model=True,
        is_function_calling_model=False,
    )
    embedding_model_name = os.getenv(
        "NVIDIA_EMBEDDING_MODEL",
        "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    )
    embedding_dim = int(os.getenv("NVIDIA_EMBEDDING_DIM", "2048"))
    Settings.embed_model = NVIDIAEmbedding(model=embedding_model_name)

    # 設定文字切分器（對應 Lab2 的 chunk_size=256, chunk_overlap=50）
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

    print("讀取 ./data 旅遊紀錄")
    documents = load_data_docs()

    # ── 建立 SummaryIndex：適合聚合型問題 ──
    # 查詢時會掃過所有 chunk 做 tree_summarize，能從全部紀錄中歸納整體偏好
    print("建立 SummaryIndex...")
    summary_index = SummaryIndex.from_documents(
        documents,
        transformations=[splitter],
    )

    # ── 建立 VectorStoreIndex + Milvus：適合細節型問題 ──
    # 將 chunk 向量化存入 Milvus，查詢時用 cosine similarity 找最相關的片段
    print("建立 VectorStoreIndex + Milvus...")
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530",
        collection_name="travel_preferences",
        dim=embedding_dim,
        overwrite=True,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )

    # ── 包成 QueryEngineTool，寫明各自適合的問題類型 ──
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            response_mode="tree_summarize"
        ),
        description=(
            "適合回答使用者整體旅遊偏好、旅行風格、行程模式、"
            "預算習慣等需要綜觀所有旅遊紀錄的聚合型問題。"
            "例如：幫我安排大阪行程、我的旅遊風格是什麼。"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(similarity_top_k=5),
        description=(
            "適合查詢特定景點體驗、美食評價、住宿細節、"
            "交通方式或某次旅行的具體經歷等細節型問題。"
            "例如：我住過什麼溫泉民宿、我去花蓮吃了什麼。"
        ),
    )

    # ── RouterQueryEngine：用 embedding 相似度穩定選路，避免 LLM selector JSON 解析失敗 ──
    router_engine = RouterQueryEngine(
        selector=EmbeddingSingleSelector.from_defaults(embed_model=Settings.embed_model),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=False,
    )

    print("RouterQueryEngine 建立完成（SummaryIndex + VectorStoreIndex）")
    return router_engine
