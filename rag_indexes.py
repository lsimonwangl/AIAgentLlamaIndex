"""
Router RAG - 索引建構
=====================
rag_indexes.py 負責把 ./data 的旅遊紀錄讀入，並建立兩種索引：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節

每個 build_*_index() 只負責「documents → index」，所需的 client 物件
（embed_model、vector_store）一律由呼叫端（rag.py）透過
rag_clients.py 建立後傳入，本檔案不處理連線設定。
"""

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter


# ── 切分設定 ─────────────────────────────────────────
# 每個 chunk 約 256 token，相鄰 chunk 重疊 50 token，避免語句被切斷後兩邊都讀不懂
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50


# ── 讀取旅遊紀錄 ─────────────────────────────────────
def load_data_docs():
    """讀取 ./data 資料夾中的文字檔，轉成 LlamaIndex Document 物件列表。"""
    reader = SimpleDirectoryReader(
        # 語料資料夾：一個 .txt 檔＝一筆旅遊紀錄
        input_dir="./data",
        # 只讀 .txt，忽略其他格式的檔案
        required_exts=[".txt"],
    )
    # 讀成 Document 物件列表，檔名等檔案資訊會自動放進 metadata
    return reader.load_data()


# ── 文件切分器 ───────────────────────────────────────
def build_splitter():
    """建立節點切分器，供三種索引共用同一套切分設定。"""
    return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# ── 建立 SummaryIndex：適合聚合型問題 ─────────────────
def build_summary_index(documents, splitter):
    """建立 SummaryIndex：查詢時掃過所有 chunk 做 tree_summarize，適合聚合型問題。"""
    print("📋 建立 SummaryIndex...")
    # 建索引前先用 splitter 切 chunk；SummaryIndex 本身不做預處理，成本在查詢時
    return SummaryIndex.from_documents(documents, transformations=[splitter])


# ── 建立 VectorStoreIndex：適合細節型問題 ─────────────
def build_vector_index(documents, splitter, embed_model, vector_store):
    """建立 VectorStoreIndex：把傳入的 vector_store（Milvus）包進 StorageContext 後向量化寫入。"""
    print("🔢 建立 VectorStoreIndex + Milvus...")
    # 把 Milvus 連線注入儲存層，向量會存進 Milvus 而非預設記憶體
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        # 從 ./data 讀進來的 Document
        documents,
        # 指定向量存到 Milvus
        storage_context=storage_context,
        # 建索引前先用 splitter 切 chunk
        transformations=[splitter],
        # 把 chunk 轉成向量的 embedding model
        embed_model=embed_model,
    )
