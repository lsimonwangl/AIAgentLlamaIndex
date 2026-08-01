"""
Router RAG - 離線建索引
=========================
index.py 負責讀取 ./data 語料，離線建立 retrieval.py 查詢時要用的四種索引，
並持久化到 STORAGE_DIR，供 main.py 啟動時直接讀回，不需要每次重跑。

DocumentSummaryIndex、KeywordTableIndex 建索引時都要逐一呼叫 LLM（逐篇生摘要、
逐 chunk 抽關鍵字），若跟著 main.py 每次啟動一起做，會讓每次重啟都重新消耗一次
LLM 額度，也是先前一直碰到 NVIDIA 端點限流的主因。拆成這支獨立腳本後，
只有 ./data 語料異動時才需要重新執行；其餘時候 main.py 只讀既有索引，不再重建。

向量資料一律存 Milvus，本機 STORAGE_DIR 只保留非向量的結構性資料（docstore、
index_store）：
    - VectorStoreIndex 的 chunk 向量 → Milvus MILVUS_COLLECTION
    - DocumentSummaryIndex 的摘要向量 → Milvus MILVUS_SUMMARY_COLLECTION（獨立
      collection，不能跟 chunk 向量共用，語意不同）
    - SummaryIndex、KeywordTableIndex 本來就不含向量
（DocumentSummaryIndex 把摘要向量寫進外部 vector_store 是 LlamaIndex 官方支援的
行為，已用原始碼與實測驗證過；MilvusVectorStore.add() 預設不自動 flush，所以
建完索引後這裡會手動呼叫 flush，避免緊接著查詢時 Milvus 還沒反映剛寫入的資料）

執行流程：
    0. 載入套件與環境變數
    1. 透過 clients 建立 LLM、Embedding Model、兩個 Milvus 連線（chunk／摘要）
    2. 讀取 ./data 語料，建立四種索引共用的切分器
    3. 依序建立 SummaryIndex、VectorStoreIndex（寫入 Milvus chunk collection）、
       DocumentSummaryIndex（docstore／index_store 與 Summary／Keyword 共用，
       摘要向量寫入 Milvus 摘要 collection）、KeywordTableIndex
    4. 將 docstore／index_store 持久化到 STORAGE_DIR，供 retrieval.py 讀回；
       所有向量資料都已經寫進 Milvus，本機不保留任何向量；手動 flush 兩個
       Milvus collection

執行方式（需在專案根目錄執行，讓 ./data、./storage 相對路徑正確）：
    python -m rag.index

（./data 語料異動後，需重新執行這支腳本，retrieval.py 才會讀到最新內容）
"""

# 載入套件與環境變數
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    DocumentSummaryIndex,
    KeywordTableIndex,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter

import clients
from .retrieval import DOC_SUMMARY_INDEX_ID, KEYWORD_INDEX_ID, STORAGE_DIR, SUMMARY_INDEX_ID


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
    """建立節點切分器，供四種索引共用同一套切分設定。"""
    # chunk_size=256：每個 chunk 約 256 token；chunk_overlap=50：相鄰 chunk 重疊 50 token，
    # 避免語句被切斷後兩邊都讀不懂
    return SentenceSplitter(chunk_size=256, chunk_overlap=50)


# ── 建立四種索引並持久化 ───────────────────────────────
def build_indexes():
    """依序建立四種索引：SummaryIndex、DocumentSummaryIndex、KeywordTableIndex
    共用同一份 docstore／index_store（存在 storage_context 裡），最後一起 persist 到
    STORAGE_DIR——這三個索引都不把向量落地本機：SummaryIndex、KeywordTableIndex
    本來就沒有向量；DocumentSummaryIndex 的摘要向量改指到 Milvus
    （summary_vector_store），做法是另外組一個 doc_summary_storage_context，
    借用同一份 docstore／index_store，只把 vector_store 換成 Milvus。
    VectorStoreIndex 則完全獨立，向量直接寫進 Milvus 的 chunk collection。
    """
    summary_llm = clients.build_summary_llm()  # 便宜快速模型：DocumentSummaryIndex、KeywordTableIndex 建索引都用它
    embed_model = clients.build_embed_model()
    # overwrite=True：離線建索引時才需要清空舊資料重建，main.py 查詢時不能這樣做
    vector_store = clients.build_milvus_vector_store(clients.MILVUS_COLLECTION, overwrite=True)
    summary_vector_store = clients.build_milvus_vector_store(
        clients.MILVUS_SUMMARY_COLLECTION, overwrite=True
    )
    splitter = build_splitter()

    print("🔨 讀取 ./data 旅遊紀錄")
    documents = load_data_docs()

    # SummaryIndex、DocumentSummaryIndex、KeywordTableIndex 共用這個 storage_context 的
    # docstore／index_store，最後才能一次 persist 到同一份 STORAGE_DIR；
    # 不傳 vector_store，向量一律走 Milvus，不會有東西寫進本機的 default__vector_store.json
    storage_context = StorageContext.from_defaults()

    # ── SummaryIndex：適合聚合型問題 ──
    # 建索引時不做任何預處理（不打 LLM、不嵌入），只把切好的 chunk 存起來
    print("📋 建立 SummaryIndex...")
    summary_index = SummaryIndex.from_documents(
        documents,
        transformations=[splitter],
        storage_context=storage_context,
    )
    summary_index.set_index_id(SUMMARY_INDEX_ID)

    # ── VectorStoreIndex：適合細節型問題 ──
    # 用 embed_model 把 chunk 嵌入成向量寫進 Milvus chunk collection（非 chat LLM，便宜快）
    print("🔢 建立 VectorStoreIndex + Milvus...")
    vector_storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=vector_storage_context,
        transformations=[splitter],
        embed_model=embed_model,
    )

    # ── DocumentSummaryIndex：以「每篇文件摘要」為檢索單位 ──
    # 替每份文件（一檔一趟旅行）各生一段 LLM 摘要，逐篇打 LLM 呼叫次數多，用便宜的 summary_llm；
    # 摘要向量寫進 Milvus 摘要 collection，docstore／index_store 借用 storage_context 的物件
    # （同一份，不是複製），才能讓這個索引的結構性資料也一起被下面的 persist() 存到 STORAGE_DIR
    print("📝 建立 DocumentSummaryIndex（每篇各生一段 LLM 摘要，摘要向量寫入 Milvus）...")
    doc_summary_storage_context = StorageContext.from_defaults(
        docstore=storage_context.docstore,
        index_store=storage_context.index_store,
        vector_store=summary_vector_store,
    )
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents,
        llm=summary_llm,
        embed_model=embed_model,  # 摘要向量化用，供查詢時以摘要相似度挑文件
        transformations=[splitter],
        storage_context=doc_summary_storage_context,
        show_progress=True,
    )
    doc_summary_index.set_index_id(DOC_SUMMARY_INDEX_ID)

    # ── KeywordTableIndex：精確名稱／專有名詞的字面命中 ──
    # 逐 chunk 打 LLM 抽關鍵字建反向表，呼叫次數比 DocumentSummaryIndex 多，同樣用便宜的 summary_llm
    print("🔑 建立 KeywordTableIndex（LLM 抽關鍵字）...")
    keyword_index = KeywordTableIndex.from_documents(
        documents,
        llm=summary_llm,
        transformations=[splitter],
        storage_context=storage_context,
        show_progress=True,
    )
    keyword_index.set_index_id(KEYWORD_INDEX_ID)

    # MilvusVectorStore.add() 預設不會自動 flush，VectorStoreIndex、DocumentSummaryIndex
    # 建索引時內部呼叫 add() 都沒有帶 force_flush；所有向量寫入都做完後才 flush 兩個
    # collection，確保緊接著啟動 main.py 查詢時，Milvus 已經能反映剛寫入的向量
    # （已用小規模測試驗證過沒 flush 時 get_collection_stats() 可能暫時看不到剛寫入的資料）
    print("💨 flush Milvus collection...")
    vector_store.client.flush(clients.MILVUS_COLLECTION)
    summary_vector_store.client.flush(clients.MILVUS_SUMMARY_COLLECTION)

    print(f"💾 持久化非向量資料到 {STORAGE_DIR}（docstore、index_store）...")
    storage_context.persist(persist_dir=STORAGE_DIR)

    print("✅ 索引建立完成（chunk 向量、摘要向量都已寫入 Milvus，本機只保留結構性資料）")


if __name__ == "__main__":
    build_indexes()
