"""
Router RAG - 離線建索引
=======================
index.py 負責讀取 ./data 資料，離線建立四種索引並存起來，供 retrieval.py 在
查詢時直接讀回。

為什麼要拆成獨立一支腳本：DocumentSummaryIndex 建索引時要逐篇文件呼叫 LLM 生摘要、
KeywordTableIndex 要逐 chunk 呼叫 LLM 抽關鍵字，資料越多呼叫次數越多。這些結果只要
資料沒變就不會變，沒有理由每次啟動都重算一次。拆開之後，只有 ./data 異動時才需要
重跑這支腳本，main.py 啟動時只讀既有索引。

索引資料分兩個地方存：
    - 向量 → Milvus。VectorStoreIndex 的 chunk 向量存 MILVUS_VECTOR_STORE_INDEX_COLLECTION，
      DocumentSummaryIndex 的摘要向量存 MILVUS_DOCUMENT_SUMMARY_INDEX_COLLECTION，兩者語意不同
      （一個比片段像不像、一個比整趟像不像），不能共用同一個 collection
    - 非向量的結構性資料（原文、索引結構）→ 本機 STORAGE_DIR
    （SummaryIndex、KeywordTableIndex 本來就不含向量，只有結構性資料）

執行流程：
    0. 載入套件與環境變數
    1. 透過 clients 建立 LLM、Embedding Model、兩個 Milvus 連線（chunk／摘要）
    2. 讀取 ./data 語料，建立四種索引共用的切分器
    3. 依序建立 SummaryIndex、VectorStoreIndex、DocumentSummaryIndex、KeywordTableIndex
    4. 把結構性資料存到 STORAGE_DIR，向量的部分寫進 Milvus

執行方式（需在專案根目錄執行，讓 ./data、./storage 相對路徑正確）：
    python -m rag.index

（./data 語料異動後要重跑這支腳本，main.py 才會讀到最新內容）
"""

# 載入套件與環境變數
from dotenv import load_dotenv
load_dotenv()

import os

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


# ── 建立四種索引並存起來 ─────────────────────────────
def build_indexes():
    """用同一批 documents 建立四種索引，把結果存到 Milvus 與 STORAGE_DIR。

    四種索引雖然檢索方式完全不同，卻共用同一批文件與同一套切分設定——這樣同一段
    旅遊紀錄在四種索引裡的邊界一致，比較起來才有意義。

    存放方式分兩類：VectorStoreIndex 的 chunk 向量與 DocumentSummaryIndex 的摘要
    向量各自寫進獨立的 Milvus collection；四種索引的原文與索引結構則集中在同一份
    storage_context，最後一次存到 STORAGE_DIR。

    這是整個專案最花時間的一步——DocumentSummaryIndex 逐篇、KeywordTableIndex 逐 chunk
    都要打 LLM，語料越大越慢，所以才拆成離線腳本，只在語料變動時執行。
    """
    summary_llm = clients.build_summary_llm()  # 便宜快速模型：兩個要打 LLM 的索引都用它
    embed_model = clients.build_embed_model()
    # overwrite=True：建索引要清掉舊向量重建；查詢時必須是 False，否則會把資料洗掉
    chunk_vector_store = clients.build_milvus_vector_store(
        clients.MILVUS_VECTOR_STORE_INDEX_COLLECTION, overwrite=True
    )
    document_summary_vector_store = clients.build_milvus_vector_store(
        clients.MILVUS_DOCUMENT_SUMMARY_INDEX_COLLECTION, overwrite=True
    )
    # 每個 chunk 約 256 token，相鄰 chunk 重疊 50 token，避免語句被切斷後兩邊都讀不懂；
    # 四種索引共用同一套切分設定
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

    print("🔨 讀取 ./data 旅遊紀錄")
    documents = load_data_docs()

    # 四種索引的原文與索引結構都放進這一份 storage_context，最後才能一次存到 STORAGE_DIR
    storage_context = StorageContext.from_defaults()

    # ── SummaryIndex：適合總覽型問題 ──
    # 四種索引裡唯一建索引時什麼都不做的：不打 LLM、不嵌入，只把切好的 chunk 依序存起來。
    # 成本全部留到查詢時——那時才會掃過全部 chunk 逐層摘要合併
    print("📋 建立 SummaryIndex...")
    summary_index = SummaryIndex.from_documents(
        documents,
        transformations=[splitter],
        storage_context=storage_context,
    )
    summary_index.set_index_id(SUMMARY_INDEX_ID)

    # ── VectorStoreIndex：適合語意型問題 ──
    # 用 embed_model 把每個 chunk 轉成向量寫進 Milvus。embedding 不是 chat LLM，
    # 呼叫便宜快速，所以這個索引雖然要處理每個 chunk，成本仍遠低於下面兩個
    print("🔢 建立 VectorStoreIndex + Milvus...")
    chunk_vector_storage_context = StorageContext.from_defaults(
        vector_store=chunk_vector_store
    )
    VectorStoreIndex.from_documents(
        documents,
        storage_context=chunk_vector_storage_context,
        transformations=[splitter],
        embed_model=embed_model,
    )

    # ── DocumentSummaryIndex：以「每篇文件摘要」為檢索單位 ──
    # 和 SummaryIndex 的差別：SummaryIndex 查詢時才掃全部 chunk；這裡在建索引時就替
    # 每份文件（一檔一趟旅行）各生一段 LLM 摘要，查詢時先比對摘要挑出最相關的幾趟，
    # 再帶回那幾趟的完整內容——檢索單位是「整篇」而非「片段」。
    # 逐篇打 LLM，所以用便宜的 summary_llm；摘要向量存到獨立的 Milvus collection
    print("📝 建立 DocumentSummaryIndex（每篇各生一段 LLM 摘要，摘要向量寫入 Milvus）...")
    document_summary_storage_context = StorageContext.from_defaults(
        docstore=storage_context.docstore,
        index_store=storage_context.index_store,
        vector_store=document_summary_vector_store,
    )
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents,
        llm=summary_llm,
        embed_model=embed_model,  # 摘要向量化用，供查詢時以摘要相似度挑文件
        transformations=[splitter],
        storage_context=document_summary_storage_context,
        show_progress=True,
    )
    doc_summary_index.set_index_id(DOC_SUMMARY_INDEX_ID)

    # ── KeywordTableIndex：精確名稱／專有名詞的字面命中 ──
    # 逐 chunk 打 LLM 抽關鍵字，建成「關鍵字 → chunk」的反向表，查詢時做字面比對。
    # 因為是逐 chunk 而非逐篇，呼叫次數比 DocumentSummaryIndex 多，同樣用便宜的 summary_llm
    print("🔑 建立 KeywordTableIndex（LLM 抽關鍵字）...")
    keyword_index = KeywordTableIndex.from_documents(
        documents,
        llm=summary_llm,
        transformations=[splitter],
        storage_context=storage_context,
        show_progress=True,
    )
    keyword_index.set_index_id(KEYWORD_INDEX_ID)

    # 把原文（docstore）與索引結構（index_store）存到 STORAGE_DIR，retrieval.py 會讀回這兩份；
    # 向量的部分已經在 Milvus，本機不重複存一份
    print(f"💾 持久化非向量資料到 {STORAGE_DIR}（docstore、index_store）...")
    storage_context.docstore.persist(persist_path=os.path.join(STORAGE_DIR, "docstore.json"))
    storage_context.index_store.persist(persist_path=os.path.join(STORAGE_DIR, "index_store.json"))

    print("✅ 索引建立完成（chunk 向量、摘要向量都已寫入 Milvus，本機只保留結構性資料）")


if __name__ == "__main__":
    build_indexes()
