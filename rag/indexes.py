"""
Router RAG - 索引建構
=====================
indexes.py 負責把 ./data 的旅遊紀錄讀入，並建立四種索引：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節
    - DocumentSummaryIndex：以「每篇文件摘要」為檢索單位，選出最相關的整趟紀錄
    - KeywordTableIndex：LLM 抽關鍵字建反向表，適合精確名稱／專有名詞的字面命中

每個 build_*_index() 只負責「documents → index」，所需的 client 物件
（llm、embed_model、vector_store）一律由呼叫端（engine.py）透過
clients.py 建立後傳入，本檔案不處理連線設定。
"""

from llama_index.core import (
    DocumentSummaryIndex,
    KeywordTableIndex,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    get_response_synthesizer,
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


# ── 建立 DocumentSummaryIndex：以「每篇文件摘要」為檢索單位 ────────
def build_document_summary_index(documents, splitter, llm, embed_model):
    """建立 DocumentSummaryIndex：每篇文件先由 LLM 生一段摘要，查詢時用摘要挑文件。

    和 SummaryIndex 的差別：SummaryIndex 查詢時才掃全部 chunk；這裡在「建索引時」
    就替每份文件（一檔一趟旅行）各生一段摘要，查詢時先比對這些摘要選出最相關的
    幾趟，再把那幾趟的完整 chunk 帶回合成——檢索單位是「整篇」而非「片段」。

    代價：建索引時每份文件各呼叫一次 LLM 生摘要（20 篇＝20 次），可能撞 NVIDIA
    端點限流；全同步（use_async=False）避免在 main.py 的 event loop 內巢狀 asyncio.run()。
    """
    print("📝 建立 DocumentSummaryIndex（每篇各生一段 LLM 摘要）...")
    # tree_summarize：把單篇的多個 chunk 分組局部摘要再逐層合併成該篇的整篇摘要
    # llm 要傳進來，否則 synthesizer 會 fallback 到全域 Settings.llm（預設 OpenAI）而非 SUMMARY_MODEL
    response_synthesizer = get_response_synthesizer(llm=llm, response_mode="tree_summarize", use_async=False)
    return DocumentSummaryIndex.from_documents(
        documents,
        # 生每篇摘要用的 LLM
        llm=llm,
        # 摘要向量化用的 embedding model，供查詢時以摘要相似度挑文件
        embed_model=embed_model,
        # 與其他索引相同的切分設定
        transformations=[splitter],
        # 生成每篇整篇摘要的合成器
        response_synthesizer=response_synthesizer,
        # 顯示建索引進度條
        show_progress=True,
    )


# ── 建立 KeywordTableIndex：精確名稱／專有名詞的字面命中 ────────
def build_keyword_index(documents, splitter, llm):
    """建立 KeywordTableIndex：LLM 逐 chunk 抽關鍵字建「關鍵字→chunk」反向表。

    查詢時抽問題關鍵字去表裡精確命中，按共同關鍵字數排序取回 chunk——屬於
    字面（sparse）檢索，適合「某個確切名稱是否出現、在哪幾趟」這類問題。
    用 LLM 抽關鍵字（非 SimpleKeywordTableIndex 的 regex），中文免斷詞、免 jieba。

    代價：建索引時逐 chunk 各打一次 LLM，呼叫次數比 DocumentSummaryIndex 多；
    傳入的 llm 建議用便宜快速模型（抽關鍵字不需高階模型）。
    """
    print("🔑 建立 KeywordTableIndex（LLM 抽關鍵字）...")
    return KeywordTableIndex.from_documents(
        documents,
        # 抽關鍵字用的 LLM（建索引逐 chunk、查詢時抽問題關鍵字都用它）
        llm=llm,
        # 與其他索引相同的切分設定
        transformations=[splitter],
        # 顯示建索引進度條
        show_progress=True,
    )
