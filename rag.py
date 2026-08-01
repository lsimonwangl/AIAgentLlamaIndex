"""
Router RAG - 旅遊偏好檢索器
===========================
rag.py 負責讀取 ./data 語料，並組裝 RouterQueryEngine：從 clients
取得模型與 Milvus 連線，在 _build_tools() 建四種索引並各自包成
QueryEngineTool，交由 RouterQueryEngine 依問題類型自動選擇檢索方式。
索引建構邏輯放在 _build_tools()（而非獨立檔案）是因為建好的索引只會
立刻被包成對應的 tool，沒有其他用途，建構與包裝放一起讀更清楚。

相較於 Lab2 只使用單一 VectorStoreIndex，Lab3 新增 SummaryIndex、
DocumentSummaryIndex 與 KeywordTableIndex，讓系統能針對不同類型的問題選擇最適合的檢索策略：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節
    - DocumentSummaryIndex：以每篇文件摘要為檢索單位，選出最相關的整趟紀錄
    - KeywordTableIndex：關鍵字反向表，適合精確名稱／專有名詞的字面命中

執行流程：
    0. 載入套件
    1. 透過 clients 建立 LLM、Embedding Model、Milvus 連線
    2. 讀取 ./data 語料，建立四種索引共用的切分器
    3. 在 _build_tools() 建立四個索引，各自包成 QueryEngineTool 並寫明適合的問題類型
    4. 透過 RouterQueryEngine + PydanticSingleSelector 自動選路

此模組提供 build_router_query_engine() 函式供 main.py 呼叫。
"""

import logging

from llama_index.core import (
    DocumentSummaryIndex,
    KeywordTableIndex,
    PromptTemplate,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool

import clients

# ── 自訂 QA prompt：把 RAG 從「直接回答問題」改為「整理過往台灣經驗作為素材」 ────
# 這樣即使使用者問海外目的地（例：京都有溪谷步道嗎），RAG 不會回「無京都資料」，
# 而是回傳使用者過往在台灣相關的具體經驗，供後續 Agent + Tavily 規劃使用
ORGANIZE_QA_TEMPLATE = PromptTemplate(
    "以下是從使用者過往「台灣」旅遊紀錄中檢索到的片段：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "使用者本次提問：{query_str}\n\n"
    "請從上述片段中整理出與本次提問『主題或感受』相關的『過往台灣經驗』，"
    "包括地點、體驗、感受、評價，用 2-4 句話歸納"
    "（若提問涉及花費、預算或統計，改為逐筆列出每筆紀錄的金額與天數，不受 2-4 句限制）。\n"
    "規則：\n"
    "1. 不要直接回答問題本身（例如不要說『京都有/沒有 X』）；"
    "但若提問涉及花費、預算、金額或統計，必須完整保留並逐筆列出片段中的"
    "金額、天數與「每人／總花費」標記，不得省略或概括成定性描述\n"
    "2. 不要捏造未在片段中出現的內容\n"
    "3. 若片段中完全無相關經驗，直接回「無相關過往經驗」\n"
    "4. 使用繁體中文輸出\n"
    "整理結果："
)


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


# ── 建立四個索引並各自包成選路工具 ───────────────────
def _build_tools(documents, splitter, llm, summary_llm, embed_model, vector_store):
    """依序建立四種索引，各自建完立刻包成 QueryEngineTool——每個索引只會被
    包成對應的那個 tool、沒有其他用途，建構與包裝放在一起讀更清楚。

    description 是 RouterQueryEngine 選路時唯一的判斷依據——selector 不會看索引
    內容，只把使用者問題連同這四段 description 交給 LLM 挑一個，所以每段都要寫清楚
    「什麼樣的問題該走這條」，以及「和其他三條的差別」。

    四個 query engine 共用 ORGANIZE_QA_TEMPLATE，讓檢索結果統一整理成「過往台灣
    經驗」而不是直接回答問題，後續才好交給 Agent 當規劃素材使用。

    回傳順序為 summary/vector/doc_summary/keyword，對應 main.py 顯示選路結果時
    用的 1~4 號名稱（見 main.py 的 tool_names），順序不可任意調動。
    """
    # ── SummaryIndex：適合聚合型問題 ──
    # 建索引時不做任何預處理（不打 LLM、不嵌入），只把切好的 chunk 存起來；
    # 成本在查詢時——tree_summarize 會掃過全部 chunk 逐層摘要合併，是四條路最吃
    # token 的，所以查詢用便宜的 summary_llm
    print("📋 建立 SummaryIndex...")
    summary_index = SummaryIndex.from_documents(documents, transformations=[splitter])
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            llm=summary_llm,
            response_mode="tree_summarize",  # 樹狀摘要：分組局部摘要，再逐層向上合併成總結
            summary_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答文件整體內容、跨文件摘要、主題總覽與綜合分析等總覽型問題，"
            "如歸納使用者整體旅遊偏好、旅行風格，或跨紀錄的統計與聚合（平均花費、預算結構）。"
            "例如：歸納我過去旅行偏好的行程節奏，安排日本行程時一天排幾個景點比較適合我。"
        ),
    )

    # ── VectorStoreIndex：適合細節型問題 ──
    # 建索引用 embed_model 把 chunk 嵌入成向量寫進 Milvus（非 chat LLM，便宜快）；
    # 查詢時把問題也轉成向量，取相似度最高的 top-k 個 chunk——語意（dense）檢索
    print("🔢 建立 VectorStoreIndex + Milvus...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        embed_model=embed_model,
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(
            llm=llm,
            similarity_top_k=5,  # 取回相似度最高的 5 個 chunk
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答景點體驗、美食評價、住宿細節、行程安排等針對單一片段的語意型問題，"
            "以語意相似度檢索，問題與紀錄用詞不同也能命中，重點是意思相近。"
            "例如：日本有沒有類似我在花蓮走過那條沿溪步道的健行路線。"
        ),
    )

    # ── DocumentSummaryIndex：以「每篇文件摘要」為檢索單位 ──
    # 和 SummaryIndex 的差別：這裡在建索引時就替每份文件（一檔一趟旅行）各生一段
    # LLM 摘要，查詢時先比對摘要選出最相關的幾趟，再帶回完整內容——檢索單位是
    # 「整篇」而非「片段」；逐篇打 LLM 呼叫次數比 SummaryIndex 多，用便宜的 summary_llm
    print("📝 建立 DocumentSummaryIndex（每篇各生一段 LLM 摘要）...")
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents,
        llm=summary_llm,
        embed_model=embed_model,  # 摘要向量化用，供查詢時以摘要相似度挑文件
        transformations=[splitter],
        show_progress=True,
    )
    doc_summary_tool = QueryEngineTool.from_defaults(
        query_engine=doc_summary_index.as_query_engine(
            llm=llm,                      # 合成用主模型；建索引時的每篇摘要才用便宜的 summary_llm
            retriever_mode="embedding",   # 以文件摘要的向量相似度挑文件（非每次 LLM 選文件）
            similarity_top_k=3,  # 以文件摘要相似度挑出最相關的 3 趟旅行紀錄
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合「以整趟旅行為單位」找出與提問最相關的幾筆完整旅遊紀錄，"
            "先比對每篇文件的摘要挑出最相關的旅行，再帶回那幾趟的完整內容回顧。"
            "重點：答案是『哪一趟／哪幾趟』整篇行程（天數、節奏、整體安排）；"
            "若只問單一景點或單一步道的細節，要用 VectorStoreIndex。"
            "例如：哪一趟整體規劃和我這次想要的最像、哪一趟玩得最不順我想避開。"
        ),
    )

    # ── KeywordTableIndex：精確名稱／專有名詞的字面命中 ──
    # 建索引時逐 chunk 打 LLM 抽關鍵字建反向表，查詢時抽問題關鍵字做字面（sparse）
    # 比對；呼叫次數比 DocumentSummaryIndex 多，同樣用便宜的 summary_llm
    print("🔑 建立 KeywordTableIndex（LLM 抽關鍵字）...")
    keyword_index = KeywordTableIndex.from_documents(
        documents,
        llm=summary_llm,
        transformations=[splitter],
        show_progress=True,
    )
    keyword_tool = QueryEngineTool.from_defaults(
        query_engine=keyword_index.as_query_engine(
            llm=llm,                          # 最終合成用 CHAT_MODEL；抽關鍵字用的是建索引時綁的便宜模型
            num_chunks_per_query=5,  # 關鍵字命中後最多取回的 chunk 數
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答『某個確切名稱、專有名詞是否出現、出現在哪幾趟』的精確比對問題，"
            "例如民宿名（木門厝）、店名（文章牛肉湯）、步道名（砂卡礑步道）、"
            "特有名詞（海龜、達悟族、螢火蟲）的字面命中查詢。"
            "與 VectorStoreIndex 的差別：這裡要精確字面命中，不是語意相似。"
        ),
    )

    return [summary_tool, vector_tool, doc_summary_tool, keyword_tool]


# ── 組裝 RouterQueryEngine ───────────────────────────
def build_router_query_engine():
    """建立 RouterQueryEngine，依問題類型自動在四種索引之間選路。

    先透過 clients 建好模型與 Milvus 連線，再用同一份 documents 與 splitter 交給
    _build_tools() 建出四個索引並包成 tool，交給 PydanticSingleSelector：selector
    只會挑一條路，由選中的 query engine 檢索，最後用主模型 llm 合成回應。
    """
    llm = clients.build_llm()
    summary_llm = clients.build_summary_llm()  # 便宜快速模型：SummaryIndex 查詢與另兩個索引建索引時都用它
    embed_model = clients.build_embed_model()
    vector_store = clients.build_milvus_vector_store()
    splitter = build_splitter()

    print("🔨 讀取 ./data 旅遊紀錄")
    documents = load_data_docs()

    tools = _build_tools(
        documents=documents,
        splitter=splitter,
        llm=llm,
        summary_llm=summary_llm,
        embed_model=embed_model,
        vector_store=vector_store,
    )

    # selector 會把使用者問題連同上面四個工具的 description 一起交給 LLM，
    # description 是 LLM 選路時唯一讀到的判斷依據：
    #   總覽型問題 → summary_tool（SummaryIndex，綜觀全部紀錄做摘要）
    #   語意型問題 → vector_tool（VectorStoreIndex，top-k 相似檢索）
    #   整趟紀錄回顧 → doc_summary_tool（DocumentSummaryIndex，以摘要挑整篇文件）
    #   精確名稱命中 → keyword_tool（KeywordTableIndex，關鍵字反向表字面命中）
    router_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        llm=llm,        # 合成用的 LLM：把選中 tool 檢索出的結果整理成最終回應
        verbose=False,  # 關掉內建選路 print，改由 main.py 統一輸出一行
    )

    # 壓掉 LlamaIndex router 內部的 INFO log（同樣會印 "Selecting query engine N"），避免重複
    logging.getLogger("llama_index.core.query_engine.router_query_engine").setLevel(
        logging.WARNING
    )

    print("✅ RouterQueryEngine 建立完成（SummaryIndex + VectorStoreIndex + DocumentSummaryIndex + KeywordTableIndex）")
    return router_engine
