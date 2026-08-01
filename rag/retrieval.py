"""
Router RAG - 旅遊偏好檢索器
=================================
retrieval.py 負責在查詢時把 index.py 離線建好的索引讀回來，組裝成 RouterQueryEngine：
從 clients 取得模型與兩個 Milvus 連線（chunk 向量、摘要向量），在 _wrap_tools()
把讀回的四個索引各自包成 QueryEngineTool，交由 RouterQueryEngine 依問題類型自動
選擇檢索方式。

四個索引裡只有 SummaryIndex、KeywordTableIndex 完全不含向量，其餘向量一律存
Milvus（VectorStoreIndex 的 chunk 向量、DocumentSummaryIndex 的摘要向量各自
獨立 collection），本機 STORAGE_DIR 只保留 docstore／index_store 這類結構性資料。

索引本身不在這裡建立——建索引屬於一次性、LLM 密集的離線工作（DocumentSummaryIndex、
KeywordTableIndex 建索引時都要逐一呼叫 LLM），若跟著 main.py 每次啟動一起做，
會讓每次重啟都重新消耗一次 LLM 額度，也是先前一直碰到 NVIDIA 端點限流的主因。
因此拆到 index.py 獨立執行（同一個 rag 套件內），執行完會把索引持久化到
STORAGE_DIR；retrieval.py 只負責在查詢時讀回已建好的索引。

相較於 Lab2 只使用單一 VectorStoreIndex，Lab3 新增 SummaryIndex、
DocumentSummaryIndex 與 KeywordTableIndex，讓系統能針對不同類型的問題選擇最適合的檢索策略：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節
    - DocumentSummaryIndex：以每篇文件摘要為檢索單位，選出最相關的整趟紀錄
    - KeywordTableIndex：關鍵字反向表，適合精確名稱／專有名詞的字面命中

執行流程：
    0. 載入套件
    1. 透過 clients 建立 LLM、Embedding Model、Milvus 連線
    2. 從 STORAGE_DIR 讀回 index.py 離線建好的四個索引
    3. 在 _wrap_tools() 把四個索引各自包成 QueryEngineTool 並寫明適合的問題類型
    4. 透過 RouterQueryEngine + PydanticSingleSelector 自動選路

此模組提供 build_router_query_engine()、retrieve_preferences() 函式供 main.py 呼叫；
STORAGE_DIR 與三個 *_INDEX_ID 常數供 index.py 建索引、持久化時使用同一份設定，
兩邊要一起改。
"""

import logging
from pathlib import Path

from llama_index.core import (
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores.simple import SimpleVectorStore

import clients

# ── 索引持久化位置與各索引的固定 index_id ─────────────
# index.py 建索引、persist 時，與這裡 load_index_from_storage() 讀回時都要用同一份，
# 才能在同一個 storage_context 裡正確取出對應的索引；VectorStoreIndex 的向量資料
# 存在外部 Milvus，不需要 index_id
STORAGE_DIR = "./storage"
SUMMARY_INDEX_ID = "summary_index"
DOC_SUMMARY_INDEX_ID = "doc_summary_index"
KEYWORD_INDEX_ID = "keyword_index"

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


# ── 把讀回的四個索引各自包成選路工具 ───────────────────
def _wrap_tools(summary_index, vector_index, doc_summary_index, keyword_index, llm, summary_llm):
    # description 是 RouterQueryEngine 選路時唯一的判斷依據——selector 不會看索引
    # 內容，只把使用者問題連同這四段 description 交給 LLM 挑一個，所以每段都要寫清楚
    # 「什麼樣的問題該走這條」，以及「和其他三條的差別」
    # 四個 query engine 共用 ORGANIZE_QA_TEMPLATE，讓檢索結果統一整理成「過往台灣
    # 經驗」而不是直接回答問題，後續才好交給 Agent 當規劃素材使用
    # 回傳順序為 summary/vector/doc_summary/keyword，對應 main.py 顯示選路結果時
    # 用的 1~4 號名稱（見 main.py 的 tool_names），順序不可任意調動

    # ── SummaryIndex：適合聚合型問題 ──
    # 查詢時 tree_summarize 會掃過全部 chunk 逐層摘要合併，是四條路最吃
    # token 的，所以查詢用便宜的 summary_llm
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
    # 查詢時把問題也轉成向量，取相似度最高的 top-k 個 chunk——語意（dense）檢索
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
    # 和 SummaryIndex 的差別：建索引時已替每份文件（一檔一趟旅行）各生一段 LLM 摘要，
    # 查詢時先比對摘要選出最相關的幾趟，再帶回完整內容——檢索單位是「整篇」而非「片段」
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
    # 查詢時抽問題關鍵字，對建索引時生成的反向表做字面（sparse）比對
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
    if not Path(STORAGE_DIR).exists():
        raise RuntimeError(
            f"找不到索引目錄 {STORAGE_DIR}，請先在專案根目錄執行「python -m rag.index」"
            "離線建立索引，再啟動 main.py"
        )

    llm = clients.build_llm()
    summary_llm = clients.build_summary_llm()  # 便宜快速模型：SummaryIndex 查詢用它
    embed_model = clients.build_embed_model()
    # overwrite=False：這裡只是接上 index.py 已經寫好的向量資料，不能覆寫
    vector_store = clients.build_milvus_vector_store(clients.MILVUS_COLLECTION, overwrite=False)
    summary_vector_store = clients.build_milvus_vector_store(
        clients.MILVUS_SUMMARY_COLLECTION, overwrite=False
    )

    print(f"📂 從 {STORAGE_DIR} 讀取離線建好的索引...")
    # 只從 STORAGE_DIR 讀 docstore、index_store 這兩份非向量資料；vector_store 給一個
    # 空的 SimpleVectorStore（純記憶體，不落地檔案）——所有向量都在 Milvus，
    # 這裡只是為了滿足 LlamaIndex 的 BaseIndex.__init__ 一定會存取
    # storage_context.vector_store 的限制，不需要對應 index.py 那邊寫出任何
    # default／image__vector_store.json、graph_store.json 這類用不到的空殼檔案
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(STORAGE_DIR),
        index_store=SimpleIndexStore.from_persist_dir(STORAGE_DIR),
        vector_store=SimpleVectorStore(),
    )

    summary_index = load_index_from_storage(storage_context, index_id=SUMMARY_INDEX_ID)
    # KeywordTableIndex.__init__ 會在建構時就 resolve 一個 llm（self._llm = llm or Settings.llm），
    # 不像 SummaryIndex 可以完全等到 as_query_engine() 才指定；沒傳的話會 fallback 去找
    # Settings.llm（預設 OpenAI），這個專案沒有設定全域 Settings.llm、也沒有 OPENAI_API_KEY，
    # 所以載入時就要明確傳入 summary_llm（跟 index.py 建索引時用的模型一致）
    keyword_index = load_index_from_storage(
        storage_context, index_id=KEYWORD_INDEX_ID, llm=summary_llm
    )

    # DocumentSummaryIndex 的摘要向量存在 Milvus 摘要 collection，docstore／index_store
    # 借用 storage_context 讀回的同一份物件（不是複製），vector_store 換成 summary_vector_store——
    # 對應 index.py 建索引時同樣的組法，才能讀回一致的結構
    doc_summary_storage_context = StorageContext.from_defaults(
        docstore=storage_context.docstore,
        index_store=storage_context.index_store,
        vector_store=summary_vector_store,
    )
    # DocumentSummaryIndex.__init__ 同樣會在建構時 resolve self._llm = llm or Settings.llm，
    # 跟 KeywordTableIndex 一樣要明確傳入，這裡對應 index.py 建索引時用的 summary_llm
    doc_summary_index = load_index_from_storage(
        doc_summary_storage_context,
        index_id=DOC_SUMMARY_INDEX_ID,
        llm=summary_llm,
        embed_model=embed_model,
    )

    # VectorStoreIndex 的 chunk 向量存在 Milvus（外部），不需要從本機 storage_context 讀回，
    # 直接接上既有 collection 即可
    vector_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    tools = _wrap_tools(
        summary_index=summary_index,
        vector_index=vector_index,
        doc_summary_index=doc_summary_index,
        keyword_index=keyword_index,
        llm=llm,
        summary_llm=summary_llm,
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


# ── 檢索過往偏好 ─────────────────────────────────────
def retrieve_preferences(router_engine, query: str) -> str:
    # 把選路結果印出來，是為了讓終端機看得到 Router 實際選了哪種索引，
    # 而不是只看到最後一段摘要文字
    print("🔍 RouterQueryEngine 檢索中...")
    rag_response = router_engine.query(query)

    # 顯示 Router 選了哪條路
    # selections 是 0-based，但 LLM 的 reason 文字用 1-based（choice (1)）描述，這裡統一轉成 1-based 並附上工具名稱，避免「選 0 卻說選 1」的混淆
    # tool_names 順序對應本檔案 _wrap_tools() 的回傳順序（summary/vector/doc_summary/keyword），兩者要一起改
    selector_result = (rag_response.metadata or {}).get("selector_result")
    if selector_result:
        tool_names = {
            1: "SummaryIndex（整體偏好）",
            2: "VectorStoreIndex（特定細節）",
            3: "DocumentSummaryIndex（整趟紀錄回顧）",
            4: "KeywordTableIndex（精確名稱命中）",
        }
        for sel in selector_result.selections:
            choice = sel.index + 1
            print(f"📋 Router 選路結果：choice ({choice}) {tool_names.get(choice, '')}")
            print(f"   理由：{sel.reason}")

    # 顯示檢索到的偏好摘要
    rag_text = str(rag_response)
    print(f"📋 偏好摘要：{rag_text[:200]}...")

    return rag_text
