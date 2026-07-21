"""
Router RAG - 旅遊偏好檢索器
===========================
rag.py 負責組裝 RouterQueryEngine：從 rag_clients.py 取得模型與 Milvus
連線、從 rag_indexes.py 取得三種索引，包成 QueryEngineTool 後交由
RouterQueryEngine 依問題類型自動選擇檢索方式。

相較於 Lab2 只使用單一 VectorStoreIndex，Lab3 新增 SummaryIndex 與
DocumentSummaryIndex，讓系統能針對不同類型的問題選擇最適合的檢索策略：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節
    - DocumentSummaryIndex：以每篇文件摘要為檢索單位，選出最相關的整趟紀錄

執行流程：
    0. 載入套件與環境變數
    1. 透過 rag_clients 建立 LLM、Embedding Model、Milvus 連線
    2. 透過 rag_indexes 讀取 ./data 並建立三個索引
    3. 將三個索引包成 QueryEngineTool，寫明各自適合的問題類型
    4. 透過 RouterQueryEngine + LLMSingleSelector 自動選路

此模組提供 build_router_query_engine() 函式供 main.py 呼叫。
"""

import logging

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool

import rag_clients
import rag_indexes

# ── 檢索設定常數 ──
VECTOR_TOP_K = 5
DOC_SUMMARY_TOP_K = 3  # 以文件摘要相似度挑出最相關的 3 趟旅行紀錄

# ── 自訂 QA prompt：把 RAG 從「直接回答問題」改為「整理過往台灣經驗作為素材」 ──
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


def _build_tools(summary_index, vector_index, doc_summary_index, llm, summary_llm):
    """把三個索引包成 QueryEngineTool；description 是 RouterQueryEngine 選路時唯一的判斷依據。

    summary_llm 是 DocumentSummaryIndex 專用的便宜快速模型，只用於該路由。
    """
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            llm=llm,
            response_mode="tree_summarize",  # 樹狀摘要：分組局部摘要，再逐層向上合併成總結
            summary_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答文件整體內容、跨文件摘要、主題總覽與綜合分析等總覽型問題，"
            "如歸納使用者整體旅遊偏好、旅行風格，或跨紀錄的統計與聚合（平均花費、預算結構）。"
            "例如：歸納我過去旅行偏好的行程節奏，安排日本行程時一天排幾個景點比較適合我。"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(
            llm=llm,
            similarity_top_k=VECTOR_TOP_K,
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答特定景點資訊、行程細節與實際建議等需要精確比對的具體型問題，"
            "以語意相似度找出某次旅行的景點體驗、美食評價、住宿細節等對應的文件片段。"
            "例如：日本有沒有類似我在花蓮走過那條沿溪步道的健行路線。"
        ),
    )

    doc_summary_tool = QueryEngineTool.from_defaults(
        query_engine=doc_summary_index.as_query_engine(
            llm=summary_llm,              # 便宜快速模型：合成也用它
            retriever_mode="embedding",   # 以文件摘要的向量相似度挑文件（非每次 LLM 選文件）
            similarity_top_k=DOC_SUMMARY_TOP_K,
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合「以整趟旅行為單位」找出與提問最相關的幾筆完整旅遊紀錄，"
            "先比對每篇文件的摘要挑出最相關的旅行，再帶回那幾趟的完整內容回顧，"
            "定位介於 VectorStoreIndex（片段級細節）與 SummaryIndex（全讀總覽）之間。"
            "例如：回顧我最相關的幾趟山區旅行、哪幾趟旅行的整體規劃和這次最像。"
        ),
    )

    return [summary_tool, vector_tool, doc_summary_tool]


def build_router_query_engine():
    """建立 RouterQueryEngine，依問題類型自動在 SummaryIndex/VectorStoreIndex/DocumentSummaryIndex 之間選路。"""
    llm = rag_clients.build_llm()
    summary_llm = rag_clients.build_summary_llm()  # DocumentSummaryIndex 專用的便宜快速模型
    embed_model = rag_clients.build_embed_model()
    vector_store = rag_clients.build_milvus_vector_store()
    splitter = rag_indexes.build_splitter()

    print("🔨 讀取 ./data 旅遊紀錄")
    documents = rag_indexes.load_data_docs()

    summary_index = rag_indexes.build_summary_index(documents, splitter)
    vector_index = rag_indexes.build_vector_index(documents, splitter, embed_model, vector_store)
    doc_summary_index = rag_indexes.build_document_summary_index(documents, splitter, summary_llm, embed_model)

    tools = _build_tools(summary_index, vector_index, doc_summary_index, llm, summary_llm)

    # selector 會把使用者問題連同上面三個工具的 description 一起交給 LLM，
    # description 是 LLM 選路時唯一讀到的判斷依據：
    #   總覽型問題 → summary_tool（SummaryIndex，綜觀全部紀錄做摘要）
    #   檢索型問題 → vector_tool（VectorStoreIndex，top-k 相似檢索）
    #   整趟紀錄回顧 → doc_summary_tool（DocumentSummaryIndex，以摘要挑整篇文件）
    router_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        llm=llm,        # 合成用的 LLM：把選中 tool 檢索出的結果整理成最終回應
        verbose=False,  # 關掉內建選路 print，改由 chat.py 統一輸出一行
    )

    # 壓掉 LlamaIndex router 內部的 INFO log（同樣會印 "Selecting query engine N"），避免重複
    logging.getLogger("llama_index.core.query_engine.router_query_engine").setLevel(
        logging.WARNING
    )

    print("✅ RouterQueryEngine 建立完成（SummaryIndex + VectorStoreIndex + DocumentSummaryIndex）")
    return router_engine
