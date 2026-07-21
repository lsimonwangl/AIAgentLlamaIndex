"""
Router RAG - 旅遊偏好檢索器
===========================
rag.py 負責組裝 RouterQueryEngine：從 rag_clients.py 取得模型與 Milvus
連線、從 rag_indexes.py 取得三種索引，包成 QueryEngineTool 後交由
RouterQueryEngine 依問題類型自動選擇檢索方式。

相較於 Lab2 只使用單一 VectorStoreIndex，Lab3 新增 SummaryIndex 與
PropertyGraphIndex，讓系統能針對不同類型的問題選擇最適合的檢索策略：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節
    - PropertyGraphIndex：知識圖譜檢索，適合按「旅伴情境」聚合偏好

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
GRAPH_TOP_K = 8       # 聚合型問題需要較廣的錨點起點
GRAPH_PATH_DEPTH = 2  # 沿圖走兩步：旅伴→旅行事件→目的地

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


def _build_tools(summary_index, vector_index, graph_index, llm):
    """把三個索引包成 QueryEngineTool；description 是 RouterQueryEngine 選路時唯一的判斷依據。"""
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

    graph_tool = QueryEngineTool.from_defaults(
        query_engine=graph_index.as_query_engine(
            llm=llm,
            include_text=True,           # 命中三元組後帶回原文 chunk 供整理
            path_depth=GRAPH_PATH_DEPTH,
            similarity_top_k=GRAPH_TOP_K,
            use_async=False,             # 走同步檢索路徑，避免在 main.py 的 event loop 內巢狀 asyncio.run()
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答「和某類旅伴出遊時」的情境偏好聚合問題，"
            "依旅伴（獨旅一個人、和朋友、和女友、和家人爸媽）分組歸納各自的"
            "景點類型、住宿、美食與步調偏好。"
            "例如：照我過去獨旅的偏好規劃行程、和朋友出遊時我喜歡住哪類住宿、"
            "帶爸媽出門要注意什麼。"
        ),
    )

    return [summary_tool, vector_tool, graph_tool]


def build_router_query_engine():
    """建立 RouterQueryEngine，依問題類型自動在 SummaryIndex/VectorStoreIndex/PropertyGraphIndex 之間選路。"""
    llm = rag_clients.build_llm()
    embed_model = rag_clients.build_embed_model()
    vector_store = rag_clients.build_milvus_vector_store()
    splitter = rag_indexes.build_splitter()

    print("🔨 讀取 ./data 旅遊紀錄")
    documents = rag_indexes.load_data_docs()

    summary_index = rag_indexes.build_summary_index(documents, splitter)
    vector_index = rag_indexes.build_vector_index(documents, splitter, embed_model, vector_store)
    graph_index = rag_indexes.build_graph_index(documents, splitter, llm, embed_model)

    tools = _build_tools(summary_index, vector_index, graph_index, llm)

    # selector 會把使用者問題連同上面三個工具的 description 一起交給 LLM，
    # description 是 LLM 選路時唯一讀到的判斷依據：
    #   總覽型問題 → summary_tool（SummaryIndex，綜觀全部紀錄做摘要）
    #   檢索型問題 → vector_tool（VectorStoreIndex，top-k 相似檢索）
    #   旅伴情境偏好 → graph_tool（PropertyGraphIndex，沿圖聚合偏好）
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

    print("✅ RouterQueryEngine 建立完成（SummaryIndex + VectorStoreIndex + PropertyGraphIndex）")
    return router_engine
