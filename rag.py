"""
Router RAG - 旅遊偏好檢索器
===========================
rag.py 負責將 ./data 中的旅遊紀錄建立三種索引，
並透過 RouterQueryEngine 依問題類型自動選擇檢索方式。

相較於 Lab2 只使用單一 VectorStoreIndex，Lab3 新增 SummaryIndex 與
PropertyGraphIndex，讓系統能針對不同類型的問題選擇最適合的檢索策略：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節
    - PropertyGraphIndex：知識圖譜檢索，適合按「旅伴情境」聚合偏好

執行流程：
    0. 載入套件與環境變數
    1. 從 ./data 讀取旅遊紀錄文字檔
    2. 建立 LLM 實例（OpenAI-compatible）與 Embedding Model（NVIDIA）
    3. 建立 SummaryIndex（聚合型問題）
    4. 建立 VectorStoreIndex + Milvus（細節型問題）
    5. 建立 PropertyGraphIndex（旅伴情境偏好聚合）
    6. 將三個索引包成 QueryEngineTool，寫明各自適合的問題類型
    7. 透過 RouterQueryEngine + LLMSingleSelector 自動選路

此模組提供 build_router_query_engine() 函式供 main.py 呼叫。
"""

# 載入套件
import logging
import os
import re
from typing import Literal

from llama_index.core import (
    Document,
    PromptTemplate,
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore


# ── PropertyGraphIndex 固定 schema ──
# 旅伴欄位只在每份紀錄開頭、偏好細節散在後文，chunk 256 切分後兩者
# 不在同一 chunk，向量檢索無法做「旅伴條件」的偏好聚合；
# 因此用固定 schema 抽三元組，把 旅次-旅伴 與 旅次-景點/住宿/美食/評價
# 串在同一張圖上，讓「和某類旅伴出遊」的偏好可以沿著圖聚合
GRAPH_ENTITIES = Literal["旅次", "旅伴", "目的地", "景點", "住宿", "美食", "評價"]
GRAPH_RELATIONS = Literal["同行", "造訪", "入住", "品嚐", "評價為"]
GRAPH_VALIDATION_SCHEMA = [
    ("旅次", "同行", "旅伴"),
    ("旅次", "造訪", "目的地"),
    ("旅次", "造訪", "景點"),
    ("旅次", "入住", "住宿"),
    ("旅次", "品嚐", "美食"),
    ("目的地", "評價為", "評價"),
    ("景點", "評價為", "評價"),
    ("住宿", "評價為", "評價"),
    ("美食", "評價為", "評價"),
]


def load_data_docs():
    """讀取 ./data 資料夾中的文字檔，轉成 LlamaIndex Document 物件列表。"""
    reader = SimpleDirectoryReader(
        input_dir="./data",
        required_exts=[".txt"],
    )
    return reader.load_data()


def split_docs_by_record(documents):
    """把整份合集依「===== 標題 =====」切成一筆旅遊紀錄一個 Document（僅供建圖使用）。

    旅伴欄位只在每份紀錄開頭，後段 chunk 抽三元組時看不到「這是哪一次旅次、
    和誰同行」；把紀錄標題與旅伴放進 metadata，抽取時每個 chunk 都帶著
    這兩項上下文，旅次節點命名才會一致、圖才接得起來。
    """
    records = []
    for doc in documents:
        parts = re.split(r"^=====\s*(.+?)\s*=====\s*$", doc.text, flags=re.MULTILINE)
        # re.split 結果為 [前導, 標題1, 內文1, 標題2, 內文2, ...]
        for title, body in zip(parts[1::2], parts[2::2]):
            companion = re.search(r"同行人數：(.+)", body)
            records.append(
                Document(
                    text=body.strip(),
                    metadata={
                        "旅遊紀錄": title,
                        "同行旅伴": companion.group(1).strip() if companion else "未知",
                    },
                )
            )
    # ponytail: 語料若無 ===== 標題就退回原始 Document，不擋建圖
    return records or documents


def build_router_query_engine():
    """建立 RouterQueryEngine，依問題類型自動在 SummaryIndex 與 VectorStoreIndex 之間選路。"""

    # 建立 LLM 實例（透過 OpenAI-compatible 端點呼叫 NVIDIA NIM）
    llm = OpenAILike(
        api_base="https://integrate.api.nvidia.com/v1",  # NVIDIA NIM 的 OpenAI 相容端點
        api_key=os.getenv("NVIDIA_API_KEY"),             # 從環境變數讀取 API 金鑰
        model=os.getenv("CHAT_MODEL"),                   # 指定使用的 chat 模型名稱
        is_chat_model=True,                              # 使用 chat completion 介面
        is_function_calling_model=True,                  # 啟用 function calling
        context_window=128000,                           # 模型最大可吃 token 數
        timeout=300.0,                                   # 請求逾時秒數
    )

    # 建立 Embedding Model 實例
    embed_model = NVIDIAEmbedding(model=os.getenv("EMBEDDING_MODEL"))

    # 設定文字切分器
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

    print("🔨 讀取 ./data 旅遊紀錄")
    documents = load_data_docs()

    # ── 建立 SummaryIndex：適合聚合型問題 ──
    # 查詢時會掃過所有 chunk 做 tree_summarize，能從全部紀錄中歸納整體偏好
    print("📋 建立 SummaryIndex...")
    summary_index = SummaryIndex.from_documents(
        documents,                    # 從 ./data 讀進來的 Document
        transformations=[splitter],   # 建索引前先用 SentenceSplitter 切成 chunk
    )

    # ── 建立 VectorStoreIndex + Milvus：適合細節型問題 ──
    # 將 chunk 向量化存入 Milvus，查詢時用 cosine similarity 找最相關的片段
    print("🔢 建立 VectorStoreIndex + Milvus...")
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530",          # 本地 Milvus 服務位址
        collection_name="travel_preferences",  # 設定 Milvus collection 名稱
        dim=2048,                              # 向量維度，需與 embedding 模型輸出一致
        overwrite=True,                        # 每次啟動覆寫，確保資料與 ./data 同步
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)  # 把 Milvus 注入儲存層
    vector_index = VectorStoreIndex.from_documents(
        documents,                       # 從 ./data 讀進來的 Document
        storage_context=storage_context, # 指定向量存到 Milvus 而非預設記憶體
        transformations=[splitter],      # 建索引前先用 SentenceSplitter 切 chunk
        embed_model=embed_model,         # 用來把 chunk 轉成向量的模型
    )

    # ── 建立 PropertyGraphIndex：適合旅伴情境的偏好聚合 ──
    # 用同一顆 LLM 依固定 schema 抽三元組，圖存進內建 SimplePropertyGraphStore
    print("🕸️ 建立 PropertyGraphIndex（LLM 逐 chunk 抽取三元組，速度較慢，請稍候）...")
    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,                                       # 與其他索引共用同一顆 LLM
        possible_entities=GRAPH_ENTITIES,              # 實體類型限定為固定 schema
        possible_relations=GRAPH_RELATIONS,            # 關係類型限定為固定 schema
        kg_validation_schema=GRAPH_VALIDATION_SCHEMA,  # 只允許 schema 內的三元組組合
        strict=True,                                   # 不符合 schema 的抽取結果直接丟棄
    )
    graph_index = PropertyGraphIndex.from_documents(
        split_docs_by_record(documents),               # 一筆紀錄一個 Document，metadata 帶旅次與旅伴
        llm=llm,                                       # 圖譜檢索（同義詞擴展）用的 LLM
        embed_model=embed_model,                       # 圖節點嵌入用的 embedding model
        kg_extractors=[kg_extractor],                  # 用上面的固定 schema 抽取器建圖
        property_graph_store=SimplePropertyGraphStore(),  # 內建記憶體圖存放區，不需外部服務
        transformations=[splitter],                    # 與其他索引相同的切分設定
        show_progress=True,                            # 顯示抽取與嵌入進度條（建圖較慢）
    )
    print("🕸️ PropertyGraphIndex 建立完成")

    # ── 自訂 QA prompt：把 RAG 從「直接回答問題」改為「整理過往台灣經驗作為素材」 ──
    # 這樣即使使用者問海外目的地（例：京都有溪谷步道嗎），RAG 不會回「無京都資料」，
    # 而是回傳使用者過往在台灣相關的具體經驗，供後續 Agent + Tavily 規劃使用
    organize_qa_tmpl = PromptTemplate(
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

    # ── summary_tool：把 SummaryIndex 包成 QueryEngineTool，寫明適合的問題類型 ──
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            llm=llm,                            # 用前面建立的 NVIDIA NIM LLM
            response_mode="tree_summarize",     # 樹狀摘要：所有 node 分組各做局部摘要，再把摘要當新內容反覆向上合併成一份總結
            summary_template=organize_qa_tmpl,   # 套用自訂的「整理式」prompt
        ),
        # description 用一段話描述這個工具適合處理的問題類型，
        # 這裡描述總覽型問題的特徵：需要綜觀全部旅遊紀錄才能歸納的素材
        description=(
            "適合回答文件整體內容、跨文件摘要、主題總覽與綜合分析等總覽型問題，"
            "如歸納使用者整體旅遊偏好、旅行風格，或跨紀錄的統計與聚合（平均花費、預算結構）。"
            "例如：歸納我過去旅行偏好的行程節奏，安排日本行程時一天排幾個景點比較適合我。"
        ),
    )

    # ── vector_tool：把 VectorStoreIndex 包成 QueryEngineTool，寫明適合的問題類型 ──
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(
            llm=llm,                            # 用前面建立的 NVIDIA NIM LLM
            similarity_top_k=5,                 # 取相似度最高的 5 個 chunk
            text_qa_template=organize_qa_tmpl,   # 套用自訂的「整理式」prompt
        ),
        # 同樣用 description 描述問題類型，
        # 這裡描述檢索型問題的特徵：只聚焦某次具體經歷、查特定細節的素材
        description=(
            "適合回答特定景點資訊、行程細節與實際建議等需要精確比對的具體型問題，"
            "以語意相似度找出某次旅行的景點體驗、美食評價、住宿細節等對應的文件片段。"
            "例如：日本有沒有類似我在花蓮走過那條沿溪步道的健行路線。"
        ),
    )

    # ── graph_tool：把 PropertyGraphIndex 包成 QueryEngineTool，寫明適合的問題類型 ──
    graph_tool = QueryEngineTool.from_defaults(
        query_engine=graph_index.as_query_engine(
            llm=llm,                            # 用前面建立的 NVIDIA NIM LLM
            include_text=True,                  # 命中三元組後帶回原文 chunk 供整理
            path_depth=2,                       # 沿圖走兩步：旅伴→旅次→景點/住宿/美食
            similarity_top_k=8,                 # 錨點節點取 8 個，聚合型問題需要較廣的起點
            text_qa_template=organize_qa_tmpl,   # 套用同一份「整理式」prompt
        ),
        # 與另外兩個工具明確區隔：這裡只負責「和某類旅伴出遊」的情境偏好聚合
        description=(
            "適合回答「和某類旅伴出遊時」的情境偏好聚合問題，"
            "依旅伴（獨旅一個人、和朋友、和女友）分組歸納各自的"
            "景點類型、住宿、美食與步調偏好。"
            "例如：照我過去獨旅的偏好規劃行程、和朋友出遊時我喜歡住哪類住宿。"
        ),
    )

    # ── RouterQueryEngine：LLM 讀取問題與 description 後自動選路 ──
    # selector 會把使用者問題連同上面三個工具的 description 一起交給 LLM，
    # description 是 LLM 選路時唯一讀到的判斷依據：
    #   總覽型問題 → summary_tool（SummaryIndex，綜觀全部紀錄做摘要）
    #   檢索型問題 → vector_tool（VectorStoreIndex，top-k 相似檢索）
    #   旅伴情境偏好 → graph_tool（PropertyGraphIndex，沿圖聚合偏好）
    router_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(llm=llm),  # 選路用的 LLM：讀問題與 description，一次選出一個 tool
        query_engine_tools=[summary_tool, vector_tool, graph_tool],  # 可選的工具清單
        llm=llm,                                                 # 合成用的 LLM：把選中 tool 檢索出的結果整理成最終回應
        verbose=False,                                           # 關掉內建選路 print，改由 chat.py 統一輸出一行
    )

    # 壓掉 LlamaIndex router 內部的 INFO log（同樣會印 "Selecting query engine N"），避免重複
    logging.getLogger("llama_index.core.query_engine.router_query_engine").setLevel(
        logging.WARNING
    )

    print("✅ RouterQueryEngine 建立完成（SummaryIndex + VectorStoreIndex + PropertyGraphIndex）")
    return router_engine
