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
    2. 建立 LLM 實例（OpenAI-compatible）與 Embedding Model（NVIDIA）
    3. 建立 SummaryIndex（聚合型問題）
    4. 建立 VectorStoreIndex + Milvus（細節型問題）
    5. 將兩個索引包成 QueryEngineTool，寫明各自適合的問題類型
    6. 透過 RouterQueryEngine + LLMSingleSelector 自動選路

此模組提供 build_router_query_engine() 函式供 main.py 呼叫。
"""

# 載入套件
import logging
import os

from llama_index.core import (
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

    # ── RouterQueryEngine：LLM 讀取問題與 description 後自動選路 ──
    # selector 會把使用者問題連同上面兩個工具的 description 一起交給 LLM，
    # description 是 LLM 選路時唯一讀到的判斷依據：
    #   總覽型問題 → summary_tool（SummaryIndex，綜觀全部紀錄做摘要）
    #   檢索型問題 → vector_tool（VectorStoreIndex，top-k 相似檢索）
    router_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(llm=llm),  # 選路用的 LLM：讀問題與 description，一次選出一個 tool
        query_engine_tools=[summary_tool, vector_tool],          # 可選的工具清單
        llm=llm,                                                 # 合成用的 LLM：把選中 tool 檢索出的結果整理成最終回應
        verbose=False,                                           # 關掉內建選路 print，改由 chat.py 統一輸出一行
    )

    # 壓掉 LlamaIndex router 內部的 INFO log（同樣會印 "Selecting query engine N"），避免重複
    logging.getLogger("llama_index.core.query_engine.router_query_engine").setLevel(
        logging.WARNING
    )

    print("✅ RouterQueryEngine 建立完成（SummaryIndex + VectorStoreIndex）")
    return router_engine
