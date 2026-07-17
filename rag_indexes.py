"""
Router RAG - 索引建構
=====================
rag_indexes.py 負責把 ./data 的旅遊紀錄讀入，並建立三種索引：
    - SummaryIndex：掃過所有紀錄做摘要，適合歸納整體旅遊風格
    - VectorStoreIndex：向量相似度檢索，適合查詢特定體驗細節
    - PropertyGraphIndex：知識圖譜檢索，適合按「旅伴情境」聚合偏好

每個 build_*_index() 只負責「documents → index」，所需的 client 物件
（llm、embed_model、vector_store）一律由呼叫端（rag.py）透過
rag_clients.py 建立後傳入，本檔案不處理連線設定。
"""

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from llama_index.core import (
    Document,
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter


# ── 切分設定 ──
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

# ── PropertyGraphIndex 固定 schema ──
# 旅伴欄位只在每份紀錄開頭、偏好細節散在後文，chunk 256 切分後兩者
# 不在同一 chunk，向量檢索無法做「旅伴條件」的偏好聚合；
# 因此用固定 schema 抽三元組，把 旅次-旅伴 與 旅次-景點/住宿/美食/評價
# 串在同一張圖上，讓「和某類旅伴出遊」的偏好可以沿著圖聚合
# 建圖成本高（逐 chunk LLM 抽取），建好後持久化到此資料夾；
# ponytail: 語料（./data）變更時請手動刪除此資料夾觸發重建
GRAPH_PERSIST_DIR = "./storage_graph"
GRAPH_EXTRACT_MAX_RETRIES = 8  # NVIDIA endpoint 對請求量敏感（503），加大重試次數靠指數退避慢慢送完

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
    """把語料整理成一筆旅遊紀錄一個 Document，並補上旅次與旅伴 metadata（僅供建圖使用）。

    旅伴欄位只在每份紀錄開頭，後段 chunk 抽三元組時看不到「這是哪一次旅次、
    和誰同行」；把紀錄標題與旅伴放進 metadata，抽取時每個 chunk 都帶著
    這兩項上下文，旅次節點命名才會一致、圖才接得起來。

    支援兩種語料形態：一檔一筆紀錄（旅次名取自檔名），
    或含「===== 標題 =====」分隔的合集檔（依標題切分）。
    """
    records = []
    for doc in documents:
        parts = re.split(r"^=====\s*(.+?)\s*=====\s*$", doc.text, flags=re.MULTILINE)
        if len(parts) > 1:
            # 合集檔：re.split 結果為 [前導, 標題1, 內文1, 標題2, 內文2, ...]
            titled_bodies = list(zip(parts[1::2], parts[2::2]))
        else:
            # 一檔一筆紀錄：旅次名取自檔名（去掉副檔名）
            file_name = doc.metadata.get("file_name", "")
            title = os.path.splitext(file_name)[0] or "未知旅次"
            titled_bodies = [(title, doc.text)]

        for title, body in titled_bodies:
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
    return records or documents


def build_splitter():
    """建立節點切分器，供三種索引共用同一套切分設定。"""
    return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def build_summary_index(documents, splitter):
    """建立 SummaryIndex：查詢時掃過所有 chunk 做 tree_summarize，適合聚合型問題。"""
    print("📋 建立 SummaryIndex...")
    return SummaryIndex.from_documents(documents, transformations=[splitter])


def build_vector_index(documents, splitter, embed_model, vector_store):
    """建立 VectorStoreIndex：把傳入的 vector_store（Milvus）包進 StorageContext 後向量化寫入。"""
    print("🔢 建立 VectorStoreIndex + Milvus...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        embed_model=embed_model,
    )


def build_graph_index(documents, splitter, llm, embed_model):
    """建立或載入 PropertyGraphIndex：沿圖聚合「旅伴情境」偏好，建圖成本高故持久化到 GRAPH_PERSIST_DIR。"""
    if os.path.exists(GRAPH_PERSIST_DIR):
        print(f"🕸️ 載入既有 PropertyGraphIndex（{GRAPH_PERSIST_DIR}；語料變更請刪除此資料夾重建）")
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=GRAPH_PERSIST_DIR),
            llm=llm,
            embed_model=embed_model,
        )

    # 建圖抽取可用 GRAPH_CHAT_MODEL 指定獨立模型（如 moonshotai/kimi-k2.6），
    # 與主 LLM 的 worker 配額隔離，建圖不會跟選路/合成/Agent 搶額度；
    # 未設定則沿用 CHAT_MODEL。抽取靠 function calling，指定的模型必須支援 tools
    kg_llm_updates = {"max_retries": GRAPH_EXTRACT_MAX_RETRIES}
    if os.getenv("GRAPH_CHAT_MODEL"):
        kg_llm_updates["model"] = os.getenv("GRAPH_CHAT_MODEL")

    print(f"🕸️ 建立 PropertyGraphIndex（{kg_llm_updates.get('model', llm.model)} 逐 chunk 抽取三元組，速度較慢，請稍候）...")
    kg_extractor = SchemaLLMPathExtractor(
        llm=llm.model_copy(update=kg_llm_updates),
        possible_entities=GRAPH_ENTITIES,
        possible_relations=GRAPH_RELATIONS,
        kg_validation_schema=GRAPH_VALIDATION_SCHEMA,
        strict=True,     # 不符合 schema 的抽取結果直接丟棄
        num_workers=1,   # 逐一送出：併發重試風暴反而會撐爆 endpoint 的請求計數
    )

    def _extract():
        return PropertyGraphIndex.from_documents(
            split_docs_by_record(documents),                  # 一筆紀錄一個 Document，metadata 帶旅次與旅伴
            llm=llm,
            embed_model=embed_model,
            kg_extractors=[kg_extractor],
            property_graph_store=SimplePropertyGraphStore(),  # 內建記憶體圖存放區，不需外部服務
            transformations=[splitter],
            show_progress=True,
        )

    # 建圖內部會呼叫 asyncio.run()，但 main.py 以 asyncio.run(main()) 進入時
    # event loop 已在跑、不允許巢狀呼叫；丟到獨立 thread 讓它有自己的 loop
    # （不用 nest_asyncio：它在 Python 3.14 會打壞 anyio，MCP 工具會載入失敗）
    try:
        asyncio.get_running_loop()
        in_running_loop = True
    except RuntimeError:
        in_running_loop = False

    if in_running_loop:
        with ThreadPoolExecutor(max_workers=1) as pool:
            graph_index = pool.submit(_extract).result()
    else:
        graph_index = _extract()

    graph_index.storage_context.persist(persist_dir=GRAPH_PERSIST_DIR)
    print(f"🕸️ PropertyGraphIndex 建立完成，已存至 {GRAPH_PERSIST_DIR}")
    return graph_index
