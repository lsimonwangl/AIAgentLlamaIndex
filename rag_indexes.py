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

import os
import re

from llama_index.core import (
    Document,
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent


# ── 切分設定 ──
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50


def load_data_docs():
    """讀取 ./data 資料夾中的文字檔，轉成 LlamaIndex Document 物件列表。"""
    reader = SimpleDirectoryReader(
        input_dir="./data",
        required_exts=[".txt"],
    )
    return reader.load_data()


def build_graph_docs(documents):
    """為每筆紀錄（一檔一筆）補上旅次與旅伴 metadata（僅供建圖使用）。

    旅伴欄位只在紀錄開頭，後段 chunk 抽三元組時看不到「這是哪一次旅次、
    和誰同行」；把檔名（旅次名）與旅伴放進 metadata，抽取時每個 chunk
    都帶著這兩項上下文，旅次節點命名才會一致、圖才接得起來。
    """
    records = []
    for doc in documents:
        title = os.path.splitext(doc.metadata.get("file_name", ""))[0] or "未知旅次"
        companion = re.search(r"同行人數：(.+)", doc.text)
        records.append(
            Document(
                text=doc.text.strip(),
                metadata={
                    "旅遊紀錄": title,
                    "同行旅伴": companion.group(1).strip() if companion else "未知",
                },
            )
        )
    return records


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


class RecordPathExtractor(TransformComponent):
    """規則抽取器：從 metadata 直接產生三元組，不呼叫 LLM。

    語料是半結構化的：旅伴寫在「同行人數：」固定欄位、旅次名是檔名、
    目的地是檔名中段，build_graph_docs() 已把旅次與旅伴放進每個 chunk
    的 metadata，這裡照規則建 (旅次)-[同行]->(旅伴) 與 (旅次)-[造訪]->(目的地)。
    每個 chunk 都掛在自己的旅次節點下，查詢時沿 旅伴→旅次 把原文 chunk
    帶回來整理——細節（景點/住宿/美食）由原文提供，不需要細粒度三元組。
    """

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            trip = EntityNode(name=node.metadata.get("旅遊紀錄", "未知旅次"), label="旅次")
            mate = EntityNode(name=node.metadata.get("同行旅伴", "未知"), label="旅伴")
            kg_nodes = [trip, mate]
            relations = [Relation(source_id=trip.id, target_id=mate.id, label="同行")]

            # 檔名格式「編號_目的地_年月」，取中段當目的地節點
            parts = trip.name.split("_")
            if len(parts) >= 2:
                dest = EntityNode(name=parts[1], label="目的地")
                kg_nodes.append(dest)
                relations.append(Relation(source_id=trip.id, target_id=dest.id, label="造訪"))

            node.metadata[KG_NODES_KEY] = kg_nodes
            node.metadata[KG_RELATIONS_KEY] = relations
        return nodes


def build_graph_index(documents, splitter, llm, embed_model):
    """建立 PropertyGraphIndex：沿 旅伴→旅次 聚合「旅伴情境」偏好。

    規則抽取不需 LLM，建圖只花節點嵌入的幾秒鐘，每次啟動重建、
    永遠與 ./data 同步；llm 供查詢時的同義詞擴展使用。
    """
    print("🕸️ 建立 PropertyGraphIndex（規則抽取，不需 LLM）...")
    return PropertyGraphIndex.from_documents(
        build_graph_docs(documents),                      # 每筆紀錄的 metadata 帶旅次與旅伴
        llm=llm,
        embed_model=embed_model,
        kg_extractors=[RecordPathExtractor()],
        property_graph_store=SimplePropertyGraphStore(),  # 內建記憶體圖存放區，不需外部服務
        transformations=[splitter],
        use_async=False,  # 全同步：避免在 main.py 的 event loop 內巢狀 asyncio.run()
        show_progress=True,
    )
