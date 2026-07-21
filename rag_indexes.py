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


# ── 解析旅遊紀錄：程式腳本方式抽取「旅行事件」與「旅伴」 ──────────
def build_graph_docs(documents):
    """為每筆紀錄（一檔一筆）補上旅行事件與旅伴 metadata（僅供建圖使用）。

    旅伴欄位只在紀錄開頭，後段 chunk 抽三元組時看不到「這是哪一次旅行事件、
    和誰同行」；把檔名（旅行事件名）與旅伴放進 metadata，抽取時每個 chunk
    都帶著這兩項上下文，旅行事件節點命名才會一致、圖才接得起來。
    """
    # 存放整理後的紀錄 Document
    records = []
    # 逐份文件處理（一檔一筆紀錄）
    for doc in documents:
        # 旅行事件名稱取自檔名（去掉副檔名），如「03_台南_2025年1月」
        title = os.path.splitext(doc.metadata.get("file_name", ""))[0] or "未知旅行事件"
        # 從內文抓固定欄位「同行人數：」的值，如「1人（自己去）」
        companion = re.search(r"同行人數：(.+)", doc.text)
        # 重建 Document：原文不變，metadata 換成旅行事件與旅伴兩個欄位
        records.append(
            Document(
                # 保留完整原文（去除前後空白）
                text=doc.text.strip(),
                metadata={
                    # 旅行事件名稱：旅行事件節點的唯一識別，所有 chunk 靠它掛回同一個旅行事件
                    "旅遊紀錄": title,
                    # 同行旅伴：旅伴節點的名字；欄位不存在時標為「未知」
                    "同行旅伴": companion.group(1).strip() if companion else "未知",
                },
            )
        )
    # 回傳整理後的紀錄清單
    return records


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


# ── 規則抽取器：從 metadata 直接產生「實體」與「關係」，不呼叫 LLM ──────────
class RecordPathExtractor(TransformComponent):
    """規則抽取器：從 metadata 直接產生三元組，不呼叫 LLM。

    語料是半結構化的：旅伴寫在「同行人數：」固定欄位、旅行事件名是檔名、
    目的地是檔名中段，build_graph_docs() 已把旅行事件與旅伴放進每個 chunk
    的 metadata，這裡照規則建 (旅行事件)-[同行]->(旅伴) 與 (旅行事件)-[造訪]->(目的地)。
    每個 chunk 都掛在自己的旅行事件節點下，查詢時沿 旅伴→旅行事件 把原文 chunk
    帶回來整理——細節（景點/住宿/美食）由原文提供，不需要細粒度三元組。
    """

    # 知識圖譜的節點與關係結構：
    #   - 旅行事件 節點代表一次旅行（如 03_台南_2025年1月），原文 chunk 都掛在它底下
    #   - 旅伴 節點代表同行者類型（如 1人（自己去）），是「按旅伴聚合」的查詢入口
    #   - 目的地 節點代表地點（如 台南），重訪同一地點時可跨旅行事件合併
    # 節點之間會用關係連起來，讓查詢可以沿著知識圖譜尋找答案：
    #   - (旅行事件)-[同行]->(旅伴)：表示這趟旅行和誰一起去
    #   - (旅行事件)-[造訪]->(目的地)：表示這趟旅行去了哪裡

    def __call__(self, nodes, **kwargs):
        # 逐個 chunk 產生三元組
        for node in nodes:
            # 建立旅行事件節點：名稱取自 metadata 的「旅遊紀錄」（即檔名），同名自動合併
            trip = EntityNode(name=node.metadata.get("旅遊紀錄", "未知旅行事件"), label="旅行事件")
            # 建立旅伴節點：名稱取自 metadata 的「同行旅伴」欄位
            mate = EntityNode(name=node.metadata.get("同行旅伴", "未知"), label="旅伴")
            # 本 chunk 產出的節點清單
            kg_nodes = [trip, mate]
            # 本 chunk 產出的關係清單：(旅行事件)-[同行]->(旅伴)
            relations = [Relation(source_id=trip.id, target_id=mate.id, label="同行")]

            # 檔名格式「編號_目的地_年月」，取中段建立目的地節點
            parts = trip.name.split("_")
            # 防呆：檔名不符格式時跳過目的地，不中斷建圖
            if len(parts) >= 2:
                # 建立目的地節點：名稱取自檔名中段
                dest = EntityNode(name=parts[1], label="目的地")
                kg_nodes.append(dest)
                # 建立 (旅行事件)-[造訪]->(目的地) 關係
                relations.append(Relation(source_id=trip.id, target_id=dest.id, label="造訪"))

            # 把節點與關係放進約定的 metadata 欄位（KG_NODES_KEY / KG_RELATIONS_KEY），
            # PropertyGraphIndex 建索引時會從這裡收貨、存入圖存放區，
            # 並在每個節點記錄來源 chunk id——查詢時 include_text 靠它把原文帶回
            node.metadata[KG_NODES_KEY] = kg_nodes
            node.metadata[KG_RELATIONS_KEY] = relations
        # 回傳帶著三元組的 chunk 清單
        return nodes


# ── 建立 PropertyGraphIndex：適合旅伴情境的偏好聚合 ────────
def build_graph_index(documents, splitter, llm, embed_model):
    """建立 PropertyGraphIndex：沿 旅伴→旅行事件 聚合「旅伴情境」偏好。

    規則抽取不需 LLM，建圖只花節點嵌入的幾秒鐘，每次啟動重建、
    永遠與 ./data 同步；llm 供查詢時的同義詞擴展使用。
    """
    print("🕸️ 建立 PropertyGraphIndex（規則抽取，不需 LLM）...")
    return PropertyGraphIndex.from_documents(
        # 每筆紀錄的 metadata 帶旅行事件與旅伴（規則抽取的資料來源）
        build_graph_docs(documents),
        # 查詢時同義詞擴展用的 LLM
        llm=llm,
        # 節點名稱嵌入用的 embedding model，供查詢時向量錨定
        embed_model=embed_model,
        # 規則抽取器：從 metadata 產生三元組，不呼叫 LLM
        kg_extractors=[RecordPathExtractor()],
        # 內建記憶體圖存放區，不需外部服務（相對於 Neo4j 這類圖資料庫）
        property_graph_store=SimplePropertyGraphStore(),
        # 與其他索引相同的切分設定
        transformations=[splitter],
        # 全同步執行：避免在 main.py 的 event loop 內巢狀 asyncio.run()
        use_async=False,
        # 顯示建圖進度條
        show_progress=True,
    )
