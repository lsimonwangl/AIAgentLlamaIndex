"""臨時腳本：用 GRAPH_CHAT_MODEL 重建 PropertyGraphIndex 並驗證圖譜查詢。"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

import rag_clients
import rag_indexes
from rag import GRAPH_PATH_DEPTH, GRAPH_TOP_K, ORGANIZE_QA_TEMPLATE

llm = rag_clients.build_llm()
embed_model = rag_clients.build_embed_model()
splitter = rag_indexes.build_splitter()
documents = rag_indexes.load_data_docs()

graph_index = rag_indexes.build_graph_index(
    documents, splitter, llm, embed_model, rag_clients.build_graph_llm()
)

print("\n--- 同行 三元組（應涵蓋 20 筆旅次） ---")
triplets = graph_index.property_graph_store.get_triplets(relation_names=["同行"])
for subj, rel, obj in sorted(triplets, key=lambda t: t[0].name):
    print(f"({subj.name}) -[{rel.label}]-> ({obj.name})")

query_engine = graph_index.as_query_engine(
    llm=llm,
    include_text=True,
    path_depth=GRAPH_PATH_DEPTH,
    similarity_top_k=GRAPH_TOP_K,
    use_async=False,
    text_qa_template=ORGANIZE_QA_TEMPLATE,
)
for question in [
    "照我過去獨旅的偏好規劃行程",
    "帶爸媽出遊時我有什麼偏好要注意",
]:
    print(f"\n--- Q: {question} ---")
    print(query_engine.query(question))
