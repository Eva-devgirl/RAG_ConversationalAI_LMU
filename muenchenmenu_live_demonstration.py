
import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-proj-DwikG6Qsk0rlnchK8czNT3BlbkFJoAgxLhHtFmyLAvg8j2nJ"
openai.api_key = os.environ["OPENAI_API_KEY"]

rawdatadir = "./rawdatahtml"
indexpath = "./muenchenmenuindex_simple_filenames"


from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader
# rebuild storage context
print("Rebuilding context.")
storage_context = StorageContext.from_defaults(persist_dir=indexpath)
print("Loading index.")
# load index
index = load_index_from_storage(storage_context)
print("Creating query engine.")
query_engine = index.as_query_engine()
print("Creating Retriever.")
retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=8)
print("Node IDs")
print(index.vector_store._data.embedding_dict.keys())
allnodes=index.vector_store._data.embedding_dict.keys()
print(len(allnodes))
print("Start querying.")

query = ""

while True:
	query = input("Query (STOP to exit) : ")
	if query.lower() == "stop":
		break
	nodes = retriever.retrieve(query)
	print("Nodes : ")
	for node in nodes:
		print("------------------------------------")
		print(node.node_id)
		print(node.metadata)
		print("....................................")
		print(node.text)
		print("------------------------------------")
	response = query_engine.query(query)
	print(response)
	print("=========================================")


