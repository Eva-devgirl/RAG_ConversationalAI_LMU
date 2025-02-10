import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-proj-DwikG6Qsk0rlnchK8czNT3BlbkFJoAgxLhHtFmyLAvg8j2nJ" #dafür ist es bezahlt worden
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
print("Define settings.")
Settings.llm = OpenAI()
print("Reading documents.")
filename_fn = lambda filename: {"file_name": filename}
documents = SimpleDirectoryReader("./datei.html", filename_as_id=True).load_data()
print([x.doc_id for x in documents])
print("Indexing.")
index = VectorStoreIndex.from_documents(
    documents,
)
print("Saving")
index.storage_context.persist(persist_dir="muenchenmenuindex_simple_filenames")
