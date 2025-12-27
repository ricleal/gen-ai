import psutil
import uvicorn
from dbos import DBOS, DBOSConfig, Queue, WorkflowHandleAsync
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

queue = Queue("example-queue")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
