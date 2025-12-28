import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich.console import Console

console = Console()


# Configure Ollama as the LLM and local embeddings
Settings.llm = Ollama(model="deepseek-r1:1.5b", request_timeout=120.0)
# Uses HuggingFace embeddings (BAAI/bge-small-en-v1.5) to convert text into vectors
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# the data was downloaded before with `llamaindex-cli download-llamadataset PaulGrahamEssayDataset --download-dir ./data`
rag_dataset = LabelledRagDataset.from_json("./data/rag_dataset.json")
documents = SimpleDirectoryReader(input_dir="./data/source_files").load_data()


# Create ChromaDB client (persists to ./chroma_db)
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("my_collection")

# Create vector store and index
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

query_engine = index.as_query_engine()

question = "What did the author do growing up?"
console.print(f"[bold blue]Question:[/bold blue] {question}")
response = query_engine.query(question)
console.print(f"[bold green]Response:[/bold green] {response}")
