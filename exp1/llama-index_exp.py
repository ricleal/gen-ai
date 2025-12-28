from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Configure Ollama as the LLM and local embeddings
Settings.llm = Ollama(model="deepseek-r1:1.5b", request_timeout=120.0)
# Uses HuggingFace embeddings (BAAI/bge-small-en-v1.5) to convert text into vectors
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# the data was downloaded before with `llamaindex-cli download-llamadataset PaulGrahamEssayDataset --download-dir ./data`
rag_dataset = LabelledRagDataset.from_json("./data/rag_dataset.json")
documents = SimpleDirectoryReader(input_dir="./data/source_files").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
