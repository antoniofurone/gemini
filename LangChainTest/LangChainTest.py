from google.cloud import aiplatform

PROJECT_ID = "cyorg-genai-test"  # @param {type:"string"}
aiplatform.init(project=PROJECT_ID, location="us-central1")

from langchain.schema import HumanMessage, SystemMessage
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.chat_models import ChatVertexAI
from google.cloud import aiplatform
import time
from typing import List

# LangChain
import langchain
from pydantic import BaseModel
print(f"LangChain version: {langchain.__version__}")

# Vertex AI
print(f"Vertex AI SDK version: {aiplatform.__version__}")



# LLM model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)


# Chat
chat = ChatVertexAI()

from langChainUtils import CustomVertexAIEmbeddings

# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

# You'll be working with simple strings (that'll soon grow in complexity!)
my_text = "What day comes after Friday?"

print(llm(my_text))

# Documents
from langchain.schema import Document
d=Document(
    page_content="This is my document. It is full of text that I've gathered from other places",
    metadata={
        "my_document_id": 234234,
        "my_document_source": "The LangChain Papers",
        "my_document_create_time": 1680013019,
    },
)
print(d)