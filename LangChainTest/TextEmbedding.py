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

from langChainUtils import CustomVertexAIEmbeddings

# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

text = "Hi! It's time for the beach"

text_embedding = embeddings.embed_query(text)
print(f"Your embedding is length {len(text_embedding)}")
print(f"Here's a sample: {text_embedding[:5]}...")