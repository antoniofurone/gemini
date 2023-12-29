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

# Chat
chat = ChatVertexAI()

# print(chat([HumanMessage(content="Hello")]).content)

# res = chat(
#     [
#         SystemMessage(
#             content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"
#         ),
#         HumanMessage(content="I like tomatoes, what should I eat?"),
#     ]
# )
# print(res.content)

# res = chat(
#     [
#         HumanMessage(
#             content="What are the ingredients required for making a tomato sandwich?"
#         )
#     ]
# )
# print(res.content)

# res = chat([HumanMessage(content="How many slices of bread you said?")])
# print(res.content)

res = chat(
    [
        SystemMessage(content="You are a helpful AI bot to figure out travel plans."),
        HumanMessage(content="I would like to go to New York, how should I do this?"),
    ]
)
print(res.content)