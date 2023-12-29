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


# LLM model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

# prompt = """
# Today is Monday, tomorrow is Wednesday.

# What is wrong with that statement?
# """

# print(llm(prompt))

from langchain import PromptTemplate

# Notice "location" below, that is a placeholder for another value later
template = """
I really want to travel to {location}. What should I do there?

Respond in one short sentence
"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location="Rome")

print(f"Final Prompt: {final_prompt}")
print("-----------")
print(f"LLM Output: {llm(final_prompt)}")