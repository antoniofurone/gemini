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


# LLM model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)



from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# How you would like your reponse structured. This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(
        name="bad_string", description="This a poorly formatted user input string"
    ),
    ResponseSchema(
        name="good_string", description="This is your response, a reformatted response"
    ),
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
print(format_instructions)

template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly including country, city and state names

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""


from langchain import PromptTemplate
prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template,
)

promptValue = prompt.format(user_input="welcom to dbln!")

print(promptValue)

llm_output = llm(promptValue)
print(llm_output)

print(output_parser.parse(llm_output))
