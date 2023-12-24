import vertexai
from vertexai.language_models import TextGenerationModel

vertexai.init(project="cyorg-genai-test", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison@001")
response = model.predict(
    """Extract the technical specifications from the text below in a JSON format.

Text: Google Nest Wifi, network speed up to 1200Mpbs, 2.4GHz and 5GHz frequencies, WP3 protocol
JSON: {
  \"product\":\"Google Nest Wifi\",
  \"speed\":\"1200Mpbs\",
  \"frequencies\": [\"2.4GHz\", \"5GHz\"],
  \"protocol\":\"WP3\"
}

Text: Google Pixel 7, 5G network, 8GB RAM, Tensor G2 processor, 128GB of storage, Lemongrass
JSON:
""",
    **parameters
)
print(f"Response from Model: {response.text}")
