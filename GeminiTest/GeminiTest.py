from vertexai.preview.generative_models import GenerativeModel, Part
#from vertexai.generative_models import GenerativeModel, Part
from typing import Optional
from google.auth.credentials import Credentials

def initSdk(
    project: Optional[str] = None,
    location: Optional[str] = None,
    experiment: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    credentials: Optional[Credentials] = None,
    encryption_spec_key_name: Optional[str] = None,
    service_account: Optional[str] = None,
):

    from google.cloud import aiplatform

    aiplatform.init(
        project=project,
        location=location,
        experiment=experiment,
        staging_bucket=staging_bucket,
        credentials=credentials,
        encryption_spec_key_name=encryption_spec_key_name,
        service_account=service_account,
    )


def get_chat_response(message):
 initSdk(project="cyorg-genai-test", location="us-central1")
 model = GenerativeModel("gemini-pro")
 chat = model.start_chat()
 response = chat.send_message(message)
 return response.text

print(get_chat_response("Hello"))
print(get_chat_response("What are all the colors in a rainbow?"))
print(get_chat_response("Why does it appear when it rains?"))