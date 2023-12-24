from typing import Optional
#from google.oauth2.credentials import Credentials
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

initSdk(project="cyorg-genai-test", location="us-central1")