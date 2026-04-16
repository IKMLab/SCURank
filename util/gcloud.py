from google import genai
from google.genai import types
from google.auth import default
from google.auth.transport.requests import Request
from mistralai_gcp import MistralGoogleCloud
from openai import OpenAI
import subprocess, time
from secret import PROJ_ID


################# LLAMA API ###################
def get_credentials():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token

def build_llama_api():
    return OpenAI(
        base_url= f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJ_ID}/locations/us-central1/endpoints/openapi/chat/completions?",
        api_key=get_credentials(),
    )

def call_llama(ins:str, content:str, model:str, times:int=0):
    client = build_llama_api()
    if times > 6: raise ValueError("Exceeded the maximum number of retries")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ins},
                {"role": "user", "content": content},
            ],
            max_tokens=512,
            temperature=0.0,
        )
    except Exception as e:
        print(e)
        print("Got Error, retrying...")
        time.sleep(2**(times+1))
        return call_llama(ins, content, model, times+1)
    
    return response.choices[0].message.content

###############################################

################# MISTRAL API #################
def build_mistral_api(loc:str, project:str):
    process = subprocess.Popen(
    "gcloud auth print-access-token", stdout=subprocess.PIPE, shell=True
    )
    (access_token_bytes, err) = process.communicate()
    access_token = access_token_bytes.decode("utf-8").strip()  # Strip newline
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    return MistralGoogleCloud(
        access_token=access_token,
        project_id=project,
        region=loc,
    )

def call_mistral(ins:str, content:str, model:str, times:int=0):
    if times > 6: raise ValueError("Exceeded the maximum number of retries")
    client = build_mistral_api("us-central1", PROJ_ID)
    try:
        resp = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": ins},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        if resp is None: raise ValueError("No response from Mistral")
        if resp.choices[0].message.content is None:
            raise ValueError("No content returned")
        return resp.choices[0].message.content
    except Exception as e:
        print(e)
        print("Got Error, retrying...")
        time.sleep(2**(times+1))
        return call_mistral(ins, content, model, times+1)

###############################################

################# GEMINI API ##################
def build_gemini_api(loc:str, project:str=PROJ_ID):
    return genai.Client(
        vertexai=True,
        location=loc,
        project=project,
    )

def get_content_config(instruction:str):
    return types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        system_instruction=[types.Part.from_text(text=instruction)]
    )

def call_gemini(ins:str, content:str, model:str, times:int=0):
    if times > 2: raise ValueError("Exceeded the maximum number of retries")

    client = build_gemini_api("us-central1")
    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text=content)])
    ]

    try:
        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=get_content_config(ins)
        )
        if response.candidates[0].content is None:
            print("No content returned")
            return ""
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        time.sleep(2**(times+1))
        print(f"Error: {e}")
        return call_gemini(ins, content, model, times+1)
    
###############################################


