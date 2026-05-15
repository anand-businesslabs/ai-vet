import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# Allow your Brilliant Directories site to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VetRequest(BaseModel):
    pet_type: str
    breed: str
    age: str
    weight: str
    city: str
    symptoms: str


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/ai-vet")
def ai_vet(payload: VetRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    prompt = f"""
You are an AI veterinary assistant for Pettinut.

Review the following pet information and return a JSON response only.

Pet Type: {payload.pet_type}
Breed: {payload.breed}
Age: {payload.age}
Weight: {payload.weight}
City: {payload.city}
Symptoms: {payload.symptoms}

Return clear, safe guidance in JSON with fields such as:
- summary
- possible_causes
- home_care
- urgency
- when_to_see_a_vet
- disclaimer

Do not return markdown. Return valid JSON only.
""".strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={"response_mime_type": "application/json"},
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Gemini did not return valid JSON",
                "raw_response": response.text,
            },
        ) from exc
