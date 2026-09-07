import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

client = genai.Client()

def get_w(loc: str) -> str:
    """Gets the weather for a location"""
    return f'Snowing heavily in {loc}'

chat = client.chats.create(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        tools=[get_w],
        temperature=0.3
    )
)
r = chat.send_message("What is the weather in Delhi?")
print("Response text:", r.text)
