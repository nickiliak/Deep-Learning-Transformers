from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv()

client = Groq()
completion = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[
      {
        "role": "system",
        "content": "system prompt is here"
      },
      {
        "role": "user",
        "content": "where is the capital of france?"
      }
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    reasoning_effort="default",
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
