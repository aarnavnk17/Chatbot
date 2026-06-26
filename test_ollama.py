from ollama import chat

response = chat(
    model="qwen2.5-coder:7b",
    messages=[
        {
            "role": "user",
            "content": "Explain vector databases in 3 sentences."
        }
    ]
)

print(
    response["message"]["content"]
)