from groq import Groq


def chatbot(prompt, groq_api_key):

    client = Groq(api_key=groq_api_key)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, knowledgeable and intelligent assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.2,
        max_completion_tokens=200
    )

    return response.choices[0].message.content