
from groq import Groq


def chatbot(prompt, groq_api_key):

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing")

    client = Groq(api_key=groq_api_key)

    try:

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_tokens=2000
        )

        # Debug information
        print("GROQ RESPONSE:", completion)

        if not completion.choices:
            raise ValueError("Groq returned no choices")

        choice = completion.choices[0]

        print("FINISH REASON:", choice.finish_reason)
        print("MESSAGE:", choice.message)

        answer = choice.message.content

        if answer is None:
            raise ValueError(
                f"Groq returned None. Finish reason: {choice.finish_reason}"
            )

        answer = answer.strip()

        if not answer:
            raise ValueError(
                "Groq returned empty text. Check the response and finish reason."
            )

        return answer

    except Exception as e:
        print("GROQ ERROR:", repr(e))
        raise
