from flask import Flask, render_template, request, session, redirect, url_for
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
import re

load_dotenv()

# Tell Flask to look for templates in the current directory
app = Flask(__name__, template_folder='.')

app.secret_key = 'your_secret_key_here'

def format_ai_response(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

def is_education_related(text):
    keywords = [
        "education", "learning", "teaching", "classroom", "student",
        "school", "university", "college", "curriculum", "knowledge", "exam"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords) or "?" in text

AZURE_KEY = os.getenv("GITHUB_TOKEN")
AZURE_ENDPOINT = "https://models.inference.ai.azure.com"

client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY)
)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    chat_history = session["chat_history"]

    if request.method == "POST":
        user_input = request.form["user_input"]
        chat_history.append({"sender": "user", "text": user_input})

        if not is_education_related(user_input):
            ai_response = "❌ Sorry, I can only assist with **education**, **knowledge**, or **questions**."
            chat_history.append({"sender": "ai", "text": format_ai_response(ai_response)})
        else:
            try:
                messages = [
                    SystemMessage(content="You are an agentic AI tutor. If the user's question is complex or goal-oriented, break it into smaller steps, answer each, and provide a summary at the end. You only answer questions related to education, learning, or knowledge.")
                ]

                for msg in chat_history:
                    if msg["sender"] == "user":
                        messages.append(UserMessage(content=msg["text"]))
                    elif msg["sender"] == "ai":
                        plain_text = re.sub(r'<.*?>', '', msg["text"])
                        messages.append(AssistantMessage(content=plain_text))

                user_task = chat_history[-1]["text"]
                agentic_prompt = f"""
Analyze the user's goal or question:
\"{user_task}\"

If it’s complex, break it into logical sub-questions. Answer each sub-question clearly, and end with a final summary.

Use this format:
1. Sub-question 1 - Answer
2. Sub-question 2 - Answer
...
✅ Final Summary: ...

Only proceed if this is an educational topic.
"""
                messages.append(UserMessage(content=agentic_prompt))

                response = client.complete(
                    messages=messages,
                    model="gpt-4o",
                    temperature=0.8,
                    max_tokens=1200,
                    top_p=1
                )

                raw_response = response.choices[0].message.content
                formatted = format_ai_response(raw_response)
                chat_history.append({"sender": "ai", "text": formatted})

            except Exception as e:
                chat_history.append({"sender": "ai", "text": f"<strong>Error:</strong> {str(e)}"})

        session["chat_history"] = chat_history
        return redirect(url_for("index"))

    return render_template("index.html", chat_history=chat_history)

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return "", 204

if __name__ == "__main__":
    app.run(debug=True)
