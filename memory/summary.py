from pathlib import Path
import json
from memory.conversation import get_hstry
from langchain_core.messages import SystemMessage, HumanMessage
from model.config import llm

summary = Path("memory/summary.json")

def load_summary():
    if not summary.exists():
        summary.parent.mkdir(parents=True, exist_ok=True)
        with open(summary, "w", encoding="utf-8") as file:
            json.dump({"summary": ""}, file, indent=4)

    with open(summary, "r", encoding="utf-8") as file:
        return json.load(file)


def save_summary(summary_txt):
    with open(summary, "w", encoding="utf-8") as file:
        json.dump({"summary": summary_txt}, file, indent=4)


def format_convo(history):
    format = []
    for meassages in history:
        role = meassages["role"]
        content = meassages["content"]
        format.append(f"{role}:{content}")
    return "\n".join(format)


def creat_summary(llm, history):
    convo = format_convo(history)

    messages = [
        SystemMessage(
            content="""
you are a very briilliant and excellent conversation summariser
store all important facts and keywords as name,place and ignore the unessacery talks. Make
the summary as short and simple asusal.your job is just to summarizeand kep the summary in the summary.json
dont use ur brain and answer just do summary thats it.I dont want any suggestion ur task is to just make summary
I wat summary in a plain english"""
        ),
        HumanMessage(content=convo)
    ]

    response = llm.invoke(messages)
    save_summary(response.content)
    return response.content