import json
from pathlib import Path

history = Path("memory/conversation.json")
#load to llm
def load_history():
    if not history.exists():
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text("[]", encoding="utf-8")

    with open(history, "r", encoding="utf-8") as file:
        return json.load(file)

#save in the sense erase

def save_history(history_data):
    with open(history, "w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=4)

#add new msg to the conversatn
def add_msg(role, content):
    history_data = load_history()

    history_data.append({
        "role": role,
        "content": content
    })

    save_history(history_data)

#return alll msg
def get_hstry():
    return load_history()

#clears history
def clear_hstry():
    save_history([])




