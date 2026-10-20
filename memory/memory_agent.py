from memory.conversation import get_hstry,add_msg
from memory.summary import creat_summary
from model.config import llm

msg_lmt=20

def memory_agent(llm,user_imput,assistant_response):
    history=get_hstry()
    add_msg("user",user_imput)
    add_msg("assistant",assistant_response)

    if len(history)>=msg_lmt:
        return creat_summary(llm,history)
    return None