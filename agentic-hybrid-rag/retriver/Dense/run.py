
from .generator import generate_answer


while True:
    question=input("puchoooo,puchoooonahh:")

    if question.lower=="exit":
        print("abeyy jhaaa bhe")
        break
    answer=generate_answer(question)
    print("THIS IS THE ANSWER U BITCH:")
    print(answer)