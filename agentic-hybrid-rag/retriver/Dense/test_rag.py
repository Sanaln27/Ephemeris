from .generator import generate_answer

while True:
    question=input("ASK:")
    if question.lower=="exit":
        break
    answer=generate_answer(question)
    print("\n+++output++++")
    print(answer)

