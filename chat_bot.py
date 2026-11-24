
def bot(message: str) -> str:
    anwser = 'MY name is Benjamin!!'
    return anwser


def main():
    input_text = input("Ask something! ")
    print(bot(input_text))

if __name__ == "__main__":
    main()