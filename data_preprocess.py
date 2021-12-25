from nltk.tokenize import sent_tokenize

with open('data/processed_text.txt', 'w') as outfile :
    with open("data/simplewiki-latest-pages-articles.txt", "r") as infile:
        try:
            for line in infile:
                if not line.startswith("."):
                     if not line.startswith("!"):
                        sentences = sent_tokenize(line)
                        for sent in sentences:
                            if len(sent) > 75:
                                outfile.write(sent + '\n')
        except IndexError as e:
            print(f"error in {line} ")
            print(f"error is {e}")
            pass
print("Sucessfully preprocessed the files and the files are written in data/processed_text.txt")



