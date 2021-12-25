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



        # try:
        #     for line in infile:
        #         #import ipdb; ipdb.set_trace();
        #         if len(line.split()) > 10: 
        #             sentences = sent_tokenize(line)
        #             for sent in sentences:
        #                 if len(sent) > 5:
        #                     outfile.write(sent + '\n')
        # except IndexError as e:
        #     pass
        #     #print(e)
    


