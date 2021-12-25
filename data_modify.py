class RandomNoiser():

    def __init__(self, n_words):
        self.n_words = n_words

    def noiser(self, text):

        self.text = text

        json_dict = {}
        for i in range(0, self.n_words):

            lower_text = string.ascii_lowercase
            sl = list(lower_text)
            random_character = random.choice(sl)
            text = re.sub('\W+',' ', self.text.lower())
            list_ = text.split(" ")
            if "" in list_:
                list_.remove("")
            ch = random.choice(list_)
            if ch not in ["", " "]:
                k = ch.replace(random.choice(ch), random.choice(sl))
            else:
                k = ch.replace(ch, random.choice(sl))
            if ch not in json_dict.keys():   
                json_dict[ch] = k

        nl = []
        for x in list_:
            if x not in json_dict.keys():
                nl.append(x)
            else:
                nl.append(json_dict[x])

        return (" ").join(nl)