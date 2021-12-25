import random
import string
import re

class Randnoise():

    def __init__(self, n_words):
        self.n_words = n_words
        self.json_dict = {}

    def replace_char(self, text):
        self.text = text
        text = re.sub('\W+',' ', self.text.lower())
        text_list = text.split(" ")
        if "" in text_list:
            text_list.remove("")
        # json_dict = {}
        for i in range(0, self.n_words):

            lower_text = string.ascii_lowercase
            sl = list(lower_text)
            # removes special characters 
            # re.sub('[^A-Za-z0-9]+', ' ', mystring) # this removes special characters, but the following line of code is much faster.     
            rand_word = random.choice(text_list)
            if rand_word not in ["", " "]:
                k = rand_word.replace(random.choice(rand_word), random.choice(sl))
            else:
                k = rand_word.replace(rand_word, random.choice(sl))
            if rand_word not in self.json_dict.keys():   
                self.json_dict[rand_word] = k

        nl = []
        for x in text_list:
            if x not in self.json_dict.keys():
                nl.append(x)
            else:
                nl.append(self.json_dict[x])

        return (" ").join(nl)

    def del_char(self, text):
        self.text = text
        text = re.sub('\W+',' ', self.text.lower())
        text_list = text.split(" ")
        if "" in text_list:
            text_list.remove("")
        # json_dict = {}
        for i in range(0, self.n_words):
            rand_word = random.choice(text_list)
            if rand_word not in ["", " "]:
                #len_word = len(rand_word)
                k = rand_word.replace(random.choice(rand_word),'',1)
            if rand_word not in self.json_dict.keys():   
                self.json_dict[rand_word] = k

        nl = []
        for x in text_list:
            if x not in self.json_dict.keys():
                nl.append(x)
            else:
                nl.append(self.json_dict[x])

        return (" ").join(nl)
        

rand_noise = Randnoise(n_words=random.randint(1, 4))
text = "  clean chit modi mission shakti speech & *narrow view complaint says yechury "
cor_text = rand_noise.replace_char(text)
corrr = rand_noise.del_char(text)
print(cor_text)
print(corrr)
