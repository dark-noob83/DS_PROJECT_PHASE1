import json
import math


def tokenizer(text):
    text = text.lower()
    return text


# for unique word and df of unique words
# unique_word = {}
# i = 0
# for document_number in range(0, 50001):
#     with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in lines:
#             word_counted = set()
#             splited_line = line.split()
#             for word in splited_line:
#                 if tokenizer(word) not in unique_word.keys():
#                     unique_word[tokenizer(word)] = {}
#                     unique_word[tokenizer(word)]["index"] = i
#                     unique_word[tokenizer(word)]["paragraph"] = 1
#                     i += 1
#                     word_counted.add(tokenizer(word))
#                 elif tokenizer(word) not in word_counted:
#                     unique_word[tokenizer(word)]["paragraph"] += 1
#                     word_counted.add(tokenizer(word))

# write for unique_words.json
# with open("./unique_words.json", "w") as myfile:
#     json.dump(unique_word, myfile)

with open("./unique_words.json") as fp:
    unique_words = json.load(fp)


### calculate tf
def Tf(selected_word, paragraph):
    tf = 0
    splited_paragraph = paragraph.split()
    for word in splited_paragraph:
        if tokenizer(word) == tokenizer(selected_word):
            tf += 1
    return tf


last_index = list(unique_words.values())[-1]['index']
paragraph_numbers = 0

# paragraph number
for document_number in range(0, 50001):
    with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
        lines = f.readlines()
        paragraph_numbers += len(lines)




def Tf_Idf(paragraph):
    vector = [0] * (last_index + 1)
    splited_paragraph = paragraph.split()
    for word in splited_paragraph:
        tf = Tf(tokenizer(word), paragraph)
        index = unique_words[tokenizer(word)]['index']
        df = unique_words[tokenizer(word)]['paragraph']
        idf = paragraph_numbers / df
        tf_idf = tf * math.log(idf, 10) / len(splited_paragraph)
        vector[index] = tf_idf
    return vector

for document_number in range(0,50001):
    with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
        lines = f.readlines()
        with open(f"./Tf_idf's/document_{document_number}.txt",mode='w',  encoding="utf-8") as f1:
            for paragraph in lines:
                f1.write(str(Tf_Idf(paragraph))+'\n')
