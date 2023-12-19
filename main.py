import json


def tokenizer(text):
    text = text.lower()
    return text


## for unique words
# unique_word = {}
# i = 0
# for document_number in range(0, 50001):
#     with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in lines:
#             splited_line = line.split()
#             for word in splited_line:
#                 if tokenizer(word) not in unique_word.keys():
#                     unique_word[tokenizer(word)] = i
#                     i += 1
# write for unique_words.json
# with open("./unique_words.json", "w") as myfile:
#     myfile.write(json.dumps(unique_word))


##print(unique_word)
