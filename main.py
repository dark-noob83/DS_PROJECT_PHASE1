import json


def tokenizer(text):
    text = text.lower()
    return text


unique_word = {}
i = 0
for document_number in range(0, 50001):
    with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            word_counted = set()
            splited_line = line.split()
            for word in splited_line:
                if tokenizer(word) not in unique_word.keys():
                    unique_word[tokenizer(word)] = {}
                    unique_word[tokenizer(word)]["index"] = i
                    unique_word[tokenizer(word)]["paragraph"] = 1
                    i += 1
                    word_counted.add(tokenizer(word))
                elif tokenizer(word) not in word_counted:
                    unique_word[tokenizer(word)]["paragraph"] += 1
                    word_counted.add(tokenizer(word))

# write for unique_words.json
with open("./unique_words.json", "w") as myfile:
    json.dump(unique_word, myfile)

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


def Df(selected_word):
    df = 0
    for document_number in range(0, 50001):
        with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if selected_word in line.lower():
                    df += 1
                    break
    return df


m = {1, 2, 3}
print(1 in m)  # unique_words_df = {}
# for key in tqdm(list(unique_words.keys())):
#     unique_words_df[key] = Df(key)

# with open("./unique_words_df.json", "w") as myfile:
#     json.dump(unique_words_df, myfile)
