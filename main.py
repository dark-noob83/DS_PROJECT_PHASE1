import array


def tokenizer(text):
    text = text.lower()
    return text


def word_map(file_name):
    with open(f"./data/document_{file_name}.txt", encoding="utf-8") as f:
        lines = f.readlines()
        i = 0
        word_index = dict()
        for line in lines:
            for word in line.split():
                if tokenizer(word) not in word_index.keys():
                    word_index[tokenizer(word)] = i
                    i += 1
        return word_index


# first piece
def tf_idf_sentence(file_name):
    with open(f"./data/document_{file_name}.txt", encoding="utf-8") as f:
        lines = f.readlines()
        word_index = word_map(file_name=file_name)
        i = 0
        sentence_vector = [None] * (len(lines))

        for line in lines:
            vector = [0] * (list(word_index.values())[-1] + 1)
            for word in line.split():
                index = word_index[tokenizer(word)]
                vector[index] += 1
            sentence_vector[i] = vector
            i += 1
        return sentence_vector


# second piece
def tf_idf(sentence_vector):
    vector = [0] * len(sentence_vector[0])
    for paragraph in sentence_vector:
        for i in range(len(paragraph)):
            vector[i] += paragraph[i]
    return vector


vectors = [0] * 50001
for i in range(0, 10):
    s_v = tf_idf_sentence(i)
    vectors[i] = tf_idf(s_v)
print(vectors)
