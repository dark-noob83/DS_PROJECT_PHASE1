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


def tf_idf(file_name):
    with open(f"./data/document_{file_name}.txt", encoding="utf-8") as f:
        lines = f.readlines()
        word_index = word_map(file_name=file_name)
        vector = [0] * (list(word_index.values())[-1] + 1)

        for line in lines:
            for word in line.split():
                index = word_index[tokenizer(word)]
                vector[index] += 1
        return vector


vectors = [0] * 50001
for i in range(0, 10):
    vectors[i] = tf_idf(i)

print(vectors)
# for i in range(0, 10):
#     tf_idf(i)
