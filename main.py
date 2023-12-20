import json
import math
import string
import difflib
import numpy as np
# nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


def tokenizer(text, tokenizer=TweetTokenizer()):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    punct = list(string.punctuation)
    stopword_list = stopwords.words('english') + punct + ['rt', 'via', '...', '“', '”', '’']
    return [tok.strip("#") for tok in tokens if tok not in stopword_list and not tok.isdigit()]


def save_unique_words():
    unique_word = {}
    i = 0
    for document_number in range(0, 50001):
        with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                word_counted = set()
                splited_line = tokenizer(line)
                for word in splited_line:
                    if word not in unique_word.keys():
                        unique_word[word] = {}
                        unique_word[word]["index"] = i
                        unique_word[word]["paragraph"] = 1
                        i += 1
                        word_counted.add(word)
                    elif word not in word_counted:
                        unique_word[word]["paragraph"] += 1
                        word_counted.add(word)

    # write for unique_words.json
    with open("./unique_words.json", "w") as myfile:
        json.dump(unique_word, myfile)


def Tf(selected_word, splited_paragraph, ):
    tf = 0
    for word in splited_paragraph:
        if word == selected_word:
            tf += 1
    return tf


def p_number():
    paragraph_numbers = 0

    # paragraph number
    for document_number in range(0, 50001):
        with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
            lines = f.readlines()
            paragraph_numbers += len(lines)

    return paragraph_numbers


def save_tokenizer():
    for document_number in range(0, 50001):
        d_paragraph_token = {}
        i = 0
        with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
            lines = f.readlines()
            for p in lines:
                d_paragraph_token[i] = tokenizer(p)
                i += 1
            with open(f"./tokenizer/{document_number}.json", "w") as myfile:
                json.dump(d_paragraph_token, myfile)


def Tf_Idf(splited_paragraph, last_index, unique_words, paragraph_number):
    vector = [0] * (last_index + 1)
    for word in splited_paragraph:
        tf = Tf(word, splited_paragraph)
        index = unique_words[word]['index']
        df = unique_words[word]['paragraph']
        idf = paragraph_number / df
        tf_idf = tf * math.log(idf, 10) / len(splited_paragraph)
        vector[index] = tf_idf
    return np.array(vector)


def save_tf_idf():
    paragraph_number = p_number()
    with open("./unique_words.json") as fp:
        unique_words = json.load(fp)
    last_index = list(unique_words.values())[-1]['index']
    for document_number in range(0, 50001):
        with open(f"./tokenizer/{document_number}.json") as f:
            d_paragraph_tokenizer = json.load(f)
            document_vector = np.array([0] * (last_index + 1), dtype='float64')

            for paragraph_index in d_paragraph_tokenizer.keys():
                paragraph_vector = Tf_Idf(d_paragraph_tokenizer[paragraph_index], last_index, unique_words,
                                          paragraph_number)
                document_vector += paragraph_vector
            document_vector = list(document_vector)

            with open(f"./Tf_idf's/{document_number}.txt", mode='w', encoding="utf-8") as f1:
                f1.write(str(document_vector))


def cosine_similarity(query_vector, document_vector):
    numerator = np.dot(query_vector, document_vector)
    query_norm = np.linalg.norm(query_vector)
    document_norm = np.linalg.norm(document_vector)
    denominator = query_norm * document_norm
    cosine = numerator / denominator
    return cosine


def select_document(query, candidate_document):
    paragraph_number = p_number()
    with open("./unique_words.json") as fp:
        unique_words = json.load(fp)
    last_index = list(unique_words.values())[-1]['index']
    cosines = []
    query_tokenized = tokenizer(query)
    # print(query_tokenized)
    try:
        query_tf_idf = Tf_Idf(query_tokenized, last_index, unique_words, paragraph_number)

    for document_name in candidate_document:
        with open(f"./Tf_idf's/{document_name}.txt", encoding='utf-8') as f:
            line = f.readline()
            line = line[1:-1]
            splited_line = line.split(',')
            document_tf_idf = np.array(splited_line, dtype='float64')
            cosine = cosine_similarity(query_tf_idf, document_tf_idf)
            cosines.append({'cosine': cosine, 'document': document_name})
    # print(cosines)
    return max(cosines, key=lambda x: x['cosine'])


with open("./data.json") as fp:
    data_form = json.load(fp)



query = data_form[6]['query']
a = "higher.domestically"
b = "higher"
s = difflib.SequenceMatcher(None, a,b)
for tag, i1, i2, j1, j2 in s.get_opcodes():
    print(a[i1:i2]==b)
#
# for (i , index) in enumerate(query):
#     if(i == '.'):

# candidate_document = data_form[6]['candidate_documents_id']
# print(select_document(query, candidate_document))
