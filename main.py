import difflib
import json
import math
import string

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


# tokenizer
def tokenizer(text, tokenizer=TweetTokenizer()):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    punct = list(string.punctuation)
    stopword_list = stopwords.words('english') + punct + ['rt', 'via', '...', '“', '”', '’']
    return [tok.strip("#") for tok in tokens if tok not in stopword_list and not tok.isdigit()]


# find and save unique word into json
# df of unique word
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


# tf
def Tf(selected_word, splited_paragraph, ):
    tf = 0
    for word in splited_paragraph:
        if word == selected_word:
            tf += 1
    return tf


# paragraph number
def p_number():
    paragraph_numbers = 0

    # paragraph number
    for document_number in range(0, 50001):
        with open(f"./data/document_{document_number}.txt", encoding="utf-8") as f:
            lines = f.readlines()
            paragraph_numbers += len(lines)

    return paragraph_numbers


# save paragraph tokenizer in json
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


# tf idf
def Tf_Idf(splited_paragraph, last_index, unique_words, paragraph_number):
    vector = [0] * (last_index + 1)
    for word in splited_paragraph:
        word = correct_spelling(word, unique_words)
        tf = Tf(word, splited_paragraph)
        index = unique_words[word]['index']
        df = unique_words[word]['paragraph']
        idf = paragraph_number / df
        tf_idf = tf * math.log(idf, 10) / len(splited_paragraph)
        vector[index] = tf_idf
    return np.array(vector)


# save tf idf of documents
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


# cosine similarity
def cosine_similarity(query_vector, document_vector):
    numerator = np.dot(query_vector, document_vector)
    query_norm = np.linalg.norm(query_vector)
    document_norm = np.linalg.norm(document_vector)
    denominator = query_norm * document_norm
    cosine = numerator / denominator
    return cosine


def get_document_tf_idf(document_name):
    with open(f"./Tf_idf's/{document_name}.txt", encoding='utf-8') as f:
        line = f.readline()
        line = line[1:-1]
        spilited_line = line.split(',')
    return spilited_line


# return best document for query
def select_document(query, candidate_document):
    paragraph_number = p_number()
    with open("./unique_words.json") as fp:
        unique_words = json.load(fp)
    last_index = list(unique_words.values())[-1]['index']
    cosines = []
    query_tokenized = tokenizer(query)

    query_tf_idf = Tf_Idf(query_tokenized, last_index, unique_words, paragraph_number)

    for document_name in candidate_document:
        spilited_line = get_document_tf_idf(document_name)
        document_tf_idf = np.array(spilited_line, dtype='float64')
        cosine = cosine_similarity(query_tf_idf, document_tf_idf)
        cosines.append({'cosine': cosine, 'document': document_name})
    print(cosines)
    return max(cosines, key=lambda x: x['cosine'])


# print best document for query of data.json
def print_document(index):
    with open("./data.json") as fp:
        data_form = json.load(fp)

    query = data_form[index]['query']
    candidate_document = data_form[index]['candidate_documents_id']
    c = None
    if '.' in query or ',' in query:
        c = query
        j = 0
        for (index, i) in enumerate(query):
            if i == '.' or i == ',':
                if j < len(query) and not query[j - 1].isdigit():
                    a = c[:j + 1]
                    b = c[j + 1:]
                    c = a + ' ' + b
                    j += 1
            j += 1
    if c:
        query = c
    print(select_document(query, candidate_document))


# for mistake spelling
def correct_spelling(word, word_list):
    closest_match = difflib.get_close_matches(word, word_list, n=1, cutoff=0.8)
    if closest_match:
        return closest_match[0]
    return word


# return top 5 most repeated
def get_most_repeated(document_name, unique_words, paragraph_number):
    document_tf_idf = get_document_tf_idf(document_name)
    tfs = []
    keys = list(unique_words.keys())
    for (index, word_tf_idf) in enumerate(document_tf_idf):
        word_tf_idf = float(word_tf_idf)
        word = keys[index]
        df = unique_words[word]['paragraph']
        tf = word_tf_idf / math.log(paragraph_number / df, 10)
        tfs.append({'word': word, 'tf': tf})
    sorted_list = sorted(tfs, key=lambda x: x['tf'], reverse=True)
    return sorted_list[:5]


# print top five most repeated
def print_most_repeated(document_name):
    paragraph_number = p_number()
    with open("./unique_words.json") as fp:
        unique_words = json.load(fp)
    for most_repeated in get_most_repeated(document_name, unique_words, paragraph_number):
        print(most_repeated['word'])


# get most important

def get_most_important(document_name, unique_words):
    document_tf_idf = get_document_tf_idf(document_name)
    tf_idfs = []
    keys = list(unique_words.keys())
    for (index, word_tf_idf) in enumerate(document_tf_idf):
        word_tf_idf = float(word_tf_idf)
        word = keys[index]
        tf_idfs.append({'word': word, 'tf_idf': word_tf_idf})
    sorted_list = sorted(tf_idfs, key=lambda x: x['tf_idf'], reverse=True)
    return sorted_list[0]


# return most important
def most_important(document_name):
    with open("./unique_words.json") as fp:
        unique_words = json.load(fp)
    important = get_most_important(document_name, unique_words)
    return (important['word'])


#######################################################################################################
##################  part 2   ##################

# reduce dimension
def dimension_reduction(documents_tf_idf):
    pca = PCA(n_components=2)
    transform = pca.fit_transform(documents_tf_idf)
    return transform


# save_2d_array in txt file of 1000 doc
def save_2d_tf_idf():
    documents_tf_idf = []
    for document_name in range(0, 1000):
        print(document_name)
        documents_tf_idf += [get_document_tf_idf(document_name)]
    documents_tf_idf = np.array(documents_tf_idf, dtype='float64')
    transform_data = list(dimension_reduction(documents_tf_idf))
    transform_dict = {}
    for (index, value) in enumerate(transform_data):
        most_imp = most_important(index)
        transform_dict[most_imp] = list(value)
    with open(f"./2d_tf_idf.json", mode='w', encoding="utf-8") as f1:
        json.dump(transform_dict, f1)


# clustering 1000 doc
def clustering_tf_idf():
    words = []
    for document_name in range(0, 1000):
        most = most_important(document_name)
        words += [most]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(words)
    most_important_cluster = {}
    num_clusters = 7
    model = KMeans(n_clusters=num_clusters)
    model.fit(X)
    clusters = model.labels_
    for i, word in enumerate(words):
        if word not in most_important_cluster:
            most_important_cluster[word] = int(clusters[i])

    return most_important_cluster


# save clustering of 1000 doc in json
def save_most_important_cluster():
    most_important_cluster = clustering_tf_idf()
    print(most_important_cluster)
    with open(f"./most_important_clustering.json", "w") as myfile:
        json.dump(most_important_cluster, myfile)


# read x and y of tf idf
def read_2d_tf_idf():
    with open("./2d_tf_idf.json") as fp:
        all_2d = json.load(fp)
        return all_2d


# read clustering
def read_clustering():
    with open("./most_important_clustering.json") as fp:
        all_clustering = json.load(fp)
        return all_clustering


# make x and y  for diagrom based on cluster
def make_coordinate():
    all_2d = read_2d_tf_idf()
    all_clustering = read_clustering()
    x = [None] * 7
    y = [None] * 7
    for key, value in list(all_2d.items()):
        index = all_clustering[key]
        if not x[index]:
            x[index] = []
        x[index].append(value[0])
        if not y[index]:
            y[index] = []
        y[index].append(value[1])
    return {'x': x, 'y': y}


# draw diagram
def draw_diagram():
    coordinate = make_coordinate()
    colors = ['y', 'r', 'g', 'b', 'purple', 'gray', 'black']

    for i in range(7):
        x = np.array(coordinate['x'][i])
        y = np.array(coordinate['y'][i])
        plt.scatter(x, y, s=120, facecolors='none', edgecolors=colors[i])

    plt.show()

draw_diagram()