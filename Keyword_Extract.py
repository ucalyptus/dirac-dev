# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:25:43 2019

@author: Shatadru Majumdar
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def keyword_extraction(tfidf_vector, document_list):
    keywords = []
    for i in range(0, len(document_list)):
        response2 = tfidf_vector.transform(
            [document_list[i]]
        )
        feature_array = np.array(
            tfidf_vector.get_feature_names()
        )
        tfidf_sorting = np.argsort(
            response2.toarray()
        ).flatten()[::-1]
        n = 12
        top_n = feature_array[tfidf_sorting][:n]
        keywords.append(top_n)
    return keywords


fileName = "report4.txt"
d = open(fileName, "r", encoding="utf8")
document = d.read()
document_list = document.split("\n\n")
tfidf_vector = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 3)
)
response = tfidf_vector.fit_transform(document_list)
keywords = keyword_extraction(tfidf_vector, document_list)
keywords = np.asarray(keywords)
print(keywords)
