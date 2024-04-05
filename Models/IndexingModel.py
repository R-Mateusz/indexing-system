from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path
import numpy as np


class ArticleIndex:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.csv_path = self.get_path_of_dataset()
        self.dataset_articles = self.load_dataset(self.csv_path)
        self.trained_model = self.fit_vector(self.dataset_articles)
    def get_path_of_dataset(self):
        current_dir = Path(__file__).resolve().parent
        csv_path = current_dir / 'datasets/medium.csv'
        return csv_path
    def load_dataset(self, dataset):
        dataset_data = pd.read_csv(dataset)
        articles_data = dataset_data['Title'] +"\n\n"+ dataset_data['Text']
        return articles_data.tolist()

    def fit_vector(self, dataset_articles):
        trained_vector = self.vectorizer.fit_transform(dataset_articles)
        return trained_vector

    def search_articles(self,document_query, result=5):
        query_vector = self.vectorizer.transform([document_query])
        cos_similarity = cosine_similarity(query_vector, self.trained_model)[0]
        results_sort = cos_similarity.argsort()[-result:]
        return [self.dataset_articles[i] for i in results_sort]

indexer = ArticleIndex()
query_input = input("Write query prompt: ")
articles_found = indexer.search_articles(query_input)

for x in articles_found:
    print(x)






