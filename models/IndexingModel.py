from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class ArticleIndex:
    """
    ArticleIndex class is a representation of indexing mechanism for datasets
    """

    def __init__(self, results, csv_path):
        """
        Initialization of ArticleIndex
        :param results: int
        :param csv_path: WindowsPath
        """
        self.csv_path = csv_path
        self.results = results
        self.vectorizer = TfidfVectorizer()
        self.dataset_articles = self.load_dataset()
        self.trained_model = self.fit_vector(self.dataset_articles)

    def load_dataset(self):
        """
        Get all articles of dataset

        :return: list
        """
        dataset_data = pd.read_csv(self.csv_path)
        articles_data = dataset_data['Title'] + "\n\n" + dataset_data['Text']
        return articles_data.tolist()

    def fit_vector(self, dataset_articles):
        """
        Fitting and transforming vectorizer into numeric representation for analysis purpose

        :param dataset_articles: list
        :return: csr_matrix
        """
        trained_vector = self.vectorizer.fit_transform(dataset_articles)
        return trained_vector

    def search_articles(self, document_query):
        """
        Search for "n" articles based on query and article text similarity

        :param document_query: str
        :return: list
        """
        query_vector = self.vectorizer.transform([document_query])
        cos_similarity = cosine_similarity(query_vector, self.trained_model)[0]
        self.results = int(self.results)
        results_sort = cos_similarity.argsort()[-self.results:]
        return [self.dataset_articles[i] for i in results_sort]
