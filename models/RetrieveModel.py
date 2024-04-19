from transformers import BartTokenizer, BartForConditionalGeneration
from .IndexingModel import ArticleIndex

class ArticleRetriever:
    """
    ArticleRetriever class is responsible for retrieving relevant fragments of every articles that were searched
    """

    def __init__(self, articles_index: ArticleIndex):
        """
        Initialization for ArticleRetriever instance

        :param articles_index: ArticleIndex
        """
        self.articles_index = articles_index
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    def retrieve_fragments(self, query: str, fragment_copy=1, max_length=600, min_length=50) ->list[str]:
        """
        Retrieve fragments from articles

        :param query: str
        :param fragment_copy: int
        :param max_length: int
        :param min_length: int
        :return: list
        """
        relevant_articles = self.articles_index.search_articles(query)

        fragments = []

        for article in relevant_articles:
            article = article[:512]
            inputs = self.tokenizer(article, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=fragment_copy,
                early_stopping=True
            )

            for i, output in enumerate(outputs):
                try:
                    fragment = self.tokenizer.decode(output, skip_special_tokens=True)
                    fragments.append(fragment)
                except IndexError:
                    print(f"Error: Index out of range for output {i}")
                    print("Outputs:", outputs)
                    print("Length of outputs:", len(outputs))
        return fragments
