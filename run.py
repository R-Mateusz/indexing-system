import sys
from pathlib import Path
import models


def main():
    """
    Main functionality of IndexingSystem application
    """

    while True:
        print("Write 'exit' to exit the application.\n")

        query_input = input("Write query prompt: ")

        if query_input == 'exit':
            sys.exit(1)

        else:
            dataset_name = input("What dataset you want to search for? ")
            indexer_results = input("How many results do you want to find? (max 3) :")

            if indexer_results not in ['1', '2', '3']:
                print("You can find maximum 3 fragments")
                continue

            print("\n")
            if dataset_name and indexer_results:
                csv_path = get_path_of_dataset(dataset_name)

                if not csv_path.exists():
                    print(f"{csv_path} doesn't exist. Try again")
                    continue

                else:
                    indexer = models.ArticleIndex(indexer_results, csv_path)

                    if indexer:
                        article_retrieval_system = models.ArticleRetriever(indexer)

                        if article_retrieval_system:
                            fragments = article_retrieval_system.retrieve_fragments(query_input)

                            if fragments:
                                for fragment in fragments:
                                    print(fragment + "\n")


def get_path_of_dataset(dataset_name: str) -> str:
    """
    Get name of dataset and returns path to it

    :param dataset_name: str
    :return: WindowsPath
    """
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir / f'datasets/{dataset_name}'
    return csv_path


if __name__ == "__main__":
    main()
