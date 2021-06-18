import os

from jina import Executor, requests, DocumentArray
from jina.excepts import PretrainedModelFileDoesNotExist

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TFIDFTextEncoder(Executor):

    def __init__(
        self,
        path_vectorizer: str = os.path.join(cur_dir, 'model/tfidf_vectorizer.pickle'),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.path_vectorizer = path_vectorizer

        import os
        import pickle

        if os.path.exists(self.path_vectorizer):
            self.tfidf_vectorizer = pickle.load(open(self.path_vectorizer, 'rb'))
        else:
            raise PretrainedModelFileDoesNotExist(
                f'{self.path_vectorizer} not found, cannot find a fitted tfidf_vectorizer'
            )

    @requests
    def encode(self,docs: DocumentArray,  *args, **kwargs) -> DocumentArray:
        iterable_of_texts = docs.get_attributes('text')
        embedding_matrix = self.tfidf_vectorizer.transform(iterable_of_texts)

        for doc, doc_embedding in zip(docs, embedding_matrix):
            doc.embedding = doc_embedding