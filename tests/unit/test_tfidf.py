from jina import Executor

def test_tfidf():
    encoder = Executor.load_config('../../config.yml')
    assert encoder.path_vectorizer.endswith('tfidf_vectorizer.pickle')

