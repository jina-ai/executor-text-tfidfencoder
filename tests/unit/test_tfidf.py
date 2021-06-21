from jina import Executor
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

def test_tfidf():
    encoder = Executor.load_config(os.path.join(cur_dir, '../../config.yml'))
    assert encoder.path_vectorizer.endswith('tfidf_vectorizer.pickle')

