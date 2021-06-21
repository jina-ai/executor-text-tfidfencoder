import os
import numpy as np
import scipy

from jina import Executor, Document, DocumentArray
from tfidf_text_executor import TFIDFTextEncoder


cur_dir = os.path.dirname(os.path.abspath(__file__))

def test_tfidf():
    encoder = Executor.load_config(os.path.join(cur_dir, '../../config.yml'))
    assert encoder.path_vectorizer.endswith('tfidf_vectorizer.pickle')


def test_tfidf_text_encoder():
    # Input
    text = 'Han likes eating pizza'

    # Encoder embedding
    encoder = TFIDFTextEncoder()
    doc = Document(text=text)
    docarray = DocumentArray([doc])
    encoder.encode(docarray, parameters = {})
    embeddeding = doc.embedding

    # Compare with ouptut
    expected = scipy.sparse.load_npz(os.path.join(cur_dir, 'expected.npz'))
    np.testing.assert_almost_equal(embeddeding.todense(), expected.todense(), decimal=4)
    assert expected.shape[0] == len(text)

