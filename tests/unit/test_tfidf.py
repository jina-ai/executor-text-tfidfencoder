import os
import numpy as np
import scipy

from jina import Executor, Document, DocumentArray

try:
    from tfidf_text_executor import TFIDFTextEncoder
except:
    from jinahub.encoder.tfidf_text_executor import TFIDFTextEncoder



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
    assert expected.shape[0] == 1


def test_tfidf_text_encoder_batch():
    # Input
    text_batch = ['Han likes eating pizza', 'Han likes pizza', 'Jina rocks']

    # Encoder embedding
    encoder = TFIDFTextEncoder()
    doc0 = Document(text=text_batch[0])
    doc1 = Document(text=text_batch[1])
    doc2 = Document(text=text_batch[2])
    docarray = DocumentArray([doc0, doc1, doc2])
    encoder.encode(docarray, parameters={})
    embeddeding_batch = scipy.sparse.vstack(docarray.get_attributes('embedding'))

    # Compare with ouptut
    expected_batch = scipy.sparse.load_npz(os.path.join(cur_dir, 'expected_batch.npz'))
    np.testing.assert_almost_equal(
        embeddeding_batch.todense(), expected_batch.todense(), decimal=2
    )
    assert expected_batch.shape[0] == len(text_batch)
