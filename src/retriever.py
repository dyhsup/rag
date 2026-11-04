"""Per-question BM25 indexes over passages and over sentences."""
from rank_bm25 import BM25Okapi

from .data import tokenize


class QuestionIndex:
    def __init__(self, question_row, k1=1.5, b=0.75):
        self.qid = question_row["qid"]
        self.docs = question_row["docs"]
        self.doc_ids = [d["doc_id"] for d in self.docs]

        self.bm25_doc = BM25Okapi(
            [tokenize(d["text"]) for d in self.docs], k1=k1, b=b
        )

        # sentence-level index: (doc_idx, sent_idx) -> row
        self.sent_map = []
        sents = []
        for di, d in enumerate(self.docs):
            for si, s in enumerate(d["sentences"]):
                self.sent_map.append((di, si))
                sents.append(tokenize(s))
        self.bm25_sent = BM25Okapi(sents, k1=k1, b=b)
        self.n_sents = len(sents)

    def score_docs(self, text):
        """Whole-query BM25 over passages."""
        return list(self.bm25_doc.get_scores(tokenize(text)))

    def score_sentences(self, text):
        """BM25 of a claim against every sentence in the pool."""
        return list(self.bm25_sent.get_scores(tokenize(text)))

    def sentence_scores_to_docs(self, sent_scores):
        """max over a document's sentences."""
        best = [float("-inf")] * len(self.docs)
        for row, (di, _si) in enumerate(self.sent_map):
            if sent_scores[row] > best[di]:
                best[di] = sent_scores[row]
        return [0.0 if b == float("-inf") else b for b in best]
