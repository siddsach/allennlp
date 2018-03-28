from allennlp.data.tokenizers.sentence_splitter import SentSplitter

def get_skipthought_triples(SentSplitter: tokenizer, str: text):
    before = None
    source = None
    after = None
    for i, sent in enumerate(tokenizer.split_sents(text)):
        before = source
        source = after
        after = sent
        if i > 2:
            yield before, source, after

