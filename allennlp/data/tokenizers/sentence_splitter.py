from typing import List

from overrides import overrides

from allennlp.common import Params, Registrable
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token


class SentenceSplitter(Registrable):
    """
    A ``SentenceSplitter`` splits strings into sentences.
    """
    default_implementation = 'spacy'

    def batch_split_sents(self, texts: List[str]) -> List[List[Token]]:
        """
        Spacy needs to do batch processing, or it can be really slow.  This method lets you take
        advantage of that if you want.  Default implementation is to just iterate of the sentences
        and call ``split_sents``, but the ``SpacySentenceSplitter`` will actually do batched
        processing.
        """
        return [self.split_sents(text) for text in texts]

    def split_sents(self, sentence: str) -> List[Token]:
        """
        Splits ``sentence`` into a list of :class:`Token` objects.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'SentenceSplitter':
        choice = params.pop_choice('type', cls.list_available(), default_to_first_choice=True)
        return cls.by_name(choice).from_params(params)




@SentenceSplitter.register('nltk_sent')
class NltkSentenceSplitter(SentenceSplitter):
    """
    A ``SentenceSplitter`` that uses nltk's ``sent_tokenize`` method.

    NLTK is actually faster than spacy for sentence tokenization
    because it doesn't use constituency parsing to do sentence
    tokenization, which can be more robust to noi sebut much
    less fast.
    """
    @overrides
    def split_sents(self, sentence: str) -> List[Token]:
        # Import is here because it's slow, and by default unnecessary.
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(sentence.lower())

    @classmethod
    def from_params(cls, params: Params) -> 'SentenceSplitter':
        params.assert_empty(cls.__name__)
        return cls()


@SentenceSplitter.register('spacy_sent')
class SpacySentenceSplitter(SentenceSplitter):
    """
    A ``SentenceSplitter`` that uses spaCy's tokenizer.
    """

    def __init__(self, language: str = 'en_core_web_sm') -> None:

        #We only need the sent tokenization so we pass False to args
        self.spacy = get_spacy_model(language, False, False, False)

    @overrides
    def batch_split_sents(self, texts: List[str]) -> List[List[Token]]:
        return (doc.sents for doc in self.spacy.pipe(texts, n_threads=-1))

    @overrides
    def split_sents(self, text: str) -> List[Token]:
        return [s for s in self.spacy(text).sents]

    @classmethod
    def from_params(cls, params: Params) -> 'SentenceSplitter':
        language = params.pop('language', 'en_core_web_sm')
        params.assert_empty(cls.__name__)
        return cls(language, pos_tags, parse, ner)
