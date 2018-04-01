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
    def __init__(self, train_on_input: bool = False, ignore_abbreviation: List[str] = None) -> None:
        # Import is here because it's slow, and by default unnecessary.
        from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

        if ignore_abbreviation is not None:
            punktparams = PunktParameters()
            punktparams.abbrev_types = set(ignore_abbreviation)
            self.tokenizer = PunktSentenceTokenizer(punktparams)
        else:
            self.tokenizer = PunktSentenceTokenizer()

        self.train_on_input = train_on_input

    @overrides
    def split_sents(self, text: str) -> List[str]:
        if self.train_on_input:
            self.tokenizer.train(text)
        return self.tokenizer.tokenize(text)

    @classmethod
    def from_params(cls, params: Params) -> 'SentenceSplitter':
        train_on_input = params.pop_bool('train_on_input', False)
        ignore_abbreviation = params.pop('ignore_abbreviation', None)
        params.assert_empty(cls.__name__)
        return cls(train_on_input, ignore_abbreviation)


@SentenceSplitter.register('spacy_sent')
class SpacySentenceSplitter(SentenceSplitter):
    """
    A ``SentenceSplitter`` that uses spaCy's tokenizer.
    """

    def __init__(self, language: str = 'en_core_web_sm', use_parse: bool = True) -> None:

        if use_parse:
            self.spacy = get_spacy_model(language, True, True, False)
        else:
            self.spacy = get_spacy_model(language, False, False, False)
            self.spacy.add_pipe(self.spacy.create_pipe('sentencizer'))


    @overrides
    def split_sents(self, text: str) -> List[Token]:
        return [s.text for s in self.spacy(text).sents]

    @classmethod
    def from_params(cls, params: Params) -> 'SentenceSplitter':
        language = params.pop('language', 'en_core_web_sm')
        use_parse = params.pop('use_parse', False)
        params.assert_empty(cls.__name__)
        return cls(language, use_parse)
