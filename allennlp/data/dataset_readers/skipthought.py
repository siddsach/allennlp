from typing import Dict
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils.sentence_splitter import SentenceSplitter
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = "@@START@@"
END_SYMBOL = "@@END@@"

@DatasetReader.register("skipthought")
class SkipthoughtDatasetReader(DatasetReader):
    """
    Read a txt file containing a long string of text, and create a dataset suitable for a
    ``Skipthought`` model by tokenizing sentences and and constructing instances, each
    with 3 sentences: the encoded sentence, the sentence preceding it and the sentence
    following it. We use the same tokenizer for each sentence, because each sentence
    is expected to come from the same text distribution.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the sequences.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the sequence.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 sentence_splitter: SentenceSplitter = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_splitter = sentence_splitter
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._add_start_token = add_start_token

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            logger.info("Reading sentences from file at: %s", file_path)
            text = data_file.read()
            before = None
            source = None
            after = None
            for i, sent in enumerate(self._sentence_splitter.split_sents(text)):
                before = source
                source = after
                after = sent
                if i > 2:
                    yield self.text_to_instance(before, source, after)

    @overrides
    def text_to_instance(self, before_string: str, source_string: str, after_string: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._tokenizer.tokenize(source_string)
        tokenized_before = self._tokenizer.tokenize(before_string)
        tokenized_after = self._tokenizer.tokenize(after_string)
        if self._add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
            tokenized_before.insert(0, Token(START_SYMBOL))
            tokenized_after.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        tokenized_before.append(Token(END_SYMBOL))
        tokenized_after.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._token_indexers)
        before_field = TextField(tokenized_before, self._token_indexers)
        after_field = TextField(tokenized_after, self._token_indexers)
        return Instance({"source_tokens":source_field, "before_tokens":before_field, "after_tokens":after_field})

    @classmethod
    def from_params(cls, params: Params) -> 'SkipthoughtDatasetReader':
        tokenizer_type = params.pop('tokenizer', None)
        tokenizer = None if tokenizer_type is None else Tokenizer.from_params(tokenizer_type)
        sentence_splitter_type = params.pop('sentence_splitter', None)
        sentence_splitter = None if sentence_splitter_type is None else SentenceSplitter.from_params(sentence_splitter_type)
        indexers_type = params.pop('token_indexers', None)
        add_start_token = params.pop_bool('add_start_token', True)
        if indexers_type is None:
            token_indexers = None
        else:
            token_indexers = TokenIndexer.dict_from_params(indexers_type)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return SkipthoughtDatasetReader(tokenizer, sentence_splitter, token_indexers, add_start_token, lazy)
