
# pylint: disable=no-self-use,invalid-name

import unittest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.sentence_splitter import NltkSentenceSplitter, SpacySentenceSplitter

class TestNltkSentenceSplitter(AllenNlpTestCase):
    def setUp(self):
        super(TestNltkSentenceSplitter, self).setUp()

    def test_handle_simple_punctuation(self):
        text = "This is a sentence. This is another sentence! Should I write a third?"

        sent_splitter = NltkSentenceSplitter()
        expected_split = ["This is a sentence.", "This is another sentence!", "Should I write a third?"]
        split_sents = sent_splitter.split_sents(text)
        assert split_sents == expected_split, "GOT: \n" + str(split_sents) + "\nINSTEAD OF: \n" + expected_split

    def test_add_abbrev(self):
        text = "An ambitious campus expansion plan was proposed by Fr. Vernon F. Gallagher in 1952. Assumption Hall, the first student dormitory, was opened in 1954, and Rockwell Hall was dedicated in November 1958, housing the schools of business and law. It was during the tenure of F. Henry J. McAnulty that Fr. Gallagher's ambitious plans were put to action."
        abbreviation = ['f', 'fr', 'k']
        sent_splitter = NltkSentenceSplitter(ignore_abbreviation = abbreviation)
        split_sents = sent_splitter.split_sents(text)
        expected_split = ['An ambitious campus expansion plan was proposed by Fr. Vernon F. Gallagher in 1952.', 'Assumption Hall, the first student dormitory, was opened in 1954, and Rockwell Hall was dedicated in November 1958, housing the schools of business and law.', "It was during the tenure of F. Henry J. McAnulty that Fr. Gallagher's ambitious plans were put to action."]


        assert split_sents == expected_split, "GOT: \n" + str(split_sents) + "\nINSTEAD OF: \n" + expected_split

    def test_train_on_input(self):
        text = "An ambitious campus expansion plan was proposed by Fr. Vernon F. Gallagher in 1952. Assumption Hall, the first student dormitory, was opened in 1954, and Rockwell Hall was dedicated in November 1958, housing the schools of business and law. It was during the tenure of F. Henry J. McAnulty that Fr. Gallagher's ambitious plans were put to action."
        word_splitter = NltkSentenceSplitter(train_on_input = True)
        split_sents = word_splitter.split_sents(text)
        expected_split = ['An ambitious campus expansion plan was proposed by Fr. Vernon F. Gallagher in 1952.', 'Assumption Hall, the first student dormitory, was opened in 1954, and Rockwell Hall was dedicated in November 1958, housing the schools of business and law.', "It was during the tenure of F. Henry J. McAnulty that Fr. Gallagher's ambitious plans were put to action."]


        assert split_sents == expected_split, "GOT: \n" + str(split_sents) + "\nINSTEAD OF: \n" + expected_split


class TestSpacySentenceSplitter(AllenNlpTestCase):
    def setUp(self):
        super(TestSpacySentenceSplitter, self).setUp()

    def test_handle_simple_punctuation(self):
        text = "This is a sentence. This is another sentence! Should I write a third?"

        sent_splitter = SpacySentenceSplitter()
        expected_split = ["This is a sentence.", "This is another sentence!", "Should I write a third?"]
        split_sents = sent_splitter.split_sents(text)
        assert split_sents == expected_split, "GOT: \n" + str(split_sents) + "\nINSTEAD OF: \n" + str(expected_split)
        self.word_splitter = SpacySentenceSplitter()

    def test_use_parse(self):
        text = "I sleep about 16 hours per day. I fall asleep accidentally. I'm tired, I have headache and I have problems with concentration. I don't eat anything and than I'm eating a lot."
        expected_split = ["I sleep about 16 hours per day.", "I fall asleep accidentally.", "I'm tired,", "I have headache", "and I have problems with concentration.", "I don't eat anything and than I'm eating a lot."]
        sent_splitter = SpacySentenceSplitter(use_parse=True)
        split_sents = sent_splitter.split_sents(text)
        assert split_sents == expected_split, "GOT: \n" + str(split_sents) + "\nINSTEAD OF: \n" + str(expected_split)
        self.word_splitter = SpacySentenceSplitter()

if __name__ == '__main__':
    unittest.main()
