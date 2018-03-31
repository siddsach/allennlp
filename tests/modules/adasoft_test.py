# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable
from torch.distributions import *

import unittest
import allennlp
from allennlp.common import Params
from allennlp.modules.adasoft import AdaptiveSoftmax
from allennlp.common.testing import AllenNlpTestCase


class TestAdaptiveSoftmax(AllenNlpTestCase):

    def setUp(self):
        super(TestAdaptiveSoftmax, self).setUp()
        self.batch_size = 10
        self.vocab_size = 10000
        self.hidden_size = 200
        self.class_cutoff_sizes = [500, 5000]
        self.adasoft = AdaptiveSoftmax(self.hidden_size, self.vocab_size, self.class_cutoff_sizes)

        zipf_probs = 1/torch.Tensor(range(1, self.vocab_size+1))
        zipf_distr = Categorical(probs = zipf_probs)
        self.inp = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            self.inp = zipf_distr.sample()

    def test_set_target(self):
        self.adasoft.set_target(self.target)

        for c in self.class_cutoff_sizes:
            #get indices in class c
            indices_in_this_class = (self.target > self.class_cutoff_sizes[c]) * (self.target < self.class_cutoff_sizes[c+1])

            #Check that we're considering the right classes
            assert bool(self.adasoft.target_indices[c]) == bool(indices_in_this_class.sum())
            if self.adasoft.target_indices[c]:
                assert self.adasoft.target_indices[c].data == indices_in_this_class.float().nonzero().squeeze(1)

    #def test_adasoft_projection(self):



if __name__ == '__main__':
    unittest.main()





