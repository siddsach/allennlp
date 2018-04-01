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
        self.inp = Variable(torch.randn(self.batch_size, self.hidden_size))
        self.target = Variable(torch.LongTensor(self.batch_size))
        self.target[:] = 0
        for i in range(self.batch_size):
            self.target[i] = zipf_distr.sample()[0]


    def test_set_target(self):
        self.adasoft.set_target(self.target)

        for i, c in enumerate(self.class_cutoff_sizes):
            #get indices in class c
            indices_in_this_class = (self.target > c) * (self.target < self.adasoft.class_cutoff_sizes[i+1])

            #Check that we're considering the right classes
            assert (self.adasoft.target_indices[i] is None) == (indices_in_this_class.data.sum() == 0)
            try:
                if self.adasoft.target_indices[i]:
                    assert (self.adasoft.target_indices[i].data == indices_in_this_class.float().nonzero().squeeze(1)).data[0]
            except:
                ValueError("Got " + str(self.adasoft.target_indices[i]) + "instead of:" + str(indices_in_this_class))


    def test_adasoft_projection(self):
        self.adasoft.set_target(self.target)

        output = self.adasoft.adasoft(self.inp)

        assert output[0].shape == (self.batch_size, self.class_cutoff_sizes[0] + len(self.class_cutoff_sizes)), output[0].shape

        for i, tail_class_i in enumerate(output[1:]):
            if self.adasoft.target_indices[i] is None:
                assert tail_class_i is None, "Expected {} got {}".format(None, tail_class_i)
            else:
                try:
                    expected_shape = (self.adasoft.target_indices[i].size(0), self.adasoft.class_cutoff_sizes[i+1] - self.adasoft.class_cutoff_sizes[i])


                    assert tail_class_i.shape == expected_shape
                except:
                    ValueError('Expected {} Got {}'.format(expected_shape, tail_class_i.shape))

    def test_logprob(self):
        probs = self.adasoft.log_prob(self.inp)
        assert probs.shape == (self.batch_size, self.vocab_size)
        sum_to_one = probs.exp().sum(dim=1)
        assert_almost_equal((sum_to_one-torch.ones(self.batch_size)).sum(), 0)

    def test_remap_target(self):
        targ = self.target.data
        remapped_target = self.adasoft._remap_target(targ)

        remapped_target_indices = [c["targets"] if c else None for c in remapped_target]

        assert remapped_target_indices[0].shape == (self.batch_size,)

        for i, c in enumerate(self.class_cutoff_sizes):
            #get indices in class c
            indices_in_this_class = (targ > c) * (targ < self.adasoft.class_cutoff_sizes[i+1])

            #Check that we're considering the right classes
            assert (remapped_target_indices[i+1] is None) == (indices_in_this_class.sum() == 0)
            if remapped_target_indices[i+1] is not None:
                expected_indices_in_this_class = remapped_target_indices[i+1] + c
                assert (expected_indices_in_this_class == self.target[indices_in_this_class].data).sum() == len(expected_indices_in_this_class)

    def test_compute_loss(self):
        criterion = torch.nn.CrossEntropyLoss()

        self.adasoft.set_target(self.target)
        log_probs = self.adasoft.adasoft(self.inp)
        loss = self.adasoft.compute_loss(log_probs, self.target)
        remapped_target = self.adasoft._remap_target(self.target.data)

        assert loss.shape == (self.batch_size,), loss.shape

        expected_loss = 0

        for i in range(len(log_probs)):
            if log_probs[i] is not None:
                new_target = torch.autograd.Variable(remapped_target[i]["targets"])
                expected_loss += criterion(log_probs[i], new_target)

        try:
            assert_almost_equal(loss.sum(), expected_loss)
        except:
            ValueError("Expected {} Got {}".format(expected_loss, loss.sum()))






