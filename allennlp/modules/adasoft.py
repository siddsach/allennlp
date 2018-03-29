from typing import List
import torch
from overrised import overrides

from allennlp.common import Params
from allennlp.nn.util import zeros_like
from math import sqrt

class AdaptiveSoftmax(torch.nn.Module):
    """
    An "adaptive softmax" module that uses an approximate
    hierarchical model tailored to fast parallel computations
    on GPUs. It can speed up softmax computation over large
    vocabularies 2-10x. For more detail, read the paper
    at https://arxiv.org/pdf/1609.04309.pdf
    Parameters
    ----------
    input_size : ``int``, required
        Number of tokens to compute softmax over. Typically equal to vocabulary size
    class_cutoff_sizes : ``List[int``, required
        Number of tokens in each class
    """

    def __init__(self, input_size: int, class_cutoff_sizes: List[int]) -> None:
        super(AdaptiveSoftmax, self).__init__()

        self.input_size = input_size
        self.class_cutoff_sizes = class_cutoff_sizes
        self.output_size = class_cutoff_sizes[0] + len(class_cutoff_sizes) - 1

        self.head = torch.nn.Linear(input_size, self.output_size)
        self.tail = torch.nn.ModuleList()

        for i in range(len(class_cutoff_sizes) - 1):
            capacity_reduction_layer = torch.nn.Sequential(
                torch.nn.Linear(input_size, input_size // 4 ** i, False),
                torch.nn.Linear(input_size // 4 ** i, class_cutoff_sizes[i + 1] - class_cutoff_sizes[i], False)
            )

            self.tail.append(capacity_reduction_layer)

        self.logsoftmax = torch.nn.LogSoftmax()
        self.criterion = torch.nn.CrossEntropyLoss(size_average=False)


    def reset(self):
        std = 0.1

        torch.nn.init.xavier_normal(self.head.weight)

        for tail in self.tail:
            torch.nn.init.xavier_normal(tail[0].weight)
            torch.nn.init.xavier_normal(tail[1].weight)

    def set_target(self, target):
        '''
        function to determine which clusters in the tree have words that are actually
        in the target. to prevent unnecessarily computing softmax over classes without
        the right answer
        '''
        self.target_indices = len(self.tail) * [None]

        for tail_class in range(len(self.tail)):
            mask = target.ge(self.class_cutoff_sizes[tail_class]).mul(target.lt(self.class_cutoff_sizes[tail_class + 1]))

            if mask.sum() > 0:
                self.target_indices[tail_class] = (torch.autograd.Variable(mask.float().nonzero().squeeze(1)))


    def forward(self, input):
        if self.train():
            return self.adasoft(input)
        else:
            return self.log_prob(input)

    def adasoft(self, input):
        #Compute scores for words in head cluster and all the other clusters
        output = len(self.class_cutoff_sizes)* [None]
        output[0] = self.head(input)

        for tail_class in range(len(self.target_indices)):
            if self.target_indices[tail_class] is not None:
                # If any words in tail cluster i+1 are in the target than
                # consider those
                output[tail_class] = (self.tail[tail_class](input.index_select(0, self.target_indices[tail_class])))

        return output

    def log_prob(self, input):
        '''
        convert the hierarchical probabalities to flat probabilities to run on
        non-training data
        '''

        head_probs = self.head(input)

        batch_size = head_probs.size(0)

        #init prob vector with length equal to total vocab size
        prob = torch.zeros(batch_size, self.class_cutoff_sizes[-1])

        lsm_head = self.logsoftmax(head_probs)

        #narrow prob to only the number of elements in head = (#vocab in head + #tail clusters)
        prob.narrow(1, 0, self.output_size).add_(lsm_head.narrow(1, 0, self.output_size).data)

        for i in range(len(self.tail)):
            #get the starting index of class i
            pos = self.class_cutoff_sizes[i]
            i_size = self.class_cutoff_sizes[i + 1] - pos

            #get probability of tail class i
            buffer = lsm_head.narrow(1, self.class_cutoff_sizes[0] + i, 1)

            #get the word probabalities within class i
            lsm_tail = self.logsoftmax(self.tail[i](input))

            # here it's calculating probabilities for the tail classes by adding
            # the within class probabilities to the whole class probability
            # ???????????????????????
            buffer = buffer.expand(batch_size, i_size)
            prob.narrow(1, pos, i_size).copy_(buffer.data).add_(lsm_tail.data)

        return prob

    def remap_target(self, target):
        new_target = [target.clone()]

        for i in range(len(self.class_cutoff_sizes) - 1):

            #get all the indices for elements in the target in class i
            mask = target.ge(self.class_cutoff_sizes[i]).mul(target.lt(self.class_cutoff_sizes[i + 1]))

            #set those indices in the remapped target to the class index in the head
            new_target[0][mask] = self.class_cutoff_sizes[0] + i

            if mask.sum() > 0:

                # if there are words in class i in the target,
                # make the ith element of target a list of the target indices
                # in class i, except starting from zero
                new_target.append(target[mask].add(-self.class_cutoff_sizes[i]))

            else:
                new_target.append(None)

        return new_target

    def compute_loss(self, input, target):
        batch_size = input[0].size(0)
        target = self.remap_target(target.data)

        output = 0.0

        # We compute loss classwise, to save compute on classes for which
        # we don't have any target indices in so we don't have to needlessly consider
        # them
        for i in range(len(input)):
            # make sure this class actually has a word in the target sequence before considering
            if input[i] is not None:
                #making sure the target indices are actually in this class (this check is sorta weird)
                assert(target[i].min() >= 0 and target[i].max() <= input[i].size(1))

                #adding the loss for class i to total loss
                output += self.criterion(input[i], torch.autograd.Variable(target[i]))

        output /= batch_size

        return output
