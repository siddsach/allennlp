from typing import List, Dict
import torch

#from allennlp.common import Params

class AdaptiveSoftmax(torch.nn.Module):
    """
    An "adaptive softmax" module that uses an approximate
    hierarchical model tailored to fast parallel computations
    on GPUs. It can speed up softmax computation over large
    vocabularies 2-10x. For more detail, read the paper
    at https://arxiv.org/pdf/1609.04309.pdf
            # assumes indices start from 0 with decreasing frequency
    Parameters
    ----------
    input_size : ``int``, required
        Number of tokens to compute softmax over. Typically equal to vocabulary size
    class_cutoff_sizes : ``List[int]``, required
        Number of tokens in each class
    """

    def __init__(self, input_size: int, vocab_size: int, class_cutoff_sizes: List[int], tail_projection_ratio: int = 4) -> None:
        super(AdaptiveSoftmax, self).__init__()

        self.input_size = input_size
        self.class_cutoff_sizes = class_cutoff_sizes + [vocab_size]
        self.output_size = class_cutoff_sizes[0] + len(self.class_cutoff_sizes) - 1

        self.head = torch.nn.Linear(input_size, self.output_size)
        self.tail = torch.nn.ModuleList()

        for tail_class_i in range(len(self.class_cutoff_sizes)-1):
            capacity_reduction_layer = torch.nn.Sequential(
                torch.nn.Linear(input_size, input_size // tail_projection_ratio ** tail_class_i, False),
                torch.nn.Linear(input_size // tail_projection_ratio ** tail_class_i, self.class_cutoff_sizes[tail_class_i + 1] - self.class_cutoff_sizes[tail_class_i], False)
            )

            self.tail.append(capacity_reduction_layer)

        self.logsoftmax = torch.nn.LogSoftmax()
        self.criterion = torch.nn.CrossEntropyLoss(reduce = False, size_average=False)


    def reset(self):
        torch.nn.init.xavier_normal(self.head.weight)

        for tail in self.tail:
            torch.nn.init.xavier_normal(tail[0].weight)
            torch.nn.init.xavier_normal(tail[1].weight)

    def set_target(self, target: torch.LongTensor) -> None:
        '''
        input_shape: (batch_size, 1)

        function to determine which clusters in the tree have words that are actually
        in the target ouput. This allows us to save computation in training by only
        computing loss for the clusters that contribute some loss
        '''
        self.target_indices = len(self.tail) * [None]

        for tail_class_i in range(len(self.tail)):
            # find elements with indices in this class
            # (batch_size, 1)
            mask = target.data.ge(self.class_cutoff_sizes[tail_class_i]) \
                         .mul(target.data.lt(self.class_cutoff_sizes[tail_class_i + 1]))

            #Check if this batch has targets in tail_class_i
            if mask.sum() > 0:
                #(mask.float().nonzero().sum(), 1)
                self.target_indices[tail_class_i] = (torch.autograd.Variable(mask.float().nonzero().squeeze(1)))


    def forward(self, input):
        if self.train():
            return self.adasoft(input)
        else:
            return self.log_prob(input)

    def adasoft(self, input:torch.Tensor) -> List[torch.Tensor]:
        '''
        input_shape: (batch_size, hidden_size)
        output_shape: [(batc
        [(batch_size, class_cutoff_sizes[0]), (batch_size, class_cutoff_sizes[1]-class_cutoff_sizes[0]), ..., (batch_size, vocab_size- class_cutoff_sizes[-1])]

        Computes projection from hidden state to class scores, where
        the class scores are sorted in a list of classes based on
        word indices (assumed to be sorted by indices) and
        '''

        #Compute scores for words in head cluster and all the other clusters
        output = len(self.class_cutoff_sizes)* [None]
        output[0] = self.head(input)

        for tail_class_i in range(len(self.target_indices)):
            if self.target_indices[tail_class_i] is not None:

                # If any words in tail cluster i+1 are in the target than
                # consider those
                # (num_rows_with_targets_in_class_i, hidden_size)
                rows_with_targets_in_class_i= input.index_select(0, self.target_indices[tail_class_i])

                #Only compute projection on examples with targets in this class
                # (num_rows_with_targets_in_class_i, num_vocab_in_class_i)
                output[tail_class_i+1] = (self.tail[tail_class_i](rows_with_targets_in_class_i))

        return output

    def log_prob(self, input: torch.Tensor) -> torch.Tensor:
        '''
        input_shape: (batch_size, hidden_size)
        output_shape: (batch_size, vocab_size)

        convert the adasoft hierarchical probabalities to flat probabilities to run
        on non-training data
        '''

        head_probs = self.head(input)

        batch_size = head_probs.size(0)

        #init prob vector with length equal to total vocab size
        prob = torch.zeros(batch_size, self.class_cutoff_sizes[-1])

        lsm_head = self.logsoftmax(head_probs)

        #add the prababilities in the head
        prob.narrow(1, 0, self.output_size).add_(lsm_head.data)

        for tail_class_i in range(len(self.tail)):
            # get the starting and size of class i
            i_start = self.class_cutoff_sizes[tail_class_i]
            i_size = self.class_cutoff_sizes[tail_class_i + 1] - i_start

            # get log probability of tail class i
            buffer = lsm_head.narrow(1, self.class_cutoff_sizes[0] + tail_class_i, 1)

            # get the word log-probabalities within class i
            lsm_tail = self.logsoftmax(self.tail[tail_class_i](input))

            # adds probabilities because its the log
            buffer = buffer.expand(batch_size, i_size)
            prob.narrow(1, i_start, i_size).copy_(buffer.data).add_(lsm_tail.data)

        return prob

    def _remap_target(self, target: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        '''
        input shape: (batch_size, 1) NOT A VARIABLE
        output shape: List[(batch_size, 1)]

        Converts target tensor to list of tensors for computing loss after adasoft
        classwise
        '''

        new_target = len(self.class_cutoff_sizes) * [None]
        new_target[0] = {"targets": target.clone(), "mask":torch.ones_like(target)}

        for tail_class_i in range(len(self.class_cutoff_sizes) - 1):

            # get all the indices for elements in the target in class i
            # (batch * max_len, 1)
            mask = target.ge(self.class_cutoff_sizes[tail_class_i]) \
                         .mul(target.lt(self.class_cutoff_sizes[tail_class_i + 1]))

            if mask.sum() > 0:
                new_target[0]["targets"][mask] = self.class_cutoff_sizes[0] + tail_class_i

                # set those indices in the remapped target to the class index in the head

                # if there are words in class i in the target,
                # make the ith element of target a 1-d tensor of the target indices
                # in class i, except with the indices starting from zero
                # (num_rows_with_targets_in_class_i)
                targets_in_class_i = target[mask].add(-self.class_cutoff_sizes[tail_class_i])
                new_target[tail_class_i+1] = {"targets": targets_in_class_i, "mask":mask}

        return new_target

    def compute_loss(self, class_scores: List[torch.Tensor],
                           target: List[torch.Tensor]) -> torch.Tensor:
        '''
        log_probs: [(batch_size, num_classes), (batch_size, num_classes)]
        target: [(batch_size, 1), (batch_size, 1)...]
        '''

        # (batch_size, 1)
        negative_log_likelihood = torch.autograd.Variable(torch.zeros(target.size(0)))

        # [(batch_size, 1), (batch_size, 1)...]
        target = self._remap_target(target.data)

        # We compute loss classwise, to save compute on classes for which
        # we don't have any target indices in so we don't have to needlessly consider
        # them
        for tail_class_i in range(len(class_scores)):
            # make sure this class actually has a word in the target sequence before considering
            if class_scores[tail_class_i] is not None:
                target_indices = target[tail_class_i]["targets"]
                target_mask = target[tail_class_i]["mask"]

                #making sure the target indices are actually in this class
                upper = class_scores[tail_class_i].size(1)
                assert target_indices.min() >= 0 and target_indices.max() <= upper, \
                        "Target indices out of range!" + str(target_indices)

                target_var = torch.autograd.Variable(target_indices)

                # adding the loss for class i to total loss
                # adds probabilities because its the log!
                # (batch_size)
                this_loss = self.criterion(class_scores[tail_class_i], target_var)
                negative_log_likelihood[target_mask] += this_loss

        return negative_log_likelihood
