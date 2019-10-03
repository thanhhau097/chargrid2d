import torch


class OneHotEncoder():
    def __init__(self, corpus):
        self.corpus = corpus
        self.classes = len(self.corpus) + 1

    def process(self, input):
        one_hot = torch.FloatTensor(self.classes, input.size(1), input.size(2)).zero_()
        target = one_hot.scatter_(0, input.data, 1)

        return target