import argparse

import torch

from model import Chargrid2D

class PredictProcedure():
    def __init__(self, model_path, **kwargs):
        self.model = Chargrid2D(302, 25)
        self.model.load_state_dict(torch.load(model_path))
        self.devide = kwargs['device']
        self.model.to(self.devide)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--gpu', type=bool)

    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu
    model_path = args.model_path

    predictor = PredictProcedure(model_path, {'device': device})