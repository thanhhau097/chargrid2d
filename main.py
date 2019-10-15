#!/usr/bin/python

from __future__ import print_function
import argparse, os, json, traceback, sys

import sys 
import os
import time 
import argparse
import random
import datetime
import subprocess


class SagemakerInference(object):
    """ Configurations for setup env and training models 

        Arguments:
            config_path (str): path of configuration json file

    """
    def __init__(self):
        # read the config file to a config dict
        super().__init__()
        
    def process(self):
        os.system('nvidia-smi')
        print("RUN TRAIN.PY FILE")
        subprocess.run('python train.py --size 1024 --root /opt/ml/input/data/train --batch-size 8', shell=True)


if __name__ == '__main__':
    inference = SagemakerInference()

    try:
        inference.process()
        sys.exit(0)
    except Exception as e:
        # Write out an error file. This will be returned as 
        # the failure reason in the describe training job result.
        trc = traceback.format_exc()
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        sys.exit(255)
