"""
This file was taken from: https://github.com/bharat-b7/MultiGarmentNetwork
Author: Bharat
"""

import os
import sys
import numpy as np
from psbody.mesh import Mesh
from os.path import join
import pickle as pkl
from evaluation.lib.serialization import backwards_compatibility_replacements, load_model
import scipy.sparse as sp

## Set your paths here
ROOT = './body_models/misc'

class SmplPaths:
    def __init__(self, project_dir='', exp_name='', gender='neutral', garment=''):
        self.project_dir = project_dir
        # experiments name
        self.exp_name = exp_name
        self.gender = gender
        self.garment = garment

    def get_smpl_file(self):
        if self.gender == 'neutral':
            return join(ROOT,
                        'smpl_models',
                        'neutral',
                        'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

        else:
            smpl_file = join(ROOT,
                             'smpl_models',
                             self.gender,
                             'model.pkl')


            return smpl_file

    def get_smpl(self):
        smpl_m = load_model(self.get_smpl_file())
        smpl_m.gender = self.gender
        return smpl_m
