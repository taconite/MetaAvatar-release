import numpy as np
from collections import defaultdict
from tqdm import tqdm


class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, val_loader, val_dataset=None):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            if val_dataset is not None:
                idx = data['idx'].numpy().reshape([-1]).astype(np.int64)
                if idx.size == 1:
                    model_dict = val_dataset.get_model_dict(idx[0])
                else:
                    model_dict = None
            else:
                model_dict = None

            eval_step_dict = self.eval_step(data, model_dict=model_dict)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError
