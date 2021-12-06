import torch
import numpy as np

from torch.utils.data import Sampler, BatchSampler

class GroupedFixedSampler(Sampler):
    '''Batch sampler that returns batches of unshuffled group indices.
       Used for creating val/test dataloaders.
    '''

    def __init__(self, indices, batch_size=16):
        ''' Initialize the sampler.

        Args:
            indices (list of list of int): indices for group instances
            batch_size (int): batch size of the sampler
        '''
        self.batch_size = batch_size
        num_samples = 0
        for group_indices in indices:
            num_samples += len(group_indices)

        self.num_samples = num_samples
        self.batch_cnt = 0

        self.batch_indices= []

        for group_indices in indices:
            for i in range(0, len(group_indices), self.batch_size):
                curr_batch_size = min(self.batch_size, len(group_indices) - i)
                self.batch_indices.extend([(idx, curr_batch_size) for idx in group_indices[i:i+curr_batch_size]])

                self.batch_cnt += 1

    def __iter__(self):
        ''' Returns an iterator for batches of data indices
        '''
        return iter(self.batch_indices)

    def __len__(self):
        ''' Returns the total number of batches.
        '''
        return self.batch_cnt


class GroupedRandomSampler(Sampler):
    '''Sampler that First randomly shuffles groups, then randomly shuffles elements within each group.
       This sampler is for line 5 of Alg. 2. In our case, a group is a set of subject/cloth-type specific
       samples.
    '''

    def __init__(self, indices, max_batch_size=32):
        ''' Initialize the sampler

        Args:
            indices (list of list of int): indices for group instances
            max_batch_size (int): maxmimum number of samples in a batch
        '''
        self.indices = indices
        self.max_batch_size = max_batch_size
        num_samples = 0
        for group_indices in indices:
            num_samples += len(group_indices)

        self.num_samples = num_samples
        self.batch_cnt = 0

    def __iter__(self):
        ''' Returns an iterator for shuffled indices
        '''
        # Step 1: shuffle within groups
        shuffled_within_group = []
        for group_indices in self.indices:
            shuffled_within_group.append([group_indices[i] for i in torch.randperm(len(group_indices))])

        # Step 2: randomly sample batch sizes, and combine batches from different groups
        shuffled_indices = []
        group_cnt = [0] * len(shuffled_within_group)
        grp_inds = list(range(len(shuffled_within_group)))
        self.batch_cnt = 0
        while len(shuffled_indices) < self.num_samples and len(grp_inds) > 0:
            grp_idx = np.random.choice(grp_inds)  # randomly sample a group
            cnt = group_cnt[grp_idx]
            if self.max_batch_size <= 0:
                batch_size = np.random.randint(len(shuffled_within_group[grp_idx]) - cnt) + 1  # randomly sample a batch_size in [1, # of remaining frames for the subject]
            else:
                batch_size = np.random.randint(self.max_batch_size) + 1  # randomly sample a batch_size in [1, batch_size]

            group_cnt[grp_idx] += batch_size
            batch_size = min(batch_size, len(shuffled_within_group[grp_idx]) - cnt)

            if group_cnt[grp_idx] >= len(shuffled_within_group[grp_idx]):
                grp_inds.remove(grp_idx)

            shuffled_indices.extend([(idx, batch_size) for idx in shuffled_within_group[grp_idx][cnt:cnt+batch_size]])

            self.batch_cnt += 1

        return iter(shuffled_indices)

    def __len__(self):
        ''' Returns the total number of data frames. Note that this is NOT the actual
            number of batches.
        '''
        return self.num_samples


class GroupedBatchSampler(BatchSampler):
    '''Batch sampler that returns batches of shuffled group indices.
    '''

    def __init__(self, sampler, drop_last=False):
        ''' Initialize the sampler

        Args:
            sampler (GroupedRandomSampler): sampler which generates shuffled group indices
            drop_last (bool): same as PyTorch dataloader drop_last option
        '''
        self.sampler = sampler
        self.drop_last = drop_last

    def __iter__(self):
        ''' Returns an iterator for batches of data indices
        '''
        batch = []
        for (idx, batch_size) in self.sampler:
            batch.append(idx)
            if len(batch) == batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        ''' Returns the total number of data frames. Note that this is 0 before self.sampler
            is iterated.
        '''
        return self.sampler.batch_cnt
