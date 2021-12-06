import yaml
from yaml import Loader

from depth2mesh import data
from depth2mesh import metaavatar

method_dict = {
    'metaavatar': metaavatar,
}

# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Datasets
def get_dataset(mode, cfg, subject_idx=None, cloth_split=None, act_split=None, subsampling_rate=1, start_offset=0):
    ''' Returns the dataset.

    Args:
        mode (str): which mode the dataset is. Can be either train, val or test
        cfg (dict): config dictionary
        subject_idx (int or list of int): which subject(s) to use, None means using all subjects
        cloth_split (list of str): which cloth type(s) to load. If None, will load all cloth types
        cloth_split (list of str): which cloth type(s) to load. If None, will load all cloth types
        act_split (list of str): which action(s) to load. If None, will load all actions
        subsampling_rate (int): frame subsampling rate for the dataset
        start_offset (int): starting frame offset
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    use_aug = cfg['data']['use_aug']
    normalized_scale = cfg['data']['normalized_scale']

    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Get cloth-type and action splits from config file, if they are
    # not specified
    if cloth_split is None:
        cloth_splits = {
            'train': cfg['data']['train_cloth_types'],
            'val': cfg['data']['val_cloth_types'],
            'test': cfg['data']['test_cloth_types'],
        }

        cloth_split = cloth_splits[mode]

    if act_split is None:
        act_splits = {
            'train': cfg['data']['train_action_names'],
            'val': cfg['data']['val_action_names'],
            'test': cfg['data']['test_action_names'],
        }

        act_split = act_splits[mode]

    # Create dataset
    if dataset_type == 'cape_corr':
        input_pointcloud_n = cfg['data']['input_pointcloud_n']
        single_view = cfg['data']['single_view']
        use_raw_scans = cfg['data']['use_raw_scans']
        input_pointcloud_noise = cfg['data']['input_pointcloud_noise']
        keep_aspect_ratio = cfg['model']['keep_aspect_ratio']

        dataset = data.CAPECorrDataset(
            dataset_folder,
            subjects=split,
            mode=mode,
            use_aug=use_aug,
            input_pointcloud_n=input_pointcloud_n,
            single_view=single_view,
            cloth_types=cloth_split,
            action_names=act_split,
            subject_idx=subject_idx,
            input_pointcloud_noise=input_pointcloud_noise,
            use_raw_scans=use_raw_scans,
            normalized_scale=normalized_scale,
            subsampling_rate=subsampling_rate,
            start_offset=start_offset,
            keep_aspect_ratio=keep_aspect_ratio,
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset
