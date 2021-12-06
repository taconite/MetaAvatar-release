import os
import logging
import time

from tensorboardX import SummaryWriter

def create_logger(out_dir, phase='train', create_tf_logs=True):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_log_file = os.path.join(out_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    if create_tf_logs:
        writter = SummaryWriter(os.path.join(out_dir, 'logs'))
    else:
        writter = None

    return logger, writter
