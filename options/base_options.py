import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--model_name', type=str, choices=['ResNet50','SeNet50','ResNet34', 'ResNet18'],
            help='the created model name')
        self._parser.add_argument('--dataset_name', type=str, choices=['FER', 'FERPlus'],
            help='the created dataset name')
        self._parser.add_argument('--batch_size', type=int, default= 20, 
            help='input batch size per task')
        self._parser.add_argument('--image_size', type=int, default= 224, 
        help='input image size') # reducing iamge size is acceptable
        self._parser.add_argument('--freeze_before', type=int, default=-1, 
            help='freeze layers before the last layer (by default -1). If set to 0, means no to freeze any layer.')
        self._parser.add_argument('--dropout_rate', type=float, default=0.3, 
            help='dropout ratio for the final dense layer')
        self._parser.add_argument('--activation', type=str, choices=['softmax', 'sigmoid'], default='softmax',
            help='activation function for the final dense layer')
        self._parser.add_argument('--reg_func', type=str, default=None, choices = ['l1', 'l2'],
            help='regularization function. (default is not to apply regularization)')
        self._parser.add_argument('--reg_layers', type=str, default=None, nargs='+', 
            help='layers to apply regularization. (default is to apply regularization to all layers if reg_func is not None)')
        self._parser.add_argument('--reg_bias',  action='store_true', 
            help='regularize the bias term')
        self._parser.add_argument('--num_gpus', type=int, default= 1, 
            help='number of gpus')
        self._parser.add_argument('--n_threads_train', default=8, type=int, 
            help='# threads for loading data')
        self._parser.add_argument('--n_threads_test', default=8, type=int, 
            help='# threads for loading data')
        self._parser.add_argument('--name', type=str, default='experiment_1', 
            help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
            help='checkpoints are saved here')
        self._parser.add_argument('--output_logs_dir', type=str, default='./loggings', 
            help='loggings are saved here')
        self._parser.add_argument('--checkpoint_path', type=str, default='',
            help='if set, the model is loaded from checkpoint_path.')
        self._parser.add_argument('--remove_prev_exp', action='store_true')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or test
        self._opt.is_train = self.is_train

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        log_dir = os.path.join(self._opt.output_logs_dir, self._opt.name)
        print(expr_dir)
        if self._opt.remove_prev_exp:
            os.system('rm -rf {}'.format(expr_dir))
            os.system('rm -rf {}'.format(log_dir))
        if self.is_train:
            os.makedirs(expr_dir)
        else:
            assert os.path.exists(expr_dir)
        if self.is_train:
            os.makedirs(log_dir)
        else:
            assert os.path.exists(log_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
