from .FER_dataset import FER_dataset, FERPlus_dataset

class DatasetFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(opt, train_mode='Train'):
        if opt.dataset_name == 'FERPlus':
            dataset = FERPlus_dataset(opt, train_mode)
        elif opt.dataset_name == 'FER':
            dataset = FER_dataset(opt, train_mode)
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_name)
        print('Dataset {} ({}) was created: number of samples {}'.format(dataset._name, train_mode, len(dataset)))
        return dataset

