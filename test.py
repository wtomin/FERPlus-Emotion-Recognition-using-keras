from options.test_options import TestOptions
from PATH import PATH
PRESET_VARS = PATH()
from models.CNN import ModelFactory
from data.datasetfactory import DatasetFactory
from data.dataloader import DataLoader
from utils.parallel import make_parallel
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import multiprocessing

class Tester(object):
    def __init__(self):
        self._opt = TestOptions().parse()
        assert len(self._opt.checkpoint_path) !=0 and os.path.exists(self._opt.checkpoint_path), "checkpoint_path does not exist."
        df = DatasetFactory()
        test_dataset = df.get_by_name(self._opt, 'Test')
        self.test_loader = DataLoader(test_dataset,
                                 batch_size = self._opt.batch_size,
                                 shuffle = True, 
                                 drop_last = True, 
                                 num_workers=self._opt.n_threads_test)
        self.num_classes = test_dataset.n_classes
        self.model = ModelFactory().get_by_name(self._opt.model_name, self._opt, self.num_classes)

        if self._opt.num_gpus > 1:
            self.model = make_parallel(self.model, self.num_gpus)
    def run(self):
        res = self.model.evaluate_generator(self.test_loader, verbose=1)
        print("Test loss:{}, test accuracy {}".format(res[0], res[1]))
if __name__ == '__main__':
    tester = Tester()
    tester.run()