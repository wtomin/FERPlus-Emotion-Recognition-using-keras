from options.train_options import TrainOptions
from PATH import PATH
PRESET_VARS = PATH()
from models.CNN import ModelFactory
from data.datasetfactory import DatasetFactory
from data.dataloader import DataLoader
from utils.regularizations import set_model_regularization
from utils.parallel import make_parallel
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, CSVLogger, ReduceLROnPlateau, EarlyStopping
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import multiprocessing
class Step_Decay:
    def __init__(self, init_lr, drop_ratio, n_epochs_to_drop):
        self.init_lr = init_lr
        self.drop_ratio = drop_ratio
        self.n_epochs_to_drop = n_epochs_to_drop

    def step_decay(self, epoch):
       lrate = self.init_lr * math.pow(self.drop_ratio,  
               math.floor((1+epoch)/self.n_epochs_to_drop))
       return lrate

class Trainer(object):
    def __init__(self):
        self._opt = TrainOptions().parse()
        df = DatasetFactory()
        train_dataset = df.get_by_name(self._opt, 'Train')
        self.train_loader = DataLoader(train_dataset,
                                 batch_size = self._opt.batch_size,
                                 shuffle = True, 
                                 drop_last = True, 
                                 num_workers=self._opt.n_threads_train)
        val_dataset = df.get_by_name(self._opt, 'Validation')
        self.val_loader = DataLoader(val_dataset,
                                 batch_size = self._opt.batch_size,
                                 shuffle = False, 
                                 drop_last = False, 
                                 num_workers=self._opt.n_threads_test)
        self.num_classes = train_dataset.n_classes
        self.model = ModelFactory().get_by_name(self._opt.model_name, self._opt, self.num_classes)

        if self._opt.reg_func is not None:
            self.model = set_model_regularization(self.model, self._opt.reg_func,
                                                  self._opt.reg_layers, self._opt.reg_bias)
        if self._opt.num_gpus > 1:
            self.model = make_parallel(self.model, self.num_gpus)
        if self._opt.optimizer == 'Adam':
            #self.optimizer = optimizers.Adam(learning_rate=self._opt.init_lr, #keras 2.3+
             self.optimizer = optimizers.Adam(lr=self._opt.init_lr, # keras 2.2.0
                 beta_1=self._opt.adam_b1, beta_2=self._opt.adam_b2)
        elif self._opt.optimizer == 'SGD':
            self.optimizer = optimizers.SGD(lr=self._opt.init_lr, 
                momentum=self._opt.momentum, nesterov=True)

        self.callback_list = list()
    def run(self):
        monitor_acc = 'val_acc'
        # Set Checkpoint to save the model with the highest accuracy
        checkpoint_acc = ModelCheckpoint(
            os.path.join(self._opt.checkpoints_dir, self._opt.name, 'model_max_acc.hdf5'),
            verbose=1,
            monitor=monitor_acc,
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )
        self.callback_list.append(checkpoint_acc)
        # Set Checkpoint to save the model with the minimum loss
        checkpoint_loss = ModelCheckpoint(
            os.path.join(self._opt.checkpoints_dir, self._opt.name, 'model_min_loss.hdf5'),
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min'
        )
        self.callback_list.append(checkpoint_loss)
        # Set Tensorboard Visualization
        tensorboard = TensorBoard(
            log_dir= os.path.join(self._opt.output_logs_dir, self._opt.name)
        )
        #tensorboard.set_model(self.model)
        self.callback_list.append(tensorboard)

        # set csv logger
        csvlogger = CSVLogger(
            os.path.join(self._opt.output_logs_dir, self._opt.name, 'training.csv'), 
            separator=',', append=False)
        self.callback_list.append(csvlogger)

        # learning rate scheduler
        if self._opt.lr_policy == 'step':
            sd = Step_Decay(self._opt.init_lr, 0.1, self._opt.lr_decay_epochs).step_decay
            lr_scheduler = LearningRateScheduler(sd, verbose=True)
        elif self._opt.lr_policy == 'plateau':
            lr_scheduler = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, patience=3, verbose=True)
        self.callback_list.append(lr_scheduler)

        # early stopper
        early_stopper = EarlyStopping(monitor = 'val_acc', min_delta=0, patience=8, verbose=1, mode='auto')
        self.callback_list.append(early_stopper)
        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self._opt.loss, 
            metrics=self._opt.metrics, 
        )

        if len(self._opt.class_weights)==1:
            self._opt.class_weights = self._opt.class_weights * self.num_classes
        assert len(self._opt.class_weights) == self.num_classes, "input class_weights should have {} elements".format(self.num_classes)
        self.class_weights = dict([(i, self._opt.class_weights[i]) for i in range(self.num_classes)])
        
        # debug
        # X, y = next(iter(self.train_loader))
        # y_pred = self.model.predict(X)
        # loss_ce_value = tf.keras.losses.categorical_crossentropy(y, y_pred)

        # Model training
        self.history = self.model.fit_generator(
            self.train_loader,
            verbose=1,
            steps_per_epoch=len(self.train_loader),
            epochs=self._opt.nepochs,
            callbacks=self.callback_list,
            validation_data=self.val_loader,
            validation_steps=len(self.val_loader),
            workers=self._opt.n_threads_train,
            class_weight=self.class_weights,
            use_multiprocessing=False,
            max_queue_size=16
        )
        # Save model at last epoch
        self.model.save(os.path.join(self._opt.checkpoints_dir, self._opt.name, 'final_model.hdf5'))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
