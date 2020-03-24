from .base_options import BaseOptions
class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--lr_policy', type=str, default = 'step', choices = ['step', 'plateau'])
        self._parser.add_argument('--lr_decay_epochs', type=int, default= 10, help="learning rate decays by 0.1 after every # epochs (lr_policy is 'step')")
        self._parser.add_argument('--nepochs', type=int, default= 50, help='# of epochs to train')
        self._parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate for optimizer')
        self._parser.add_argument('--adam_b1', type=float, default=0.5, help='beta1 for adam')
        self._parser.add_argument('--adam_b2', type=float, default=0.999, help='beta2 for adam')
        self._parser.add_argument('--optimizer', type=str, default='Adam', choices = ['Adam', 'SGD'])
        self._parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
        #self._parser.add_argument('--class_weights', type=float, nargs='+', default=[2.78, 3.64, 7.4, 8.3, 12.5, 100, 50, 100],
        self._parser.add_argument('--class_weights', type=float, nargs='+', default=[2.74398525, 3.56296941, 7.9199129, 8.44722581, 12.27334083, 215.34868421, 50.51388889, 230.51408451],
            help="class weights for unbalanced dataset.")
        self._parser.add_argument('--loss', type=str, default='categorical_crossentropy')
        self._parser.add_argument('--metrics', type=str, default=['acc'], nargs='+')
        self.is_train = True
