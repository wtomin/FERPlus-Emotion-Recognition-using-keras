from data.datasetfactory import DatasetFactory
from data.dataloader import DataLoader
from utils.data_utils import plotImages
from keras.applications.resnet50 import preprocess_input
import numpy as np
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",  type=int, default = 16, 
      help="number of layers for resnet model")
    parser.add_argument("--image_size", type=int, default=224, 
      help="input image size :(3, w, w)")
    parser.add_argument("--dataset_name", type=str, default='FERPlus', choices = ['FER', 'FERPlus' ])
    args = parser.parse_args()
    df = DatasetFactory()
    train_dataset = df.get_by_name(args, 'Train')
    print(train_dataset[0])
    train_loader = DataLoader(train_dataset,
                             batch_size = args.batch_size,
                             shuffle = True, 
                             drop_last = True, 
                             num_workers=8)
    for i, batch_data in enumerate(train_loader):
        print(batch_data[0].max())
        print(batch_data[0].min())
        print(batch_data[0].std())
        plotImages(batch_data[0])
        



