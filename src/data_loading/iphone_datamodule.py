import pandas as pd
from data_loading.lstm_dataset_creator import LSTMDatasetCreator
from utils.model_parameters import ModelParameters
from data_loading.iphone_dataset import IphoneDataset
from torch.utils.data import DataLoader
from data_loading.preprocessing import preprocessing
import pytorch_lightning as pl

class IphoneDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_params: ModelParameters,
            data_path: str,
            device,
            batch_size: int = 128,
            out_dict: bool = False

    ):
        super().__init__()

        # Defining batch size of our data
        self.data_path = data_path
        self.batch_size = batch_size

        self.out_dict = out_dict
        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.device = device
        self.scaler = None
        self.model_params = model_params

    def setup(self, stage=None):
        dset = pd.read_csv(self.data_path)
        dset = preprocessing(dset, self.model_params)

        dset_creator = LSTMDatasetCreator(self.model_params, self.device)
        (self.dataX, self.dataY), (self.trainX, self.trainY, self.train_spec), (self.testX, self.testY, self.test_spec), self.scaler = dset_creator.generate_data(dset)
        self.train_data = IphoneDataset(self.trainX, self.trainY, self.train_spec)
        self.valid_data = IphoneDataset(self.testX, self.testY, self.test_spec)
        print(self.dataX.shape, self.dataY.shape)

    def train_dataloader(self):
        # Generating train_dataloader
        return DataLoader(self.train_data,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        # Generating val_dataloader
        return DataLoader(self.valid_data,
                          batch_size=self.batch_size)
