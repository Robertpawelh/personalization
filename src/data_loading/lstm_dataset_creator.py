import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.utils import get_spec_id


# NOTE: It is not an optimal solution for large datasets.

class LSTMDatasetCreator: # sliding window dataset creator
    def __init__(self, model_params, device):
        self.model_params = model_params
        self.scaler = MinMaxScaler()
        self.train_size = model_params.train_size
        self.device = device

    def _sliding_window(self, data):
        x = []
        y = []
        n_days_in = self.model_params.n_days_in
        n_days_out = self.model_params.n_days_out

        for i in range(len(data) - n_days_in - n_days_out + 1):
            _x = data[i:(i + n_days_in)]
            _y = np.stack(data[i + n_days_in:i + n_days_in + n_days_out], axis=1)[0]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    def generate_data(self, data):
        self.scaler = self.scaler.fit(data['price'].values.reshape(-1, 1))
        data = data.sort_values(by='scrap_time')
        aggregated_data = data.groupby(['specification', 'scrap_time']).agg(
            {
                'price': 'median',
            }).reset_index()


        grouped_data = aggregated_data.groupby('specification')#.interpolate(method='linear')
        x_all, y_all = [], []

        dataX = torch.Tensor()
        dataY = torch.Tensor()
        trainX = torch.Tensor()
        train_spec = torch.Tensor()
        trainY = torch.Tensor()
        testX = torch.Tensor()
        testY = torch.Tensor()
        test_spec = torch.Tensor()

        specs = data['specification'].unique()
        for name, group in grouped_data:
            name_id = get_spec_id(name, specs)
            group = self.scaler.transform(group['price'].values.reshape(-1, 1))
            x, y = self._sliding_window(group)
            x_all.append(x)
            y_all.append(y)

            split_id = int(len(y) * self.train_size)

            dataX = torch.cat([dataX, torch.Tensor(x)], dim=0)
            dataY = torch.cat([dataY, torch.Tensor(y)], dim=0)

            trainX = torch.cat([trainX, torch.Tensor(x[0:split_id])], dim=0)
            trainY = torch.cat([trainY, torch.Tensor(y[0:split_id])], dim=0)
            train_spec = torch.cat([train_spec, torch.Tensor([name_id] * split_id)], dim=0)
            testX = torch.cat([testX, torch.Tensor(x[split_id:len(x)])], dim=0)
            testY = torch.cat([testY, torch.Tensor(y[split_id:len(y)])], dim=0)
            test_spec = torch.cat([test_spec, torch.Tensor([name_id] * (len(x) - split_id))], dim=0)

        return (dataX.to(self.device), dataY.to(self.device)), \
            (trainX.to(self.device), trainY.to(self.device), train_spec.to(self.device)), \
            (testX.to(self.device), testY.to(self.device), test_spec.to(self.device)), \
            self.scaler