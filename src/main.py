from pytorch_lightning import Trainer
from utils.model_parameters import ModelParameters
from data_loading.iphone_datamodule import IphoneDataModule
from models.personalized_lstm.personalized_lstm import PersonalizedLSTM


def run():
    model_params = ModelParameters(
        n_days_in=7,
        hidden_dim=256,
        num_layers=2,
        n_days_out=30,
        n_specs=10
    )

    dmodule = IphoneDataModule(model_params=model_params, data_path='data/olx.csv', device='cpu')
    model = PersonalizedLSTM(model_params)

    trainer = Trainer(
        max_epochs=5,
        accelerator='cpu')

    trainer.fit(model, datamodule=dmodule)

if __name__ == '__main__':
    run()
