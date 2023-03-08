import torch
import torch.nn as nn
import pytorch_lightning as pl
import time
from utils.model_metrics import ModelMetrics
from typing import Tuple

# The idea behind the model - a combination of the LSTM model and personalization
# - specification passed as an additional parameter, having trainable weights
# atm it is only the concept, maybe i will expand it in the future 
class PersonalizedLSTM(pl.LightningModule):
    def __init__(
        self,
        model_params
    ):
        super(PersonalizedLSTM, self).__init__()
        self.num_layers = model_params.num_layers
        self.hidden_dim = model_params.hidden_dim

        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(p=model_params.dropout)
        self.output_layer = nn.Linear(2 * self.hidden_dim, model_params.n_days_out)

        self._learning_rate = model_params.learning_rate
        self._loss_fn = nn.MSELoss()

        self.spec_embedding = nn.Embedding(
            num_embeddings = model_params.n_specs + 1,
            embedding_dim = model_params.n_days_out,
            padding_idx=0,
        )
        self.spec_embedding.weight.data.uniform_(-0.001, 0.001)

        self.metrics = ModelMetrics(model_params.n_days_out, self.device)

    def forward(self, x, specs):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        x = self.output_layer(x)

        x = x + self.spec_embedding(specs.long() + 1)

        return x

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx
    ) -> torch.Tensor:
        logits, y_true, metrics = self._common_step(batch)

        loss = self._loss_fn(logits, y_true.float())

        self.metrics.log_metrics(self, loss, metrics, self.trainer.current_epoch, 'train')

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx
    ) -> torch.Tensor:
        logits, y_true, metrics = self._common_step(batch)

        loss = self._loss_fn(logits, y_true.float())

        self.metrics.log_metrics(self, loss, metrics, self.trainer.current_epoch, 'val')

        return {'val_loss': loss}

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx
    ) -> torch.Tensor:
        logits, y_true, metrics = self._common_step(batch)

        loss = self._loss_fn(logits, y_true.float())

        self.metrics.log_metrics(self, loss, metrics, self.trainer.current_epoch, 'test')

        return {'test_loss': loss}

    def _common_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        x, y_true, specs = batch[0], batch[1].long(), batch[2]
        start = time.time()
        preds = self(x, specs) # forward
        end = time.time()

        metrics = self.metrics.calculate(preds, y_true, end - start)

        return preds, y_true, metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self._learning_rate,
            weight_decay=5e-4
        )
