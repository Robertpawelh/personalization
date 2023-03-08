from torchmetrics.regression import R2Score, MeanAbsoluteError, MeanSquaredError
import torch.nn as nn
import pandas as pd
import os

class ModelMetrics(nn.Module):
    def __init__(self, output_dim, device):
        super().__init__()

        metrics = {
            'mae': MeanAbsoluteError,
            # 'mse': MeanSquaredError,
            # 'R2': R2Score,
        }
        metrics = {name: metric().to(device) for name, metric in metrics.items()}
        self.metrics = nn.ModuleDict(metrics)

        self.global_metrics_filepath = 'global_metrics.csv'
        self.class_metrics_filepath = 'class.csv'

    @property
    def device(self):
        return next(self.children()).device

    def calculate(self, preds, y_true, time):
        results = {metric_name: metric(preds, y_true) for metric_name, metric in self.metrics.items()}
        results['prediction_time'] = time

        return results


    def log_metrics(self, logger, loss, avg_metrics, current_epoch, metric_prefix):
        logger.log("epoch", current_epoch, on_epoch=True, on_step=False)
        logger.log(f"{metric_prefix}/loss", loss.item(), on_epoch=True, on_step=False)
        for k, v in avg_metrics.items():
            logger.log(f"{metric_prefix}/{k}", v, on_epoch=True, on_step=False)
