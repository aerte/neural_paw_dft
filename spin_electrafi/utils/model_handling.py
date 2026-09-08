import logging
import os
import torch
from torch import nn
from datetime import date
import dill as pickle

def get_tag(name) -> str:
    today = date.today()
    return f'ELECTRA_{today}_{name}'


class ModelIO:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.root_name = tag
        self._suffix = '.model_state_dict'
        self._iter_suffix = '.txt'

    def _get_model_path(self) -> str:
        return os.path.join(self.directory, self.root_name + self._suffix)

    def _get_info_path(self) -> str:
        return os.path.join(self.directory, self.root_name + self._iter_suffix)

    def save(self, module: nn.Module):
        # Save model
        model_path = self._get_model_path()
        print(f'Saving model: {model_path}')
        torch.save(obj=module.state_dict(), f=model_path)

    def load(self) -> nn.Module:
        # Load model
        model_path = self._get_model_path()
        logging.info(f'Loading model: {model_path}')
        model = torch.load(f=model_path)

        return model


def load_model(model_path) -> nn.Module:
    model = torch.load(model_path)
    return model
