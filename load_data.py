import torch
from utils import get_classes, get_object_detection_model
import json
import logging


def load_up():
    data = dict()

    with open(f'data/osetr/classes_idx.json', 'r') as file:
        classes_idx_osetr = json.load(file)
    classes_osetr = list(classes_idx_osetr.keys())
    model_osetr = get_object_detection_model(len(classes_osetr))
    model_osetr.load_state_dict(torch.load(f'logs/osetr/pytorch_model-e100.pt', weights_only=False))

    with open(f'data/karp/classes_idx.json', 'r') as file:
        classes_idx_karp = json.load(file)
    classes_karp = list(classes_idx_karp.keys())
    model_karp = get_object_detection_model(len(classes_karp))
    model_karp.load_state_dict(torch.load(f'logs/karp/pytorch_model-e3.pt', weights_only=False))

    data['osetr'] = dict()
    data['osetr']['classes'] = classes_osetr
    data['osetr']['model'] = model_osetr

    data['karp'] = dict()
    data['karp']['classes'] = classes_karp
    data['karp']['model'] = model_karp
    return data
