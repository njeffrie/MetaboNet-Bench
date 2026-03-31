from models.gluformer import Gluformer
from models.zoh import ZeroOrderHold
from models.linear import LinearRegression
from models.lstm import LSTM
from models.UniTS import UniTS
from models.gluforecast import Gluforecast
from models.lstm_trainable import TrainableLSTM
from models.units_trainable import TrainableUniTS

ABLATION_CKPT_DIR = 'studies/feature_ablation/checkpoints'

_ABLATION_MODELS = {
    'lstm-cgm':               ('lstm', 'cgm'),
    'lstm-cgm-insulin':       ('lstm', 'cgm_insulin'),
    'lstm-cgm-carbs':         ('lstm', 'cgm_carbs'),
    'lstm-cgm-insulin-carbs': ('lstm', 'cgm_insulin_carbs'),
    'units-cgm':               ('units', 'cgm'),
    'units-cgm-insulin':       ('units', 'cgm_insulin'),
    'units-cgm-carbs':         ('units', 'cgm_carbs'),
    'units-cgm-insulin-carbs': ('units', 'cgm_insulin_carbs'),
}


def get_model(name, device='cpu'):
    if name == 'gluformer':
        return Gluformer('njeffrie/Gluformer', device)
    elif name == 'gluformer-tiny':
        return Gluformer('njeffrie/Gluformer-tiny', device)
    elif name == 'zoh':
        return ZeroOrderHold()
    elif name == 'linear':
        return LinearRegression(15)
    elif name == 'lstm':
        return LSTM('njeffrie/LSTMGlucosePrediction', device)
    elif name == 'units':
        return UniTS('checkpoints/units.pth', device)
    elif name == 'gluforecast':
        return Gluforecast(model_path='checkpoints/gluforecast.pth', device=device)
    elif name in _ABLATION_MODELS:
        model_type, feature_set = _ABLATION_MODELS[name]
        ckpt = f'{ABLATION_CKPT_DIR}/{model_type}_{feature_set}.pth'
        if model_type == 'lstm':
            return TrainableLSTM(ckpt, feature_set=feature_set, device=device)
        else:
            return TrainableUniTS(ckpt, feature_set=feature_set, device=device)
    else:
        raise ValueError(f'Model {name} not found')
