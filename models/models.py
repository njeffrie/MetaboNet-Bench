from models.gluformer import Gluformer
from models.zoh import ZeroOrderHold
from models.linear import LinearRegression
from models.lstm import LSTM
from models.UniTS import UniTS
from models.glucose_decoder import GlucoseDecoderModel

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
    elif name == 'glucose_decoder':
        return GlucoseDecoderModel(model_path='checkpoints/gluforecast.pth')
    else:
        raise ValueError(f'Model {name} not found')
