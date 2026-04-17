import torch
import numpy as np

from models.UniTS import Model


FEATURE_COLUMNS = {
    'cgm': ['CGM'],
    'cgm_insulin': ['CGM', 'Insulin'],
    'cgm_carbs': ['CGM', 'Carbs'],
    'cgm_insulin_carbs': ['CGM', 'Insulin', 'Carbs'],
}


class _DefaultUniTSHParams:
    d_model = 128
    n_heads = 8
    e_layers = 2
    patch_len = 16
    prompt_num = 10
    dropout = 0.1


def _default_units_hparams():
    return _DefaultUniTSHParams()


def _units_hparams_from_ckpt(d: dict):
    class H:
        pass

    o = H()
    for k, v in d.items():
        setattr(o, k, v)
    if not hasattr(o, 'patch_len'):
        o.patch_len = 16
    if not hasattr(o, 'stride'):
        o.stride = o.patch_len
    return o


def _args_from_units_hparams(hp):
    """Build a namespace compatible with UniTS Model(args, ...)."""
    class Args:
        pass

    a = Args()
    a.d_model = hp.d_model
    a.n_heads = hp.n_heads
    a.e_layers = hp.e_layers
    a.patch_len = hp.patch_len
    a.stride = hp.patch_len
    a.prompt_num = hp.prompt_num
    a.dropout = hp.dropout
    return a


def build_units_model(units_hparams, seq_len=180, pred_len=12):
    """Construct a UniTS Model for long-term forecasting.

    units_hparams: UniTSHParams-like object with d_model, n_heads, e_layers,
    patch_len, prompt_num, dropout (stride follows patch_len).
    """
    args = _args_from_units_hparams(units_hparams)
    configs_list = [
        'CGM',
        {'task_name': 'long_term_forecast', 'seq_len': seq_len, 'pred_len': pred_len},
    ]
    return Model(configs_list=[configs_list], args=args)


class TrainableUniTS:
    """Benchmark-compatible wrapper for a trained-from-scratch UniTS."""

    def __init__(self, checkpoint_path, feature_set='cgm', device='cpu'):
        self.feature_set = feature_set
        self.feature_cols = FEATURE_COLUMNS[feature_set]
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        udict = ckpt.get('units_hparams')
        if udict is not None:
            units_hp = _units_hparams_from_ckpt(udict)
        else:
            units_hp = _default_units_hparams()

        self.model = build_units_model(
            units_hp,
            seq_len=ckpt.get('seq_len', 180),
            pred_len=ckpt.get('pred_len', 12),
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict(self, timestamps, cgm, insulin, carbs):
        batch_size = cgm.shape[0]
        channels = {'CGM': cgm, 'Insulin': insulin, 'Carbs': carbs}

        selected = [channels[c] for c in self.feature_cols]
        x = np.stack(selected, axis=-1)
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.model(
                x_enc=x_t, x_mark_enc=None,
                task_id=0, task_name='long_term_forecast',
            )
        out = out[:, :, 0] if out.ndim == 3 else out
        return out.detach().cpu().numpy().reshape(batch_size, -1)
