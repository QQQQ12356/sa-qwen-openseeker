"""Keep the historical Search-R1 path unchanged; explicitly select tool trajectories."""
from .searchr1_dataset import SearchR1SFTDataset


def require_data_mode(mode):
    if mode not in ('searchr1', 'openseeker'):
        raise ValueError(f'unsupported data_mode={mode!r}')


def make_dataset(data_path, tokenizer, data_mode='searchr1', max_length=16384, **kwargs):
    require_data_mode(data_mode)
    if data_mode == 'openseeker':
        from .openseeker import OpenSeekerDataset
        return OpenSeekerDataset(data_path, tokenizer, max_length=max_length, **kwargs)
    return SearchR1SFTDataset(data_path, tokenizer, **kwargs)
