import torch
import numpy as np
from stable_worldmodel.data.dataset import Dataset

class MockDataset(Dataset):
    def __init__(self, lengths, frameskip=1, num_steps=1, fixed_step_size=True):
        offsets = np.cumsum([0] + list(lengths[:-1]))
        super().__init__(np.array(lengths), offsets, frameskip, num_steps, fixed_step_size=fixed_step_size)
        self.call_count = 0
    
    @property
    def column_names(self):
        return ['data']
    
    def _load_slice(self, ep_idx, start, end) -> dict:
        self.call_count += 1
        # Simulate loading indices with frameskip
        indices = np.arange(start, end, self.frameskip)
        return {'data': torch.from_numpy(indices).float()}

def test_fixed_step_size_true():
    print("Testing fixed_step_size=True...")
    # Long episode
    # span = 5 * 10 = 50. length 100 >= 50.
    ds = MockDataset(lengths=[100], frameskip=10, num_steps=5, fixed_step_size=True)
    assert len(ds) > 0
    item = ds[0]
    assert item['data'].shape[0] == 5
    assert torch.equal(item['data'], torch.tensor([0, 10, 20, 30, 40]).float())

    # Short episode (length < span but length >= num_steps? No, clip_indices skips it)
    ds = MockDataset(lengths=[5], frameskip=10, num_steps=5, fixed_step_size=True)
    assert len(ds) == 0
    print("Fixed step size tests passed!")

def test_fixed_step_size_false():
    print("Testing fixed_step_size=False...")
    # Long episode, middle slice
    ds = MockDataset(lengths=[100], frameskip=10, num_steps=5, fixed_step_size=False)
    item = ds[0]
    assert item['valid_steps'] == 5
    assert item['padding_mask'].all()
    assert torch.equal(item['data'][:5], torch.tensor([0, 10, 20, 30, 40]).float())

    # Long episode, end slice (index 95 corresponds to start=95)
    # Dataset calculates clip_indices as (ep, start, length) for all start in range(length)
    ds.call_count = 0
    item = ds[95] 
    # start=95, end=100. end-start=5 <= fs=10. ep_end=100 > fs.
    # Shift back: start = 100 - 10 - 1 = 89.
    # _load_slice(0, 89, 100) -> indices 89, 99
    assert ds.call_count == 1
    assert item['valid_steps'] == 2
    assert item['data'][0] == 89
    assert item['data'][1] == 99
    assert item['padding_mask'][0] == True
    assert item['padding_mask'][1] == True
    assert item['padding_mask'][2] == False

    # Short episode (len < fs)
    ds = MockDataset(lengths=[5], frameskip=10, num_steps=5, fixed_step_size=False)
    # Indices 0 to 4 are available
    ds.call_count = 0
    item = ds[0] # start=0, ep_end=5. valid_steps=1. ep_end > 1.
    # This should trigger the fallback and call _load_slice a second time
    assert ds.call_count == 2
    assert item['valid_steps'] == 2
    assert item['data'][0] == 0
    assert item['data'][1] == 4
    
    item = ds[4] # start=4, ep_end=5. valid_steps=1.
    # start == ep_end - 1. first_step = _load_slice(0, 0, 1) -> [0]
    # cat([0], [4]) -> [0, 4]
    assert item['valid_steps'] == 2
    assert item['data'][0] == 0
    assert item['data'][1] == 4

    # Extremely short episode (len 1)
    ds = MockDataset(lengths=[1], frameskip=10, num_steps=5, fixed_step_size=False)
    item = ds[0]
    assert item['valid_steps'] == 2
    assert item['data'][0] == 0
    assert item['data'][1] == 0
    print("Variable step size tests passed!")

if __name__ == "__main__":
    test_fixed_step_size_true()
    test_fixed_step_size_false()
    print("All tests passed!")
