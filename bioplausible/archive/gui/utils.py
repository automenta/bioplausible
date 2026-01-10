
from pathlib import Path
import torch

def load_shakespeare():
    """Load Shakespeare dataset."""
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    path = Path('data/shakespeare.txt')
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        text = f.read()
    chars = sorted(set(text))
    c2i = {ch: i for i, ch in enumerate(chars)}
    i2c = {i: ch for ch, i in c2i.items()}
    data = torch.tensor([c2i[ch] for ch in text], dtype=torch.long)
    return data, c2i, i2c
