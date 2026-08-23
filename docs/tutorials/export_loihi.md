# Neuromorphic Export Tutorial (Intel Loihi 2 / NxSDK)

This tutorial shows how to export a trained Spiking Neural Network (SNN) kernel to Intel Loihi 2 using NxSDK.

## Prerequisites

- Python 3.14+ with `uv` package manager
- Intel NxSDK installed (requires Intel account access)
- Trained SNN model (STDP or backprop-trained)
- Intel Loihi 2 hardware (Kapoho Bay, Nahuku, or cloud access)
- `lava-nc` (Lava Neurocomputing) for simulation

## Quick Start

```bash
# Train and export an SNN to Loihi
uv run biopl-export-trained-kernel \
    --algorithm snn \
    --target cpu \
    --epochs 20 \
    --dataset mnist \
    --output ./snn_loihi \
    --format nxsdk
```

## Supported Algorithms for Loihi Export

| Algorithm | Kernel Backend | Loihi Support | Notes |
|-----------|----------------|---------------|-------|
| `snn` | `SNNKernelBackend` | ✅ Full | LIF + 3-factor STDP |
| `eqprop` | `EqPropKernelBackend` | ⚠️ Partial | Requires settling loops |
| `tile` (snn mode) | `TileKernelBackend` | ✅ Full | Tile substrate SNN |

---

## Step-by-Step Workflow

### 1. Train the SNN Model

```bash
# Train from scratch with STDP (bio-plausible)
uv run biopl-export-trained-kernel \
    --algorithm snn \
    --target cpu \
    --epochs 30 \
    --dataset mnist \
    --output ./snn_loihi \
    --format nxsdk

# Or train with surrogate gradient backprop, then export
uv run biopl-export-trained-kernel \
    --algorithm snn \
    --target cuda \
    --epochs 50 \
    --dataset mnist \
    --output ./snn_loihi \
    --format nxsdk \
    --surrogate-gradient
```

### 2. Export Configuration

```bash
# Full configuration
uv run biopl-export-trained-kernel \
    --algorithm snn \
    --target cpu \
    --epochs 30 \
    --precision fp16 \
    --format nxsdk \
    --loihi-gen loihi2 \
    --output ./snn_loihi2
```

Key parameters:
- `--format nxsdk`: Generates NxSDK-compatible Python scripts
- `--loihi-gen`: `loihi1` or `loihi2` (default: `loihi2`)
- `--precision`: Weights quantized to 8-bit on Loihi

### 3. Generated NxSDK Project Structure

```
snn_loihi/
├── network.py           # NxSDK network definition
├── weights.npz          # Trained weights (8-bit quantized)
├── run_on_loihi.py      # Execution script for Loihi hardware
├── run_sim.py           # Lava simulation script
├── manifest.json        # Network metadata
├── state_dict.pt        # PyTorch state dict
├── export_summary.json  # Export statistics
├── loihi_config.json    # Loihi-specific config
└── README.md            # Build/run instructions
```

---

## NxSDK Network Structure

The exported network follows this architecture:

```python
# network.py (auto-generated)
import nxsdk.api.n2a as nx
import numpy as np


def create_snn_network(weights_path: str, loihi_gen: str = "loihi2"):
    """Create SNN network for Loihi deployment."""

    net = nx.NxNet()

    # Load quantized weights
    weights = np.load(weights_path)
    W1 = weights["W1"]  # (hidden, input)
    W2 = weights["W2"]  # (output, hidden)

    # Input layer (virtual - spike generators)
    num_input = W1.shape[1]
    input_layer = net.createSpikeGenProcess(num_input)

    # Hidden LIF layer
    hidden_layer = net.createCompartmentGroup(
        size=W1.shape[0],
        compartmentParams=nx.CompartmentPrototype(
            vThMant=255,  # Threshold
            compartmentVoltageDecay=4096,  # Tau = 20ms @ 1kHz
            logicalCoreId=0,
        ),
    )

    # Output LIF layer
    output_layer = net.createCompartmentGroup(
        size=W2.shape[0],
        compartmentParams=nx.CompartmentPrototype(
            vThMant=255, compartmentVoltageDecay=4096, logicalCoreId=1
        ),
    )

    # Connections: Input -> Hidden (weights quantized to 8-bit)
    conn_ih = net.createConnection(
        input_layer,
        hidden_layer,
        prototype=nx.ConnectionPrototype(
            weight=W1.astype(np.int8), signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY
        ),
    )

    # Connections: Hidden -> Output
    conn_ho = net.createConnection(
        hidden_layer,
        output_layer,
        prototype=nx.ConnectionPrototype(
            weight=W2.astype(np.int8), signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY
        ),
    )

    # STDP learning rule (if enabled)
    if "stdp_params" in weights:
        stdp = nx.STDPRule(
            tauPlus=weights["stdp_tau_plus"],
            tauMinus=weights["stdp_tau_minus"],
            wMax=127,
            wMin=-128,
        )
        conn_ih.setSTDPRule(stdp)
        conn_ho.setSTDPRule(stdp)

    return net
```

---

## Running on Loihi Hardware

### 1. Prepare Input Data

```python
# run_on_loihi.py
import nxsdk.api.n2a as nx
import numpy as np
from network import create_snn_network


def encode_image_to_spikes(image, duration=100, rate_scale=100):
    """Rate coding: pixel intensity -> spike rate."""
    spikes = []
    for t in range(duration):
        # Poisson spike generation
        spike_mask = np.random.random(image.shape) < (image * rate_scale / duration)
        spikes.append(spike_mask.astype(np.int8))
    return np.array(spikes)  # (time, pixels)


# Load MNIST test data
from torchvision import datasets, transforms

test_data = datasets.MNIST(
    "./data", train=False, download=True, transform=transforms.ToTensor()
)
image, label = test_data[0]
spikes = encode_image_to_spikes(image.numpy().flatten())
```

### 2. Execute on Loihi

```python
# run_on_loihi.py (continued)
def run_on_loihi(network, input_spikes, num_steps=100):
    """Run network on Loihi hardware."""

    # Compile for Loihi
    compiler = nx.N2Compiler()
    board = compiler.compile(network, target="loihi2")

    # Configure input spikes
    input_proc = network.getProcesses()[0]  # Spike generator
    for t in range(num_steps):
        input_proc.setSpikes(input_spikes[t])

    # Run
    board.run(num_steps)

    # Read output
    output_layer = network.getCompartmentGroups()[-1]
    output_spikes = output_layer.getSpikes()

    board.disconnect()
    return output_spikes


# Main
if __name__ == "__main__":
    net = create_snn_network("./weights.npz")
    output_spikes = run_on_loihi(net, spikes)

    # Decode: count output spikes per neuron
    spike_counts = np.sum(output_spikes, axis=0)
    prediction = np.argmax(spike_counts)
    print(f"Prediction: {prediction}, True label: {label}")
```

---

## Simulation with Lava (No Hardware Required)

```python
# run_sim.py
import lava.lib.dl.slayer as slayer
import torch
from network import create_snn_network


# Convert to Lava SNN
def create_lava_network(weights_path):
    weights = np.load(weights_path)
    net = slayer.block.cuba.Dense(
        weights["W1"].shape[1],  # input
        weights["W1"].shape[0],  # hidden
        weights["W2"].shape[0],  # output
    )
    # Load weights
    net.synapse.weight.grad = torch.from_numpy(weights["W1"]).float()
    # ... load W2 ...
    return net


# Simulate
net = create_lava_network("./weights.npz")
input_tensor = torch.from_numpy(spikes).float()  # (time, batch, input)
output = net(input_tensor)
prediction = output.sum(0).argmax(1)
print(f"Simulation prediction: {prediction.item()}")
```

---

## Weight Quantization

Loihi uses 8-bit signed weights (-128 to 127). The export handles quantization automatically:

```python
# export_summary.json shows quantization stats
{
    "quantization": {
        "W1": {"min": -0.84, "max": 0.92, "scale": 0.007, "zero_point": 0},
        "W2": {"min": -1.23, "max": 1.15, "scale": 0.009, "zero_point": 0},
    },
    "accuracy": {"fp32": 0.942, "quantized": 0.931, "drop": 0.011},
}
```

**If accuracy drop > 2%:**
```bash
# Re-train with quantization-aware training
uv run biopl-export-trained-kernel \
    --algorithm snn \
    --target cuda \
    --epochs 50 \
    --quantize-aware \
    --output ./snn_loihi_qat
```

---

## Loihi 1 vs Loihi 2 Differences

| Feature | Loihi 1 (Nahuku) | Loihi 2 (Kapoho Bay) |
|---------|------------------|----------------------|
| Neurons/core | 1,024 | 1,024 (up to 4,096 with virtual) |
| Synapses/core | 128k | 256k |
| Weight precision | 8-bit | 8-bit + 24-bit |
| STDP | Basic | Programmable (3-factor) |
| Graded spikes | No | Yes |
| Algorithm | `--loihi-gen loihi1` | `--loihi-gen loihi2` |

---

## Advanced: Custom STDP Rules

```python
# In loihi_config.json (auto-generated)
{
    "stdp": {
        "enabled": true,
        "rule": "three_factor",
        "tau_plus": 20.0,
        "tau_minus": 20.0,
        "a_plus": 0.01,
        "a_minus": 0.012,
        "w_min": -128,
        "w_max": 127,
        "modulator": "reward",  # or "error", "novelty"
    }
}
```

For 3-factor STDP (reward-modulated):

```python
# three_factor_stdp.py (custom learning rule)
class ThreeFactorSTDP(nx.STDPRule):
    def __init__(self, modulator_signal):
        super().__init__(...)
        self.modulator = modulator_signal
    
    def update(self, pre_spike, post_spike, modulator):
        if modulator > 0:  # Reward
            return self.a_plus * np.exp(-dt / self.tau_plus)
        else:  # Punishment
            return -self.a_minus * np.exp(-dt / self.tau_minus)
```

---

## Troubleshooting

### "NxSDK not found"
```bash
# Install from Intel (requires account)
pip install nxsdk --index-url https://pypi.intel.com
# Or use Lava for simulation only
pip install lava-nc
```

### "Weight quantization accuracy drop > 5%"
- Enable quantization-aware training: `--quantize-aware`
- Increase weight bits in config (Loihi 2 supports 24-bit)
- Check weight distribution in `export_summary.json`

### "Network too large for single chip"
```bash
# Partition across chips
uv run biopl-export-trained-kernel \
    --algorithm snn \
    --partition-chips 4 \
    --output ./snn_loihi_multi
```

### "Spike encoding too slow"
- Use `nxsdk.utils.spike_gen` for hardware spike generation
- Pre-compute spike trains offline
- Use batch processing

### "Compilation fails on Loihi 2"
- Check `compartmentVoltageDecay` values (must be power of 2)
- Ensure `vThMant` <= 1023
- Verify core assignment doesn't exceed 128 cores/chip

---

## Complete Example: MNIST SNN on Loihi 2

```bash
# 1. Train STDP SNN (bio-plausible, no backprop)
uv run biopl-export-trained-kernel \
    --algorithm snn \
    --target cpu \
    --epochs 50 \
    --dataset mnist \
    --precision fp16 \
    --format nxsdk \
    --loihi-gen loihi2 \
    --output ./mnist_snn_loihi2

# 2. Check export summary
cat ./mnist_snn_loihi2/export_summary.json

# 3. Simulate with Lava (no hardware)
cd mnist_snn_loihi2
python run_sim.py

# 4. Run on Loihi 2 hardware (requires Intel account)
python run_on_loihi.py

# 5. Expected results:
#    - FP32 accuracy: ~94%
#    - Quantized accuracy: ~93%
#    - Latency: ~50ms per image @ 1kHz
#    - Energy: ~50µJ per inference
```

---

## References

- [Intel NxSDK Documentation](https://intel.github.io/nxsdk/)
- [Lava Neurocomputing Framework](https://github.com/lava-nc/lava)
- [Loihi 2 Data Sheet](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- [biopl-export-trained-kernel CLI](../api/acceleration.md#export-pipeline)
- [SNN Kernel Backend](../api/acceleration.md#spiking-kernels-snn_kernelspy)