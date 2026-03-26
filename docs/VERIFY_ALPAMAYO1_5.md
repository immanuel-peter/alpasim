# Verification Steps for Alpamayo 1.5 Driver Integration

Run these steps on a Linux machine with a CUDA-capable GPU and a valid HuggingFace token
that has access to `nvidia/Alpamayo-1.5-10B`.

## Prerequisites

- NVIDIA GPU with ≥24 GB VRAM (H100 80 GB recommended)
- `HF_TOKEN` environment variable set
- Alpasim repo checked out and environment set up (`source setup_local_env.sh`)

## 1. Install updated dependencies

```bash
cd src/driver
uv sync
```

Expected: resolves and installs `alpamayo1_5` from
`https://github.com/NVlabs/alpamayo1.5.git` at rev `2eff703`.

## 2. Verify entry point registration

```bash
uv run python -c "
from alpasim_plugins.plugins import PluginRegistry
r = PluginRegistry('alpasim.models')
available = r.get_available()
print('Registered models:', available)
assert 'alpamayo1_5' in available, 'alpamayo1_5 entry point not found!'
print('OK')
"
```

Expected output includes `alpamayo1_5` alongside `ar1`, `vam`, `manual`.

## 3. Run existing driver unit tests

```bash
uv run pytest src/driver/tests -v
```

Expected: all existing tests pass (no regressions).

## 4. Model load smoke test

Requires GPU and HF access. Downloads ~20 GB of weights on first run.

```bash
uv run python -c "
import torch
from alpasim_driver.models.alpamayo1_5_model import Alpamayo1_5Model

camera_ids = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov',
    'camera_cross_right_120fov',
    'camera_front_tele_30fov',
]
model = Alpamayo1_5Model(
    checkpoint_path='nvidia/Alpamayo-1.5-10B',
    device=torch.device('cuda'),
    camera_ids=camera_ids,
    context_length=4,
)
assert model._pred_num_waypoints == 64, f'Expected 64 waypoints, got {model._pred_num_waypoints}'
assert model.output_frequency_hz == 10
assert list(model._camera_indices.numpy()) == [0, 1, 2, 6]  # sorted by CAMERA_NAME_TO_INDEX
print(f'OK: pred_num_waypoints={model._pred_num_waypoints}, camera_indices={model._camera_indices.tolist()}')
"
```

## 5. Full wizard run

From the repo root:

```bash
uv run alpasim_wizard \
    +deploy=local \
    wizard.log_dir=./logs/alpamayo1_5_test \
    driver=[alpamayo1_5,alpamayo1_5_runtime_configs]
```

Check the run log for:
- No `ValueError` on model init
- `"Initialized Alpamayo 1.5 with 4 cameras, context_length=4"` in driver logs
- `"Alpamayo 1.5 Chain-of-Causation:"` lines appearing once inference starts
- Ego vehicle moving (non-zero trajectory waypoints returned)
