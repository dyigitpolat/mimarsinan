# Soft-core explosion for conv layers: source of inefficiency

## Summary

The large number of soft cores (e.g. 2048 in the first layer) comes from **Conv2DPerceptronMapper**: it creates **one neural core per spatial output position** (and per output-channel group when `max_neurons` is set). There is no spatial packing.

## Where it happens

- **File**: `src/mimarsinan/mapping/mapping_utils.py`
- **Class**: `Conv2DPerceptronMapper`
- **Method**: `_map_to_ir()`

Relevant comment and loop (around lines 1321–1374):

```python
# For conv, we create one core per spatial position, all sharing weights.
# Each core processes one patch (receptive field) and outputs out_channels values.
# Total cores = h_out * w_out, each with patch_size inputs and out_channels outputs.
...
for oh in range(h_out):
    for ow in range(w_out):
        ...
        for g_idx, g in enumerate(group_sizes):  # output-channel groups if max_neurons set
            core_outputs = ir_mapping.add_neural_core(...)
```

So:

- **Cores per conv layer** = `h_out * w_out * num_output_groups`
- **num_output_groups** = 1 if `max_neurons` is None, else `ceil(out_channels / max_neurons)` (from `_chunk_sizes`).

## Example: VGG16 first conv on CIFAR-10

- Input: (3, 32, 32), conv 3→64, 3×3, pad 1 → output (64, 32, 32).
- `h_out = w_out = 32` → **1024 positions**.
- If `max_neurons` is None: **1024 soft cores** (one per position).
- If `max_neurons = 32`: 2 output groups → **2048 soft cores** (1024 × 2).

So “2048 in the first layer” is consistent with either:

1. First conv with `max_neurons = 32` (1024 × 2 = 2048), or  
2. First two conv layers both 32×32 (1024 + 1024 = 2048).

## Why it’s inefficient

- Each core handles a **single** receptive field (one (oh, ow)) and outputs `out_channels` (or a chunk of them).
- There is **no spatial packing**: multiple positions are not merged into one core even when the hardware has enough axons/neurons (e.g. 50k×50k).
- So core count grows as **O(h_out × w_out)** per conv layer, which blows up for moderate resolutions and leads to “No more hard cores available” when packing into a limited number of hard cores.

## Possible improvements

1. **Use IRMapping’s max_neurons in Conv2D mappers**  
   Torch-converted mappers are created without `max_neurons`; only `IRMapping` has it. Passing the same `max_neurons` into the mapper graph (or into each Conv2D mapper before `map_to_ir`) would at least make output-channel tiling consistent and predictable.

2. **Spatial packing**  
   In `Conv2DPerceptronMapper._map_to_ir`, allow one neural core to serve **multiple** spatial positions when the core’s axon/neuron limits allow it (e.g. one core = multiple (oh, ow) patches), so that the number of soft cores grows slower than `h_out * w_out`.
