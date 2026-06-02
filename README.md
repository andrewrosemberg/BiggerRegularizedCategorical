# Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners

https://arxiv.org/pdf/2505.23150

This branch contains the implementation of the BRC algorithm.

## Example usage

To run the BRC algorithm in a single task mode, just pass a single task name to the `env_names` variable:

`python3 train.py --env_names=dog-run`

By passing a list of task names, multi-task mode will be enabled. 

## mjlab Backend (Single-Object Cube)

The `mjlab` backend runs Shadow Hand cube manipulation via MuJoCo-mjlab instead of Gymnasium.
Currently supported for **cube only** with `--conditioning_mode=none` (online metrics only; no offline evaluation or rendering).

### Virtual environment

Use the dedicated venv `.venv_mjlab_check`, which contains:

- `jax[cuda12]`, `flax`, `optax`, `distrax`, `ml_collections`, `gymnasium`
- torch/warp (pre-installed for mjlab)

### CUDA runtime workaround

JAX (CUDA 12) and torch/warp (CUDA 13) ship conflicting NVRTC libraries.
Prepend the cu13 path so `libnvrtc-builtins.so.13.0` is found first:

```bash
export LD_LIBRARY_PATH=.venv_mjlab_check/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

### Example

```bash
LD_LIBRARY_PATH=.venv_mjlab_check/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
.venv_mjlab_check/bin/python train.py \
  --env_backend=mjlab --env_names=cube --conditioning_mode=none \
  --mjlab_num_envs=64 --max_steps=1000000 --start_training=1000 \
  --eval_interval=5000 --offline_evaluation=False --render=False \
  --log_to_wandb=False --seed=0
```

## Citation

If you find this repository useful, feel free to cite our paper using the following bibtex.

```
@article{nauman2025bigger,
  title={Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners},
  author={Nauman, Michal and Cygan, Marek and Sferrazza, Carmelo and Kumar, Aviral and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2505.23150},
  year={2025}
}
```
