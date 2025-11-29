# UniClothDiff: Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation

## Abstract
Cloth manipulation is challenging due to its highly complex dynamics, near-infinite degrees of freedom, and frequent self-occlusions, which complicate both state estimation and dynamics modeling. Inspired by recent advances in generative models, we hypothesize that these expressive models can effectively capture intricate cloth configurations and deformation patterns from data. Therefore, we propose a diffusion-based generative approach for both perception and dynamics modeling. Specifically, we formulate state estimation as reconstructing full cloth states from partial observations and dynamics modeling as predicting future states given the current state and robot actions. Leveraging a transformer-based diffusion model, our method achieves accurate state reconstruction and reduces long-horizon dynamics prediction errors by an order of magnitude compared to prior approaches. We integrate our dynamics models with model predictive control and show that our framework enables effective cloth folding on real robotic systems, demonstrating the potential of generative models for deformable object manipulation under partial observability and complex dynamics.


## Installation
```bash
conda env create -f environment.yml
conda activate clothdiff

pip install "git+https://github.com/Jiayuan-Gu/torkit3d.git"

# Install the customized diffusers fork used by the repo
cd third_party/diffusers
pip install .

# Install UniClothDiff
pip install -e .

```

## Usage

### 1) Template mesh tokenization
Prepare and preprocess your template mesh before training:
```bash
python scripts/tmpl_patchify.py \
  --mesh_path PATH_TO_YOUR_TEMPALTE_MESH_FILE \
  --patchify_method ['voronoi', 'knn'] \
  --num_patches NUM_PATCHES \
  --output_prefix OUTPUT_PREFIX
```
For example:
```bash
python scripts/tmpl_patchify.py \
  --mesh_path assets/Tshirt.obj \
  --patchify_method voronoi \
  --num_patches 100 \
  --output_prefix tshirt
```


### 2) Data collection
Use your preferred soft-body simulator (e.g., [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html), [PyFlex](https://github.com/YunzhuLi/PyFleX), [Isaac Lab](https://github.com/isaac-sim/IsaacLab)) and export HDF5 in these schemas:
- **Cloth dynamics** (`uniclothdiff/datasets/cloth_dynamics.py`): `q_prev` (n_history_frames, n_v, 3), `q_next` (n_future_frames, n_v, 3), `action` (n_future_frames, n_points, 3) , `point_index` (int; which vertex being grasped).
- **Cloth state estimation** (`uniclothdiff/datasets/cloth_state_est.py`): `points` (N, 3) point cloud, `q` (V, 3) cloth vertices, plus a patchified template pickle via `template_mesh_path`.

The simulator used in this work is not yet public; follow the release timeline in the [ManiSkill roadmap](https://maniskill.readthedocs.io/en/latest/roadmap/index.html) if you are interested.


### 3) Training
Example command for training.
```bash
python scripts/train.py \
  --data_dir=path_to_data_dir \
  --per_gpu_batch_size=16 \
  --num_train_epochs=99999 \
  --checkpointing_steps=2000 \
  --checkpoints_total_limit=10 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=1000 \
  --lr_scheduler="cosine" \
  --seed=1 \
  --num_workers=16 \
  --mixed_precision="bf16" \
  --exp_name="debug" \
  --resume_from_checkpoint="latest" \
  --pretrained_model_name_or_path="None" \
  --config="configs/train_dynamics.yaml" \
```

## Citation
```
@article{tian2025uniclothdiff,
  author    = {Tian, Tongxuan and Li, Haoyang and Ai, Bo and Yuan, Xiaodi and Huang, Zhiao and Su, Hao},
  title     = {Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```


## License
MIT License.
