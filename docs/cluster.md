# Bocconi SLURM execution

The launcher restores settings from the repository's earlier working jobs:
account `3191856`, partition and QoS `stud`, Conda environment `teacherattr`,
and modules `miniconda3` and `cuda/12.4`. Each job requests one GPU, eight CPUs,
80 GB host RAM and 24 hours. These are historical settings, not a fresh check
of cluster policy. The available hardware is A100 80 GB; the historical generic
`--gres=gpu:1` is retained because no typed A100 resource name is documented.
No email notifications are enabled.

## Prepare once on the login node

Compute nodes have no internet. Use shared storage visible from both nodes.
The default cache is inside the repository, matching the old student setup
which did not assume access to `/scratch`. Override `CACHE_ROOT` or `HF_HOME`
consistently for both preparation and submission if using another shared disk.

```bash
cd /home/3191856/NLP_project/Teacher_attribution
source jobs/00_setup_env.sh
# Update the existing environment for this version of the project.
python -m pip install -e '.[models,data,research]'
python -m spacy download en_core_web_sm
# Authenticate interactively if necessary; never put tokens into job scripts.
hf auth login
python -m teacher_attr preflight
python -m teacher_attr pin --output configs/research.pinned.yaml
export CONFIG=configs/research.pinned.yaml
python scripts/cache_cluster_models.py --config "$CONFIG"
python -m teacher_attr --config "$CONFIG" prepare
# Confirm tokenizer access from the exact cache the jobs will use.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m teacher_attr --config "$CONFIG" preflight
```

The environment must contain a CUDA-enabled PyTorch build compatible with the
cluster driver. Loading CUDA 12.4 alone does not establish that compatibility.
The batch launcher checks CUDA and BF16 support before running the command.
The existing environment needs upgrading: old Transformers releases cannot load
all current teacher architectures. `uv.lock` is available for a locked environment
(`uv sync --all-extras --locked`); set `VENV_PATH="$PROJECT_ROOT/.venv"` to use it.
`VENV_PATH` must be an absolute Linux path. `CONDA_ENV_NAME`, `CUDA_MODULE`, and
`SKIP_CLUSTER_MODULES=1` are optional setup overrides.

Caching downloads complete pinned model snapshots, including weights, and can
consume substantial disk space. It covers the primary teachers, common student
base, attribution encoder and generic encoder. Dataset preparation must complete
online; jobs then read the saved prompt pools. Optional `diagnose --bertscore`
also requires its configured BERTScore model to be cached separately.
Create pinned configs on the cluster, not Windows, because they contain resolved
paths. Do not generate model outputs or train on the login node.

## Submit individual stages

```bash
export CONFIG=configs/research.pinned.yaml
bash scripts/submit_slurm.sh -- generate --role teachers --model qwen --split distill_train
bash scripts/submit_slurm.sh -- generate --role teachers --model qwen --split distill_val
```

Repeat for Ministral, Gemma and Granite. Submit only as many jobs as the `stud`
QoS permits; an old script specifically documented `QOSMaxSubmitJobPerUserLimit`.
Each command returns its job ID. Check that the job completed successfully before
running a dependent stage. Do not run two writers for the same model and split.

After all four teachers finish both pools, run QC (these commands do not load
GPU models and may run on the login node if local policy permits):

```bash
python -m teacher_attr --config "$CONFIG" qc --split distill_train
python -m teacher_attr --config "$CONFIG" qc --split distill_val
bash scripts/submit_slurm.sh -- distill --teacher qwen
# Repeat distill for ministral, gemma, granite after QC succeeds.
```

Follow `docs/implementation.md` for remaining commands; pass each GPU command
through the same submission helper. Prefer one teacher or student per generation
job using `--role`, `--model` and `--split`, rather than putting all generation
inside one 24-hour allocation. Student IDs are listed in the resolved config.
Four independent student jobs can use four GPUs concurrently if your QoS allows
it. Allocating four GPUs to one command does not enable distributed training.

Scheduler options go before `--`, application options after it:

```bash
bash scripts/submit_slurm.sh --time=08:00:00 --job-name=distill-qwen -- distill --teacher qwen
# Only if the QoS permits queued dependencies, substitute a real prerequisite ID:
bash scripts/submit_slurm.sh --dependency=afterok:12345 -- distill --teacher qwen
```

The helper creates `logs/` before SLURM opens the output files and sets the working
directory. Direct `sbatch jobs/run.sbatch ...` remains supported when submitted
from the repository root after `mkdir -p logs`. Logs include the GPU, PyTorch and
CUDA version. Compute jobs force Hugging Face offline mode and never install
packages. Email is disabled by default, even if the old account has mail settings.

## Limits

The historical 24-hour request is not a measured duration or verified current
queue limit. Run a small, separate pilot configuration first and inspect memory,
runtime and QC before using the full dataset. Generation resumes from its cache;
student training still has no optimizer/checkpoint resume. An interrupted student
run requires a fresh run directory. Do not automatically requeue training jobs.
No jobs have been submitted and no A100 benchmark has been performed locally.
