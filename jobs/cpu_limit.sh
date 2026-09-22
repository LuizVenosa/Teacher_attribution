#!/bin/bash
# Source only inside the batch job, after bashrc/Conda activation.
_teacher_cpu_soft="$(ulimit -St)"
_teacher_cpu_hard="$(ulimit -Ht)"
echo "CPU-time limit before setup: soft=$_teacher_cpu_soft hard=$_teacher_cpu_hard seconds"
if [[ "$_teacher_cpu_soft" != unlimited ]]; then
  if [[ "$_teacher_cpu_hard" != unlimited ]]; then
    echo "Finite CPU-time hard limit ($_teacher_cpu_hard seconds). Ask cluster support for an appropriate batch limit before running long generation jobs." >&2
    return 2
  fi
  ulimit -S -t unlimited || return 2
fi
[[ "$(ulimit -St)" == unlimited ]] || return 2
echo 'CPU-time soft limit: unlimited; SLURM allocation time limit remains in force.'
unset _teacher_cpu_soft _teacher_cpu_hard
