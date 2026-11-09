#!/usr/bin/env bash
# Probe a Princeton cluster login node for what HAZARD + ThreeDWorld need.
# Run on the login node of della or neuronic after ssh. Paste the output back.
# Read-only: prints facts, changes nothing.

echo "== host =="; hostname; cat /etc/redhat-release 2>/dev/null || cat /etc/os-release | head -2
echo "== user/quota =="; whoami; checkquota 2>/dev/null || quota -s 2>/dev/null || echo "no quota tool"
echo "== scratch candidates =="; ls -d /scratch/network/$USER /scratch/gpfs/$USER /n/fs/* 2>/dev/null | head
echo "== gpu on this node =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "no nvidia-smi on login node (expected)"
echo "== display tools (TDW recipe: Xvfb + vglrun -d egl) =="; for t in Xvfb vglrun glxinfo xpra; do printf "%-8s " $t; command -v $t || echo MISSING; done
echo "== slurm partitions =="; sinfo -o "%P %a %l %D %G" 2>/dev/null | head -20
echo "== gpu partition sample node =="; sinfo -N -p gpu -o "%N %G %c %m" 2>/dev/null | head -5
echo "== modules =="; module avail anaconda3 2>&1 | grep -i anaconda | head -5
echo "== internet from login node =="; curl -s -m 5 -o /dev/null -w "github %{http_code}\n" https://github.com; curl -s -m 5 -o /dev/null -w "s3 %{http_code}\n" https://tdw-public.s3.amazonaws.com/
echo "== done =="
echo "Next: run the same display-tools and internet lines inside 'salloc --gres=gpu:1 -t 0:10:00' to see what a compute node has."
