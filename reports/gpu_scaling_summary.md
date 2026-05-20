# GPU Scaling Summary

Generated: 2026-05-20T10:17Z

Original Isaac Gym scaling is blocked on this Blackwell machine because the Python 3.8 torch build fails CUDA kernels on `sm_120`. The measured row below is a bounded Isaac Lab Cartpole runtime smoke, not SimToolReal training throughput.

| status | backend | gpu | num_envs | steps | exit | peak_vram_mib | avg_util | peak_util | fps | report |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| success_with_dependency_conflicts | Isaac Lab Cartpole | 0 | 12288 | 32 | 0 | 34858 | 35.08 | 39 | 798156.94 | `reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json` |
