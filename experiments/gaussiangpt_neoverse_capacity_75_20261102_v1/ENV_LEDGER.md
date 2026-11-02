### env: gaussiangpt-capacity-ssh75@2660cdc8
spec_sha256: 2660cdc8c6f92b006535b208d5573085bfde6b62c656f41d56cb4a6066a8a1a2
how: Isolated environment copied from 134, plus explicit private glibc 2.35 ELF loader at /data/CJH/1/bin/python.
root: /data/CJH/1; env /data/CJH/1/envs/gaussiangpt; caches /data/CJH/1/cache; temporary /data/CJH/1/tmp.
tier: one RTX4090 24GiB per worker, 6 CPU threads, allocator budget20GiB. Four cards idle at allocation.
weights: exact original codec and round5 decoder under /data/CJH/1/assets, asset manifest3702a70b2f13dc9f214700f5f6601d873829bbc6ba4ee289933200c87fc14b5e.
validated: tier1/2 native rendered two-view forward/backward passed in13.65s; independent agent-follows-doc pending.
gotcha: raw copied MinkowskiEngine requires GLIBC_2.34 but host provides2.31. Private process loader resolves it; host system and NVIDIA driver unchanged. All spawned Python processes must use wrapper, not sys.executable.
limits: no action-training quality claim; no new system package or driver installation.
