# quality-and-format SYNC_MANIFEST

- synced_at: 2026-11-06T14:43:20+08:00 (Asia/Shanghai)
- source: myserver:/cluster/home/zengy/quality-constrained-kv-page-mapping
- method: remote tar /tmp/kvpm_qa_sync.tgz -> extract (CR-sanitized) -> copy into quality-and-format only
- tarball_bytes: 7913617
- destination: F:\quality-constrained-kv-page-mapping\quality-and-format
- root_untouched_check: before=['Quality-Constrained Joint Mapping of Low-Bit KV Cache Pages Research Scope and Topic Boundaries.md', 'experiments', 'joint-mapping', 'quality-and-format', 'read-datapath', 'runs', 'server', 'write-datapath']; after=['Quality-Constrained Joint Mapping of Low-Bit KV Cache Pages Research Scope and Topic Boundaries.md', 'experiments', 'joint-mapping', 'quality-and-format', 'read-datapath', 'runs', 'server', 'write-datapath']; changed=[]

## Protected local protocol files (NOT overwritten)
- README.md: exists=True mtime=2026-11-05T17:21:59+08:00 sha256=0954998830f3e677d19c0d13ae67d046cd724ff953fab063032f4ecbaad5d604
- 同页配对实验实现清单.md: exists=True mtime=2026-11-05T17:25:15+08:00 sha256=1116c5fdf1a815d9a65f18fc35b5c930799ca6595f42f7910d69c0e2f8732829
- 敏感度分布与混合配置空间.md: exists=True mtime=2026-11-05T16:20:40+08:00 sha256=2cd91c674e62295f72dbca9525ab0b399a17de3542f521d300860b0b7c0eff86
- 试跑冻结与同页配对实验协议.md: exists=True mtime=2026-11-05T17:25:31+08:00 sha256=8f0cfcaf83833d3a6776ce7dd13a6d4c3ab4f1f3e116c964f281836240c45cbf

## Helpers copied to quality-and-format root
- check_nonretr_gates.py
- extract_nonretr_report.py
- launch_smoke.sh
- patch_pmp_outdefault.py
- poll_nonretr.sh
- README_server.md
- run_protect_multipage.sh
- run_protect_multipage_full.sh
- run_protect_multipage_nonretr_full.sh
- run_protect_multipage_nonretr_smoke.sh
- verify_nonretr_align.py

## Server run dirs expected
- [OK] runs/pilot_32
- [OK] runs/pilot_32_validate
- [OK] runs/pilot_b2_v1
- [OK] runs/pilot_multipage_b2_v1
- [OK] runs/pilot_multipage_v1
- [OK] runs/pilot_strong_pair_v1
- [OK] runs/protect_close_feat_v1
- [OK] runs/protect_close_feat_v1_smoke
- [OK] runs/protect_multipage_nonretr_v1
- [OK] runs/protect_multipage_nonretr_v1_smoke
- [OK] runs/protect_multipage_v1
- [OK] runs/protect_multipage_v1_smoke
- [OK] runs/protect_strategy_v1
- [OK] runs/rep_sample_b2_v1
- [OK] runs/rep_sample_nonretr_v1
- [OK] runs/rep_sample_v1
- [OK] runs/smoke_1x1

## Local run dirs: 17
- runs/pilot_32
- runs/pilot_32_validate
- runs/pilot_b2_v1
- runs/pilot_multipage_b2_v1
- runs/pilot_multipage_v1
- runs/pilot_strong_pair_v1
- runs/protect_close_feat_v1
- runs/protect_close_feat_v1_smoke
- runs/protect_multipage_nonretr_v1
- runs/protect_multipage_nonretr_v1_smoke
- runs/protect_multipage_v1
- runs/protect_multipage_v1_smoke
- runs/protect_strategy_v1
- runs/rep_sample_b2_v1
- runs/rep_sample_nonretr_v1
- runs/rep_sample_v1
- runs/smoke_1x1

## __pycache__ trees removed under DEST: 0
## files_copied_count: 16436

## Notes
- Prior scp -r failed on runs/protect_close_feat_v1_smoke due to trailing CR in remote dirname; tar extract sanitized CR.
- Parent handles b=2 GPU jobs; this sync does not start jobs.
- Future stage completions should re-sync into quality-and-format only; leave F:\quality-constrained-kv-page-mapping root siblings as-is.
