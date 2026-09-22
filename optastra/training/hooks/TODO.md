# Hooks TODO

## Remove deprecated `metric=` / `mode=` shorthand (use `tracker=` only)

Hooks that care about "best" should take a `BestMetricTracker` only. The
`metric`/`mode` arguments exist so older call sites keep working. They build a
private tracker via `resolve_tracker()`, which defeats having one shared
definition of "best".

- [ ] `early_stopping.py` — `EarlyStoppingHook.__init__`: drop `metric`, `mode`; make `tracker` required.
- [ ] `checkpoint.py` — `BestCheckpointHook.__init__`: drop `metric`, `mode`; make `tracker` required.
- [ ] `best_metric.py` — delete `resolve_tracker()` once nothing calls it.
- [ ] `defaults.py` — `default_hooks(best_metric=, best_mode=)` builds a private tracker the same way.
      Replace with `best_tracker: BestMetricTracker | None = None`.

Call sites to migrate first:
- [ ] `astro_augment/*.py` — ~24 scripts call `EarlyStoppingHook(metric="val_total_loss", patience=10)`.
- [ ] `optastra/tests/training/test_hooks.py` — `EarlyStoppingHook(metric=...)`,
      `BestCheckpointHook(..., metric=..., mode=...)`, and
      `test_early_stopping_rejects_tracker_and_metric_together` (delete it along with the shorthand).

## Remove backwards-compat for old checkpoint formats

Checkpoints written before the `BestMetricTracker` refactor store `EarlyStoppingHook` state
as `{"best", "bad_evals"}` instead of `{"tracker": {"best"}, "bad_evals"}`.
Once no old runs need resuming:

- [ ] `early_stopping.py` — `load_state_dict`: drop the `state.get("tracker", {"best": state["best"]})` fallback.
- [ ] `checkpoint.py` — `BestCheckpointHook.load_state_dict`: drop the `if "tracker" in state` guard.
- [ ] `optastra/tests/training/test_hooks.py` — delete `test_early_stopping_loads_pre_tracker_checkpoint_state`.

## Cleanup made redundant by `Hook.priority`

- [ ] `defaults.py` — `default_hooks_list.insert(0, ResumeHook(...))` / `insert(1, BestCheckpointHook(...))`:
      position no longer matters (priority sorting handles it); just append.
