"""Tasks from the shared `tasks` package (data directory: $YGGDRASIL_DATA, default <repo>/data)."""
import tasks


def build_task(cfg):
    """-> task with batch(rng, size), evaluate(fn, split), vocab, classes, kind ("text" or "state")."""
    return tasks.get(cfg.task, cfg.seed, cfg.seq_len, cfg.val_windows, cfg.test_tokens)
