"""Top-level guard so one structure's failure can't kill a whole slice job.

The SBATCH wrapper loops over a slice, invoking a runner once per structure and
branching on its exit code:

    0  -> structure done OK, run the next one
    2  -> slice exhausted, stop the loop
    3  -> skip THIS structure, continue with the next one
    *  -> "unexpected" -> the wrapper does `exit $status`, which kills the
          SLURM job and wastes every remaining structure in the slice.

So a single unhandled Python exception (a VASP quirk, an MP data gap, a flaky
network call, a malformed CHGCAR ...) used to take down the entire run and
force a manual requeue. ``run_guarded`` closes that hole: it lets the
deliberate exit codes (0/2/3 via ``SystemExit``) through untouched, but turns
ANY other escaping exception into a clean per-structure skip (exit 3) after
printing the traceback, so the loop always advances.

Each runner marks its current row ``performed=True`` before the risky work, so a
skipped structure is not retried on the next iteration either.
"""
import signal
import sys
import traceback
from contextlib import contextmanager


class WatchdogTimeout(RuntimeError):
    """A guarded block exceeded its wall-clock budget."""


@contextmanager
def watchdog(seconds, label):
    """Raise ``WatchdogTimeout`` if the block runs longer than ``seconds``.

    Unbounded blocking calls wedge a whole slice job at 0% CPU until someone
    kills it: an MP API request has no request timeout, and an NFS write whose
    byte-range lock never comes back blocks below ``busy_timeout`` (so
    ``runs_db.run_with_retry`` never even sees it). Both have burned 24-core
    nodes for hours. SIGALRM turns the wedge into an ordinary exception, which
    ``run_guarded`` converts to a per-structure skip (exit 3) so the slice loop
    advances. Main thread only — which is where the runners call this.
    """
    def _fire(_signum, _frame):
        raise WatchdogTimeout(f"{label} exceeded {seconds}s")

    prev = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def run_guarded(main):
    """Run ``main()`` and guarantee the process never exits with an
    "unexpected" status that would abort the SBATCH slice loop.

    Returns by re-raising ``SystemExit`` (preserving codes 0/2/3). Any other
    exception is logged and converted to ``SystemExit(3)`` ("skip & continue").
    """
    try:
        main()
    except SystemExit:
        # Deliberate exit code from the runner (0 ok / 2 exhausted / 3 skip).
        raise
    except BaseException:
        traceback.print_exc()
        # Best-effort: mark any open W&B run as crashed so the dashboard isn't
        # left showing a phantom "running" job. Never let cleanup raise.
        try:
            import wandb
            if getattr(wandb, "run", None) is not None:
                wandb.finish(exit_code=1)
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        # 3 == "skip this structure, keep going" in the SBATCH wrapper.
        sys.exit(3)
