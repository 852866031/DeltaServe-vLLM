"""[fill_graph_padding] FT samples that fit the rows of the graph size a step
already runs at (CPU; no engine)."""
import sys
import types

from deltaserve.config import FinetuneConfig
from deltaserve.config_loader import FINETUNE_SECTIONS
from deltaserve.estimator import StepFeatures
from deltaserve.ft_scheduler import FinetuneScheduler

_n = [0, 0]


def check(name, ok, detail=""):
    _n[0 if ok else 1] += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")


room = FinetuneScheduler._graph_padding_room
SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)


def sched(sizes=SIZES, ft_only=False):
    return types.SimpleNamespace(_ft_graph_sizes=tuple(sizes), _graph_ft_only=ft_only)


def feats(t_in=0, b_d=0):
    return StepFeatures(t_in=t_in, t_ft=0, b_d=b_d)


print("A. the room left in the graph")
check("300 inference tokens + 40 FT run the 384 graph: 44 rows left",
      room(sched(), feats(t_in=300), 40) == 44)
check("decode requests count one token each",
      room(sched(), feats(t_in=100, b_d=20), 30) == 10)
check("exactly on a captured size: nothing left",
      room(sched(), feats(t_in=200), 56) == 0)
check("above the largest has_ft graph: no graph, no room",
      room(sched(), feats(t_in=500), 40) == 0)
check("an FT-only step runs eager ...", room(sched(), feats(), 100) == 0)
check("... unless FT-only steps replay graphs too",
      room(sched(ft_only=True), feats(), 100) == 28)
check("key off (no sizes): never", room(sched(sizes=()), feats(t_in=300), 40) == 0)

print("B. config")
cfg = FinetuneConfig()
check("off by default, in the finetune_features section",
      cfg.fill_graph_padding is False
      and "fill_graph_padding" in FINETUNE_SECTIONS["finetune_features"])

print(f"\n{_n[0]} passed, {_n[1]} failed")
sys.exit(1 if _n[1] else 0)
