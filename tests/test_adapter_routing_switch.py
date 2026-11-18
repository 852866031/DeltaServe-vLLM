#!/usr/bin/env python
"""Adapter-routing on/off switch (deltaserve.config.AdapterRoutingConfig).

CPU-only: builds configs, never loads a model or touches a GPU.

    python tests/test_adapter_routing_switch.py

Checks
  * AdapterRoutingConfig defaults to off and rejects bad values / unknown keys;
  * a YAML without a ``adapter_routing`` section produces exactly the additional_config
    it produced before the feature existed; with one, the section rides along
    as its own entry and relative paths are resolved;
  * VllmConfig.adapter_routing_config reads it back, routing does not enter the
    compile-cache hash, and routing-only (finetuning off) changes only the
    worker class (``deltaserve.route_worker.RouteWorker``: the stock worker +
    the route encoder) — the scheduler class is untouched;
  * with routing off nothing under ``deltaserve.adapter_routing`` is imported, and no
    module outside that package imports it at module level.
"""

import ast
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_PKG = _REPO / "deltaserve-plugin" / "deltaserve"
_CONFIG = _REPO / "configs" / "serving_config_finetuning_qwen3_0.6b.yaml"

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def green(msg):
    print(f"{_GREEN}{msg}{_RESET}")


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {_GREEN}[PASS]{_RESET} {name}")
    else:
        _failed += 1
        print(f"  {_RED}[FAIL]{_RESET} {name}")


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def routing_modules():
    return sorted(m for m in sys.modules if m.startswith("deltaserve.adapter_routing"))


def test_config_defaults_and_validation():
    green("AdapterRoutingConfig:")
    from deltaserve.config import AdapterRoutingConfig

    rc = AdapterRoutingConfig()
    check("off by default", rc.enable_adapter_routing is False)
    check("default route name is 'auto'", rc.route_model_name == "auto")
    check("default pool capacity is 32", rc.max_active_adapters == 32)
    check("unknown key rejected", raises(lambda: AdapterRoutingConfig(no_such_knob=1)))
    check("empty route name rejected",
          raises(lambda: AdapterRoutingConfig(route_model_name="")))
    check("pool capacity 0 rejected",
          raises(lambda: AdapterRoutingConfig(max_active_adapters=0)))
    check("threshold outside [-1, 1] rejected",
          raises(lambda: AdapterRoutingConfig(fallback_score_threshold=1.5)))


def test_loader():
    green("config_loader routing section:")
    import yaml

    from deltaserve.config_loader import (
        ADAPTER_ROUTING_KEY,
        additional_config,
        finetune_kwargs,
        load_yaml_config,
        adapter_routing_config,
        split_config,
    )

    raw = load_yaml_config(_CONFIG, verbose=False)
    check("preset has no routing section", "adapter_routing" not in raw)
    ac = additional_config(raw)
    check("no routing section → additional_config is the finetune entry only",
          ac == {"deltaserve": {k: v for k, v in finetune_kwargs(raw).items()
                                if v is not None}})
    check("no routing section → AdapterRoutingConfig is off",
          adapter_routing_config(raw).enable_adapter_routing is False)

    body = yaml.safe_load(_CONFIG.read_text())
    body["adapter_routing"] = {"enable_adapter_routing": True, "route_model_name": "pick",
                       "index_path": "idx/pool"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        p.write_text(yaml.safe_dump(body))
        cfg = load_yaml_config(p, verbose=False)
        ac2 = additional_config(cfg)
        check("routing section rides along as its own entry",
              ac2.get(ADAPTER_ROUTING_KEY, {}).get("enable_adapter_routing") is True)
        check("finetune entry unchanged by the routing section",
              set(ac2["deltaserve"]) == set(ac["deltaserve"]))
        check("relative index_path resolved against the YAML",
              cfg["adapter_routing"]["index_path"] == str((Path(d) / "idx/pool").resolve()))
        rc = adapter_routing_config(cfg)
        check("AdapterRoutingConfig built from the section",
              rc.enable_adapter_routing and rc.route_model_name == "pick")
        engine_kwargs, _, _ = split_config(cfg)
        check("routing keys do not leak into EngineArgs kwargs",
              not {"enable_adapter_routing", "route_model_name"} & set(engine_kwargs))

        body["adapter_routing"]["bogus"] = 1
        p.write_text(yaml.safe_dump(body))
        bad = load_yaml_config(p, verbose=False)
        check("unknown routing key rejected by split_config",
              raises(lambda: split_config(bad)))


def test_plugin_wiring():
    green("plugin: VllmConfig.adapter_routing_config + compile-cache hash:")
    from deltaserve.plugin import DeltaServeConfig, register

    register()
    from vllm.config import VllmConfig

    ft = {"enable_finetuning": False}
    plain = VllmConfig()
    check("no additional_config → routing off",
          plain.adapter_routing_config.enable_adapter_routing is False)

    only_ft = VllmConfig(additional_config={"deltaserve": dict(ft)})
    ac = only_ft.additional_config
    check("finetune-only launch: no routing entry in the mapping",
          isinstance(ac, DeltaServeConfig)
          and DeltaServeConfig.ADAPTER_ROUTING_KEY not in ac and len(ac) == 1)
    check("finetune-only launch: routing off",
          only_ft.adapter_routing_config.enable_adapter_routing is False)

    both = VllmConfig(additional_config={
        "deltaserve": dict(ft),
        "deltaserve_adapter_routing": {"enable_adapter_routing": True}})
    check("routing entry → routing on", both.adapter_routing_config.enable_adapter_routing is True)
    check("routing entry visible in the mapping",
          both.additional_config[DeltaServeConfig.ADAPTER_ROUTING_KEY]["enable_adapter_routing"]
          is True and len(both.additional_config) == 2)
    check("routing is not part of the compile-cache hash",
          both.additional_config.compute_hash() == ac.compute_hash())

    only_routing = VllmConfig(additional_config={
        "deltaserve_adapter_routing": {"enable_adapter_routing": True}})
    check("routing without finetuning: routing on, finetuning off",
          only_routing.adapter_routing_config.enable_adapter_routing
          and not only_routing.finetune_config.enable_finetuning)
    check("routing without finetuning: the worker is the stock one + the "
          "route encoder (RouteWorker)",
          only_routing.parallel_config.worker_cls
          == "deltaserve.route_worker.RouteWorker"
          and plain.parallel_config.worker_cls
          != "deltaserve.route_worker.RouteWorker")
    check("routing without finetuning: scheduler class untouched",
          only_routing.scheduler_config.scheduler_cls
          == plain.scheduler_config.scheduler_cls)

    clone = pickle.loads(pickle.dumps(both.additional_config))
    check("config survives pickling (spawned processes)",
          clone.adapter_routing.enable_adapter_routing is True
          and clone.compute_hash() == both.additional_config.compute_hash())
    check("bad routing entry rejected at config time",
          raises(lambda: VllmConfig(additional_config={
              "deltaserve_adapter_routing": {"enable_adapter_routing": True, "bogus": 1}})))


def test_import_boundary():
    green("routing off → deltaserve.adapter_routing never loaded:")
    # Checked in a fresh process: this one has loaded routing-ON configs above,
    # which legitimately import the package (the job validation registers the
    # stages the feature owns).
    code = (
        "import sys\n"
        "from deltaserve.plugin import register\n"
        "register()\n"
        "from vllm.config import VllmConfig\n"
        "from deltaserve.config_loader import (load_yaml_config,\n"
        "    additional_config, split_config)\n"
        f"cfg = load_yaml_config({str(_CONFIG)!r}, verbose=False)\n"
        "split_config(cfg)   # launch-time validation of the job\n"
        "vc = VllmConfig(additional_config=additional_config(cfg))\n"
        "from deltaserve.ft_job import FinetuneJobSpec, FtJobManager\n"
        "mgr = FtJobManager(vc.adapter_routing_config.enable_adapter_routing)\n"
        "mgr.submit(FinetuneJobSpec.from_finetune_config(vc.finetune_config))\n"
        "assert vc.finetune_config.enable_finetuning\n"
        "assert not vc.adapter_routing_config.enable_adapter_routing\n"
        "import deltaserve.api, deltaserve.engine_config  # API-server side\n"
        "from types import SimpleNamespace as NS\n"
        "added = []\n"
        "app = NS(state=NS(args=NS(additional_config=additional_config(cfg),\n"
        "         model='m', served_model_name=None)),\n"
        "         add_middleware=lambda c, **k: added.append(k),\n"
        "         include_router=lambda r: None)\n"
        "deltaserve.api.attach_router(app)   # installs the routing-off answer\n"
        "assert len(added) == 1 and added[0]['route_model_name'] == 'auto'\n"
        "bad = sorted(m for m in sys.modules if m.startswith('deltaserve.adapter_routing'))\n"
        "print('LOADED', bad)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=tempfile.gettempdir())
    check("fresh process, finetuning on + routing off: nothing loaded "
          f"(rc={out.returncode})",
          out.returncode == 0 and "LOADED []" in out.stdout)
    if out.returncode != 0:
        print(out.stderr[-2000:])

    offenders = []
    for path in sorted(_PKG.rglob("*.py")):
        if (_PKG / "adapter_routing") in path.parents:
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:   # module level only; guarded lazy imports are fine
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "deltaserve.adapter_routing" or n.startswith("deltaserve.adapter_routing.")
                   for n in names):
                offenders.append(f"{path.relative_to(_PKG)}:{node.lineno}")
    check(f"no module-level import of deltaserve.adapter_routing outside it {offenders}",
          not offenders)


def main():
    green("=== adapter-routing switch ===")
    test_config_defaults_and_validation()
    test_loader()
    test_plugin_wiring()
    test_import_boundary()
    print()
    if _failed:
        print(f"{_RED}{_failed} failed{_RESET}, {_passed} passed")
        sys.exit(1)
    green(f"all {_passed} checks passed")


if __name__ == "__main__":
    main()
