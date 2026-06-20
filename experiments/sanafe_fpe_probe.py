"""Dump the SANA-FE net structure passed to chip.load (which SIGFPEs on the mmixcore
net) to find the degenerate structural value the C++ loader divides by."""
import sys, os
_cfg = sys.argv[1]
sys.argv = ["run.py", "--headless", _cfg]
sys.path.insert(0, "."); sys.path.append("./src"); sys.path.append("./spikingjelly")

import sanafe

_orig_load = sanafe.SpikingChip.load


def _dump_net_and_exit(self, net, *a, **k):
    print("=== chip.load called; net dir ===", flush=True)
    print([x for x in dir(net) if not x.startswith("_")], flush=True)
    for accessor in ("groups", "neuron_groups", "get_neuron_groups", "mapped_neuron_groups"):
        obj = getattr(net, accessor, None)
        if obj is None:
            continue
        print(f"--- net.{accessor} ({type(obj).__name__}) ---", flush=True)
        try:
            items = obj.items() if hasattr(obj, "items") else (
                list(enumerate(obj)) if hasattr(obj, "__iter__") else [])
            print(f"   count={len(items)}", flush=True)
            for key, g in items:
                try:
                    neurons = list(g)
                except TypeError:
                    neurons = getattr(g, "neurons", [])
                size = len(neurons) if hasattr(neurons, "__len__") else "?"
                tag = " <<< EMPTY" if size == 0 else ""
                print(f"   group {key}: size={size}{tag} dir={[x for x in dir(g) if not x.startswith('_')][:8]}", flush=True)
        except Exception as e:
            print("   enumerate failed:", type(e).__name__, e, flush=True)
        break
    print("=== describe(net) / repr ===", flush=True)
    try:
        print(str(net)[:1500], flush=True)
    except Exception as e:
        print("repr failed:", e, flush=True)
    os._exit(0)


sanafe.SpikingChip.load = _dump_net_and_exit

if __name__ == "__main__":
    from src.init import init
    init()
    from run import _run_headless
    _run_headless(_cfg)
