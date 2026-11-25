""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S
import matplotlib.pyplot as plt

S.setup()
plt.rcParams["figure.constrained_layout.use"] = False
S.INK_CLEAR_PT = 1.5


def findings(upper, lower, dy_pt, dx_pt):
    ""
    fig = plt.figure(figsize=(3, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.annotate(upper, xy=(0.5, 0.6), xycoords="axes fraction", ha="center", va="center", fontsize=7.0)
    ax.annotate(lower, xy=(0.5, 0.6), xycoords="axes fraction", xytext=(dx_pt, -dy_pt), textcoords="offset points",
                ha="left", va="baseline", fontsize=6.5)
    fig.canvas.draw()
    fails, _ = S.check_fig(fig, "t")
    plt.close(fig)
    return [f for f in fails if "ink clearance" in f]


assert not findings("Required rate", "Required rate met", 10.0, 2.0), "a descender outside the span counted"

assert findings("Required rate", "Required rate met", 10.0, -25.0), "a descender inside the span was not counted"

assert findings("Declared rate", "Declared rate met", 3.0, 2.0), "touching labels passed"

assert not findings("Declared rate", "Declared rate met", 20.0, 2.0), "distant labels failed"
print("OK tests/test_ink_overlap.py: 4 checks")
