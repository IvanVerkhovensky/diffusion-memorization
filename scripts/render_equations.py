# scripts/render_equations.py
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

def render(eq: str, out_name: str, fontsize: int = 22):
    # White background so it is visible in GitHub dark mode
    fig = plt.figure(figsize=(0.01, 0.01), facecolor="white")
    fig.text(0, 0, eq, fontsize=fontsize, color="black")

    fig.savefig(
        OUT / out_name,
        bbox_inches="tight",
        pad_inches=0.10,
        transparent=False,     # IMPORTANT: no transparency
        facecolor="white",     # IMPORTANT: bake white background into SVG
    )
    plt.close(fig)

render(r"$r(x)=\frac{d_1^2}{d_2^2+\varepsilon}$", "eq_ratio.svg", fontsize=24)
render(r"$r(x)<k,\quad k=\frac{1}{3}$", "eq_threshold.svg", fontsize=24)
render(r"$f_{\mathrm{mem}}(\tau)=\mathbb{E}\left[\mathbf{1}\{r(x_\tau)<k\}\right]$", "eq_fmem.svg", fontsize=20)

print("Saved:")
print(" -", OUT / "eq_ratio.svg")
print(" -", OUT / "eq_threshold.svg")
print(" -", OUT / "eq_fmem.svg")