from pathlib import Path
import matplotlib.pyplot as plt

# LaTeX-like look (Computer Modern)
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

def render(eq: str, out_name: str, fontsize: int = 24):
    """
    Render a LaTeX-like math expression into a tightly-cropped SVG.
    Uses matplotlib mathtext (no external LaTeX install needed).
    """
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, eq, fontsize=fontsize)
    fig.savefig(
        OUT / out_name,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=True,
    )
    plt.close(fig)

render(r"$r(x)=\frac{d_1^2}{d_2^2+\varepsilon}$", "eq_ratio.svg", fontsize=26)
render(r"$r(x)<k,\quad k=\frac{1}{3}$", "eq_threshold.svg", fontsize=26)
render(r"$f_{\mathrm{mem}}(\tau)=\mathbb{E}\left[\mathbf{1}\{r(x_\tau)<k\}\right]$", "eq_fmem.svg", fontsize=24)

print("Saved:")
print(" -", OUT / "eq_ratio.svg")
print(" -", OUT / "eq_threshold.svg")
print(" -", OUT / "eq_fmem.svg")
