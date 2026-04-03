import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

try:
    from dashboard_pass_data import coords_by_match, passes_errados_by_match
except ImportError as e:
    raise SystemExit(
        "Could not import `dashboard_pass_data.py`. "
        "Make sure `dashboard_pass_data.py` exists in the same folder."
    ) from e


st.set_page_config(layout="wide", page_title="Pass Map Dashboard")
st.title("Pass Map Dashboard")

# ==========================
# Configuration
# ==========================
GOAL_X = 120
GOAL_Y = 40
FINAL_THIRD_LINE_X = 80  # entry: start outside (x < 80) and end inside (x >= 80)
PROGRESSIVE_THRESHOLD = 0.75  # Opta rule (your notebook)

MATCHES = ["Vs Los Angeles", "Vs Slavia Praha", "Vs Sockers"]

st.sidebar.header("Match selection")
selected_match = st.sidebar.radio("Choose the match", MATCHES, index=MATCHES.index("Vs Los Angeles"))


def build_df(coords: list[tuple[float, float]], passes_errados: list[int]) -> pd.DataFrame:
    passes = []
    for i in range(0, len(coords), 2):
        start = coords[i]
        end = coords[i + 1]
        numero = i // 2 + 1  # 1-indexed within the match
        passes.append(
            {
                "numero": numero,
                "x_start": float(start[0]),
                "y_start": float(start[1]),
                "x_end": float(end[0]),
                "y_end": float(end[1]),
            }
        )

    df = pd.DataFrame(passes)
    df["errado"] = df["numero"].isin(passes_errados)
    df["certo"] = ~df["errado"]

    # Opta-like progressive rule (your notebook)
    dist_inicio = np.sqrt((GOAL_X - df["x_start"]) ** 2 + (GOAL_Y - df["y_start"]) ** 2)
    dist_fim = np.sqrt((GOAL_X - df["x_end"]) ** 2 + (GOAL_Y - df["y_end"]) ** 2)
    df["progressive"] = dist_fim <= dist_inicio * PROGRESSIVE_THRESHOLD

    # Final third entry: starts outside and ends inside
    df["into_final_third"] = (df["x_start"] < FINAL_THIRD_LINE_X) & (
        df["x_end"] >= FINAL_THIRD_LINE_X
    )

    # Directions
    df["forward"] = df["x_end"] > df["x_start"]
    df["backward"] = df["x_end"] < df["x_start"]
    df["right"] = df["y_end"] > df["y_start"]
    df["left"] = df["y_end"] < df["y_start"]
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    total_passes = len(df)
    successful = int(df["certo"].sum())
    unsuccessful = int(df["errado"].sum())

    accuracy = (successful / total_passes * 100.0) if total_passes else 0.0

    progressive_count = int(df["progressive"].sum())
    progressive_pct = (progressive_count / total_passes * 100.0) if total_passes else 0.0

    final_third_total = int(df["into_final_third"].sum())
    final_third_success = int((df["into_final_third"] & ~df["errado"]).sum())
    final_third_unsuccess = int((df["into_final_third"] & df["errado"]).sum())
    final_third_accuracy = (
        final_third_success / final_third_total * 100.0 if final_third_total else 0.0
    )

    forward_count = int(df["forward"].sum())
    backward_count = int(df["backward"].sum())
    right_count = int(df["right"].sum())
    left_count = int(df["left"].sum())

    forward_pct_total = (forward_count / total_passes * 100.0) if total_passes else 0.0

    return {
        "total_passes": total_passes,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "progressive_passes": progressive_count,
        "progressive_pct": round(progressive_pct, 2),
        "final_third_entries": final_third_total,
        "final_third_success": final_third_success,
        "final_third_unsuccess": final_third_unsuccess,
        "final_third_accuracy_pct": round(final_third_accuracy, 2),
        "forward_passes": forward_count,
        "forward_pct_total": round(forward_pct_total, 2),
        "backward_passes": backward_count,
        "right_passes": right_count,
        "left_passes": left_count,
    }


def draw_pass_map(df: pd.DataFrame):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f5f5f5", line_color="#4a4a4a")

    # Smaller map + similar resolution
    fig, ax = pitch.draw(figsize=(6.4, 4.2))
    fig.set_dpi(100)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)

    # Colors (stronger)
    for _, row in df.iterrows():
        if row["errado"]:
            # weak red but stronger than previous
            color = (0.95, 0.18, 0.18, 0.65)
            width = 1.55
            headwidth = 2.25
            headlength = 2.25
        elif row["progressive"]:
            # stronger blue
            color = (0.18, 0.55, 1.0, 0.60)
            width = 1.70
            headwidth = 2.35
            headlength = 2.35
        else:
            # light gray for completed non-progressive
            color = (0.78, 0.78, 0.78, 0.18)
            width = 1.25
            headwidth = 1.95
            headlength = 1.95

        pitch.arrows(
            row["x_start"],
            row["y_start"],
            row["x_end"],
            row["y_end"],
            color=color,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
            ax=ax,
        )

    ax.set_title(f"Pass Map - {selected_match}", fontsize=12)

    # Elegant smaller legend top-left
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=(0.18, 0.55, 1.0, 0.60),
            lw=2.5,
            label="Progressive Pass",
        ),
        Line2D(
            [0],
            [0],
            color=(0.95, 0.18, 0.18, 0.65),
            lw=2.5,
            label="Unsuccessful Pass",
        ),
        Line2D(
            [0],
            [0],
            color=(0.78, 0.78, 0.78, 0.18),
            lw=2.5,
            label="Successful Pass",
        ),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        shadow=False,
        fontsize="x-small",
        labelspacing=0.5,
        borderpad=0.5,
    )
    legend.get_frame().set_alpha(1.0)

    # Attack direction arrow: middle-bottom
    arrow = FancyArrowPatch(
        (0.45, 0.05),
        (0.55, 0.05),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2,
        color="#333333",
    )
    fig.patches.append(arrow)
    fig.text(
        0.5,
        0.02,
        "Attack Direction",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()

    # Render controlled to avoid oversized display
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


coords = coords_by_match[selected_match]
errados = passes_errados_by_match[selected_match]
df = build_df(coords, errados)
stats = compute_stats(df)

# ==========================
# Dashboard layout
# ==========================
col_stats, col_map = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Passes", stats["total_passes"])
    c2.metric("Successful", stats["successful_passes"])
    c3.metric("Accuracy", f'{stats["accuracy_pct"]:.1f}%')
    c4.metric("Unsuccessful", stats["unsuccessful_passes"])

    st.divider()

    c5, c6 = st.columns(2)
    c5.metric("Progressive Passes", stats["progressive_passes"])
    c6.metric("Progressive % of Total", f'{stats["progressive_pct"]:.1f}%')

    st.divider()

    st.subheader("Final Third (Entry)")
    c7, c8, c9 = st.columns(3)
    c7.metric("Total Entries", stats["final_third_entries"])
    c8.metric("Successful", stats["final_third_success"])
    c9.metric("Unsuccessful", stats["final_third_unsuccess"])
    st.metric("Entry Accuracy", f'{stats["final_third_accuracy_pct"]:.1f}%')

    st.divider()

    st.subheader("Directions")
    d1, d2 = st.columns(2)
    d1.metric("Forward", f'{stats["forward_passes"]} ({stats["forward_pct_total"]:.1f}% of total)')
    d2.metric("Backward", stats["backward_passes"])

    d3, d4 = st.columns(2)
    d3.metric("Right", stats["right_passes"])
    d4.metric("Left", stats["left_passes"])

with col_map:
    st.subheader("Pass Map")
    img = draw_pass_map(df)
    st.image(img, width=620)
