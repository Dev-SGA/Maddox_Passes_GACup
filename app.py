import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

coords_by_match = {
    'Vs Los Angeles': [
        (55.01, 13.24), (48.36, 3.59),
        (51.36, 8.75), (83.94, 5.76),
        (106.55, 11.57), (114.52, 20.38),
        (113.36, 1.10), (104.72, 18.72),
        (79.78, 41.50), (69.14, 28.53),
        (99.06, 53.13), (107.88, 61.11),
        (57.17, 4.76), (77.62, 40.83),
        (86.10, 12.74), (99.90, 10.58)
    ],
    'Vs Slavia Praha': [
        (26.75, 10.58), (33.07, 20.72),
        (64.99, 6.42), (52.52, 10.24),
        (58.67, 22.21), (70.14, 6.75),
        (77.29, 1.77), (91.09, 9.08),
        (91.92, 3.26), (86.93, 12.57),
        (88.59, 5.09), (112.53, 61.11),
        (92.25, 24.54), (108.54, 24.37),
        (111.53, 1.60), (117.35, 5.76),
        (114.69, 18.89), (108.04, 38.84),
        (103.22, 17.06), (108.54, 20.88),
        (100.89, 22.05), (106.38, 27.53),
        (96.41, 7.92), (109.54, 60.61),
        (88.76, 37.34), (95.24, 25.37)
    ],
    'Vs Sockers': [
        (53.68, 5.42), (64.82, 13.74),
        (92.58, 8.08), (97.74, 13.24),
        (107.54, 10.41), (114.69, 48.64),
        (90.59, 38.67), (104.38, 22.05),
        (88.43, 28.53), (97.74, 28.20),
        (46.70, 60.28), (57.51, 60.61),
        (108.54, 73.25), (101.23, 76.74),
        (104.05, 21.88), (117.18, 42.33)
    ]
}

passes_errados_by_match = {
    'Vs Los Angeles': [7, 8],
    'Vs Slavia Praha': [9, 10, 11, 12, 13],
    'Vs Sockers': [8]
}

st.set_page_config(layout="wide", page_title="Pass Map Dashboard")
st.title("Pass Map Dashboard")

# ==========================
# Configuration
# ==========================
GOAL_X = 120
GOAL_Y = 40
FINAL_THIRD_LINE_X = 80  # entry: start outside (x < 80) and end inside (x >= 80)

MATCHES = ["Vs Los Angeles", "Vs Slavia Praha", "Vs Sockers", "All Matches"]

st.sidebar.header("Match selection")
selected_match = st.sidebar.radio("Choose the match", MATCHES, index=0)

st.sidebar.header("Pass filter")
pass_filter = st.sidebar.radio(
    "Filter passes",
    ["All Passes", "Successful Only", "Unsuccessful Only"],
    index=0
)


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

    # Passes in final third: x_end >= 80
    df["in_final_third"] = df["x_end"] >= FINAL_THIRD_LINE_X

    # Passes to the box: x_end >= 100
    df["to_box"] = df["x_end"] >= 100
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    total_passes = len(df)
    successful = int(df["certo"].sum())
    unsuccessful = int(df["errado"].sum())

    accuracy = (successful / total_passes * 100.0) if total_passes else 0.0

    final_third_total = int(df["in_final_third"].sum())
    final_third_success = int((df["in_final_third"] & ~df["errado"]).sum())
    final_third_unsuccess = int((df["in_final_third"] & df["errado"]).sum())
    final_third_accuracy = (
        final_third_success / final_third_total * 100.0 if final_third_total else 0.0
    )

    box_total = int(df["to_box"].sum())
    box_success = int((df["to_box"] & ~df["errado"]).sum())
    box_unsuccess = int((df["to_box"] & df["errado"]).sum())
    box_accuracy = (
        box_success / box_total * 100.0 if box_total else 0.0
    )

    return {
        "total_passes": total_passes,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "final_third_total": final_third_total,
        "final_third_success": final_third_success,
        "final_third_unsuccess": final_third_unsuccess,
        "final_third_accuracy_pct": round(final_third_accuracy, 2),
        "box_total": box_total,
        "box_success": box_success,
        "box_unsuccess": box_unsuccess,
        "box_accuracy_pct": round(box_accuracy, 2),
    }


def draw_pass_map(df: pd.DataFrame):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f5f5f5", line_color="#4a4a4a")

    # Smaller map + similar resolution
    fig, ax = pitch.draw(figsize=(6.4, 4.2))
    fig.set_dpi(100)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)

    # Colors
    for _, row in df.iterrows():
        if row["errado"]:
            # red for unsuccessful
            color = (0.95, 0.18, 0.18, 0.65)
            width = 1.55
            headwidth = 2.25
            headlength = 2.25
        else:
            # green for successful
            color = (0.18, 0.8, 0.18, 0.65)
            width = 1.55
            headwidth = 2.25
            headlength = 2.25

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
            color=(0.18, 0.8, 0.18, 0.65),
            lw=2.5,
            label="Successful Pass",
        ),
        Line2D(
            [0],
            [0],
            color=(0.95, 0.18, 0.18, 0.65),
            lw=2.5,
            label="Unsuccessful Pass",
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


if selected_match == "All Matches":
    all_coords = []
    all_errados = []
    offset = 0
    for match in MATCHES[:-1]:  # exclude "All Matches"
        coords_match = coords_by_match[match]
        errados_match = passes_errados_by_match[match]
        all_coords.extend(coords_match)
        all_errados.extend([e + offset for e in errados_match])
        offset += len(coords_match) // 2
    coords = all_coords
    errados = all_errados
else:
    coords = coords_by_match[selected_match]
    errados = passes_errados_by_match[selected_match]

df = build_df(coords, errados)

# Apply pass filter
if pass_filter == "Successful Only":
    df = df[df["certo"]].reset_index(drop=True)
elif pass_filter == "Unsuccessful Only":
    df = df[df["errado"]].reset_index(drop=True)

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

    st.subheader("Final Third")
    c7, c8, c9 = st.columns(3)
    c7.metric("Total", stats["final_third_total"])
    c8.metric("Successful", stats["final_third_success"])
    c9.metric("Unsuccessful", stats["final_third_unsuccess"])
    st.metric("Accuracy", f'{stats["final_third_accuracy_pct"]:.1f}%')

    st.divider()

    st.subheader("Passes to the Box")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total", stats["box_total"])
    d2.metric("Successful", stats["box_success"])
    d3.metric("Unsuccessful", stats["box_unsuccess"])
    st.metric("Accuracy", f'{stats["box_accuracy_pct"]:.1f}%')

with col_map:
    st.subheader("Pass Map")
    img = draw_pass_map(df)
    st.image(img, width=620)
