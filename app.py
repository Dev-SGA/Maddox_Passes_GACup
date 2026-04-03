import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Pass Map Dashboard (Interactive)")
st.title("Pass Map Dashboard")
st.caption("Clique na bolinha no início do passe para ver o vídeo (se houver).")

# ==========================
# Configuration
# ==========================
FINAL_THIRD_LINE_X = 80

# ==========================
# DATA
# ==========================
matches_data = {
    "Vs Los Angeles": [
        ("PASS WON", 55.01, 13.24, 48.36, 3.59, None),
        ("PASS WON", 51.36, 8.75, 83.94, 5.76, None),
        ("PASS WON", 106.55, 11.57, 114.52, 20.38, None),
        ("PASS WON", 113.36, 1.10, 104.72, 18.72, None),
        ("PASS WON", 79.78, 41.50, 69.14, 28.53, None),
        ("PASS WON", 99.06, 53.13, 107.88, 61.11, None),

        ("PASS LOST", 57.17, 4.76, 77.62, 40.83, None),
        ("PASS LOST", 86.10, 12.74, 99.90, 10.58, None),
    ],
    "Vs Slavia Praha": [
        ("PASS WON", 26.75, 10.58, 33.07, 20.72, None),
        ("PASS WON", 64.99, 6.42, 52.52, 10.24, None),
        ("PASS WON", 58.67, 22.21, 70.14, 6.75, None),
        ("PASS WON", 77.29, 1.77, 91.09, 9.08, None),
        ("PASS WON", 91.92, 3.26, 86.93, 12.57, None),
        ("PASS WON", 88.59, 5.09, 112.53, 61.11, None),
        ("PASS WON", 92.25, 24.54, 108.54, 24.37, None),
        ("PASS WON", 111.53, 1.60, 117.35, 5.76, None),

        ("PASS LOST", 114.69, 18.89, 108.04, 38.84, None),
        ("PASS LOST", 103.22, 17.06, 108.54, 20.88, None),
        ("PASS LOST", 100.89, 22.05, 106.38, 27.53, None),
        ("PASS LOST", 96.41, 7.92, 109.54, 60.61, None),
        ("PASS LOST", 88.76, 37.34, 95.24, 25.37, None),
    ],
    "Vs Sockers": [
        ("PASS WON", 53.68, 5.42, 64.82, 13.74, None),
        ("PASS WON", 92.58, 8.08, 97.74, 13.24, None),
        ("PASS WON", 107.54, 10.41, 114.69, 48.64, "videos/Maddox - Assist.mp4"),
        ("PASS WON", 90.59, 38.67, 104.38, 22.05, None),
        ("PASS WON", 88.43, 28.53, 97.74, 28.20, None),
        ("PASS WON", 46.70, 60.28, 57.51, 60.61, None),
        ("PASS WON", 108.54, 73.25, 101.23, 76.74, None),

        ("PASS LOST", 104.05, 21.88, 117.18, 42.33, None),
    ],
}

# ==========================
# Build DataFrames
# ==========================
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(
        events,
        columns=["type", "x_start", "y_start", "x_end", "y_end", "video"]
    )
    dfm["numero"] = np.arange(1, len(dfm) + 1)
    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All Matches": df_all}
full_data.update(dfs_by_match)

# ==========================
# Helpers
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""

def compute_stats(df: pd.DataFrame) -> dict:
    total_passes = len(df)
    successful = int(df["type"].str.contains("WON", case=False).sum())
    unsuccessful = int(df["type"].str.contains("LOST", case=False).sum())
    accuracy = (successful / total_passes * 100.0) if total_passes else 0.0

    in_final_third = df["x_end"] >= FINAL_THIRD_LINE_X
    final_third_total = int(in_final_third.sum())
    final_third_success = int((in_final_third & df["type"].str.contains("WON", case=False)).sum())
    final_third_unsuccess = int((in_final_third & df["type"].str.contains("LOST", case=False)).sum())
    final_third_accuracy = (final_third_success / final_third_total * 100.0) if final_third_total else 0.0

    to_box = df["x_end"] >= 100
    box_total = int(to_box.sum())
    box_success = int((to_box & df["type"].str.contains("WON", case=False)).sum())
    box_unsuccess = int((to_box & df["type"].str.contains("LOST", case=False)).sum())
    box_accuracy = (box_success / box_total * 100.0) if box_total else 0.0

    return {
        "total_passes": total_passes,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "assists": 1,
        "final_third_total": final_third_total,
        "final_third_success": final_third_success,
        "final_third_unsuccess": final_third_unsuccess,
        "final_third_accuracy_pct": round(final_third_accuracy, 2),
        "box_total": box_total,
        "box_success": box_success,
        "box_unsuccess": box_unsuccess,
        "box_accuracy_pct": round(box_accuracy, 2),
    }

# ==========================
# Draw pass map
# ==========================
def draw_pass_map(df: pd.DataFrame, title: str):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f5f5f5", line_color="#4a4a4a")
    fig, ax = pitch.draw(figsize=(7.9, 5.3))
    fig.set_dpi(110)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)

    START_DOT_SIZE = 45

    for _, row in df.iterrows():
        is_lost = "LOST" in row["type"].upper()
        has_vid = has_video_value(row["video"])

        if is_lost:
            color = (0.95, 0.18, 0.18, 0.65)
        else:
            color = (0.18, 0.8, 0.18, 0.65)

        pitch.arrows(
            row["x_start"], row["y_start"],
            row["x_end"], row["y_end"],
            color=color,
            width=1.55,
            headwidth=2.25,
            headlength=2.25,
            ax=ax,
            zorder=3,
        )

        if has_vid:
            pitch.scatter(
                row["x_start"], row["y_start"],
                s=95,
                marker="o",
                facecolors="none",
                edgecolors="#FFD54F",
                linewidths=2.0,
                ax=ax,
                zorder=4,
            )

        pitch.scatter(
            row["x_start"], row["y_start"],
            s=START_DOT_SIZE,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=0.8,
            ax=ax,
            zorder=5,
        )

    ax.set_title(title, fontsize=12)

    legend_elements = [
        Line2D([0], [0], color=(0.18, 0.8, 0.18, 0.65), lw=2.5, label="Successful Pass"),
        Line2D([0], [0], color=(0.95, 0.18, 0.18, 0.65), lw=2.5, label="Unsuccessful Pass"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="gray", markeredgecolor="white",
               markersize=6, label="Start point (click)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="gray", markeredgecolor="#FFD54F",
               markeredgewidth=2, markersize=7, label="Has video"),
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

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig

# ==========================
# Sidebar
# ==========================
st.sidebar.header("Match selection")
selected_match = st.sidebar.radio("Choose the match", list(full_data.keys()), index=0)

st.sidebar.header("Pass filter")
pass_filter = st.sidebar.radio(
    "Filter passes",
    ["All Passes", "Successful Only", "Unsuccessful Only"],
    index=0
)

df = full_data[selected_match].copy()

if pass_filter == "Successful Only":
    df = df[df["type"].str.contains("WON", case=False)].reset_index(drop=True)
elif pass_filter == "Unsuccessful Only":
    df = df[df["type"].str.contains("LOST", case=False)].reset_index(drop=True)

stats = compute_stats(df)

# ==========================
# Layout
# ==========================
col_stats, col_right = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Passes", stats["total_passes"])
    c2.metric("Successful", stats["successful_passes"])
    c3.metric("Accuracy", f'{stats["accuracy_pct"]:.1f}%')
    c4.metric("Assists", stats["assists"])

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

with col_right:
    st.subheader("Pass Map (click the start dot)")

    img_obj, ax, fig = draw_pass_map(df, title=f"Pass Map - {selected_match}")
    click = streamlit_image_coordinates(img_obj, width=780)

    selected_pass = None

    if click is not None:
        real_w, real_h = img_obj.size
        disp_w, disp_h = click["width"], click["height"]

        pixel_x = click["x"] * (real_w / disp_w)
        pixel_y = click["y"] * (real_h / disp_h)

        mpl_pixel_y = real_h - pixel_y
        coords_clicked = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
        field_x, field_y = coords_clicked[0], coords_clicked[1]

        df_sel = df.copy()
        df_sel["dist"] = np.sqrt(
            (df_sel["x_start"] - field_x) ** 2 +
            (df_sel["y_start"] - field_y) ** 2
        )

        RADIUS = 7.0
        candidates = df_sel[df_sel["dist"] < RADIUS]

        if not candidates.empty:
            selected_pass = candidates.loc[candidates["dist"].idxmin()]

    plt.close(fig)

    st.divider()
    st.subheader("Video")

    if selected_pass is None:
        st.info("Clique na bolinha no início do passe para ver o vídeo (se houver).")
    else:
        st.success(f"Selected pass: #{int(selected_pass['numero'])} ({selected_pass['type']})")
        st.write(
            f"Start: ({selected_pass['x_start']:.2f}, {selected_pass['y_start']:.2f})  \n"
            f"End: ({selected_pass['x_end']:.2f}, {selected_pass['y_end']:.2f})"
        )

        if has_video_value(selected_pass["video"]):
            try:
                st.video(selected_pass["video"])
            except Exception:
                st.error(f"Video file not found: {selected_pass['video']}")
        else:
            st.warning("Não há vídeo carregado para este evento.")
