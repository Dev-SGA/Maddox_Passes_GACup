import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(layout="wide", page_title="Pass Map Dashboard")
st.title("Pass Map Dashboard - Maddox")

# ==========================
# Constants
# ==========================
GOAL_X = 120
GOAL_Y = 40
FINAL_THIRD_LINE_X = 80  # final third line: x=80
PENALTY_AREA_X = 102  # penalty area starts at x=102
PROGRESSIVE_THRESHOLD = 0.75  # Opta-like progressive rule

# ==========================
# Data Setup
# ==========================
matches_data = {
    "Vs Los Angeles": [
        ("PASS SUCCESSFUL", 55.01, 13.24, 48.36, 3.59, None),
        ("PASS SUCCESSFUL", 51.36, 8.75, 83.94, 5.76, None),
        ("PASS SUCCESSFUL", 106.55, 11.57, 114.52, 20.38, None),
        ("PASS SUCCESSFUL", 113.36, 1.10, 104.72, 18.72, None),
        ("PASS SUCCESSFUL", 79.78, 41.50, 69.14, 28.53, None),
        ("PASS SUCCESSFUL", 99.06, 53.13, 107.88, 61.11, None),
        ("PASS UNSUCCESSFUL", 57.17, 4.76, 77.62, 40.83, None),
        ("PASS UNSUCCESSFUL", 86.10, 12.74, 99.90, 10.58, None),
    ],
    "Vs Slavia Praha": [
        ("PASS SUCCESSFUL", 26.75, 10.58, 33.07, 20.72, None),
        ("PASS SUCCESSFUL", 64.99, 6.42, 52.52, 10.24, None),
        ("PASS SUCCESSFUL", 58.67, 22.21, 70.14, 6.75, None),
        ("PASS SUCCESSFUL", 77.29, 1.77, 91.09, 9.08, None),
        ("PASS SUCCESSFUL", 91.92, 3.26, 86.93, 12.57, None),
        ("PASS SUCCESSFUL", 88.59, 5.09, 112.53, 61.11, None),
        ("PASS SUCCESSFUL", 92.25, 24.54, 108.54, 24.37, None),
        ("PASS SUCCESSFUL", 111.53, 1.60, 117.35, 5.76, None),
        ("PASS UNSUCCESSFUL", 114.69, 18.89, 108.04, 38.84, None),
        ("PASS UNSUCCESSFUL", 103.22, 17.06, 108.54, 20.88, None),
        ("PASS UNSUCCESSFUL", 100.89, 22.05, 106.38, 27.53, None),
        ("PASS UNSUCCESSFUL", 96.41, 7.92, 109.54, 60.61, None),
        ("PASS UNSUCCESSFUL", 88.76, 37.34, 95.24, 25.37, None),
    ],
    "Vs Sockers": [
        ("PASS SUCCESSFUL", 53.68, 5.42, 64.82, 13.74, None),
        ("PASS SUCCESSFUL", 92.58, 8.08, 97.74, 13.24, None),
        ("PASS SUCCESSFUL", 107.54, 10.41, 114.69, 48.64, None),
        ("PASS SUCCESSFUL", 90.59, 38.67, 104.38, 22.05, None),
        ("PASS SUCCESSFUL", 88.43, 28.53, 97.74, 28.20, None),
        ("PASS SUCCESSFUL", 46.70, 60.28, 57.51, 60.61, None),
        ("PASS SUCCESSFUL", 108.54, 73.25, 101.23, 76.74, None),
        ("PASS UNSUCCESSFUL", 104.05, 21.88, 117.18, 42.33, None),
    ],
}

# Create DataFrames for each match and combined
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfs_by_match[match_name] = pd.DataFrame(events, columns=["type", "start_x", "start_y", "end_x", "end_y", "video"])

# All games combined
df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All games": df_all}
full_data.update(dfs_by_match)

def get_style(event_type, has_video):
    """Returns marker, color (rgba), size, and linewidth based on event type"""
    event_type = event_type.upper()
    
    # 1. PASS SUCCESSFUL
    if "SUCCESSFUL" in event_type:
        # Green
        return 'o', (0.1, 0.95, 0.1, 1.0), 110, 0.5
    # 2. PASS UNSUCCESSFUL
    if "UNSUCCESSFUL" in event_type:
        # Red
        return 'x', (0.95, 0.1, 0.1, 1.0), 120, 3.0

    # Default
    return 'o', (0.5, 0.5, 0.5, 0.8), 90, 0.5


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute pass statistics"""
    total = len(df)
    
    # Pass counts
    is_pass = df['type'].str.contains('PASS', case=False)
    is_successful = df['type'].str.contains('SUCCESSFUL', case=False)
    
    all_passes = df[is_pass]
    total_passes = len(all_passes)
    successful_passes = all_passes[is_successful].shape[0]
    pass_rate = (successful_passes / total_passes * 100) if total_passes > 0 else 0
    
    # Final third passes
    final_third_mask = df['end_x'] > FINAL_THIRD_LINE_X
    final_third_passes = df[final_third_mask & is_pass]
    final_third_total = len(final_third_passes)
    final_third_successful = final_third_passes[is_successful].shape[0]
    final_third_rate = (final_third_successful / final_third_total * 100) if final_third_total > 0 else 0
    
    # Progressive passes (Opta-like)
    dist_start = np.sqrt((GOAL_X - df['start_x'])**2 + (GOAL_Y - df['start_y'])**2)
    dist_end = np.sqrt((GOAL_X - df['end_x'])**2 + (GOAL_Y - df['end_y'])**2)
    progressive_mask = dist_end <= dist_start * PROGRESSIVE_THRESHOLD
    progressive_passes = df[progressive_mask & is_pass]
    progressive_total = len(progressive_passes)
    progressive_successful = progressive_passes[is_successful].shape[0]
    progressive_rate = (progressive_successful / progressive_total * 100) if progressive_total > 0 else 0
    
    # Passes to the area (penalty area)
    area_mask = df['end_x'] > PENALTY_AREA_X
    area_passes = df[area_mask & is_pass]
    area_total = len(area_passes)
    area_successful = area_passes[is_successful].shape[0]
    area_rate = (area_successful / area_total * 100) if area_total > 0 else 0
    
    return {
        "pass_total": total_passes,
        "pass_wins": successful_passes,
        "pass_rate": pass_rate,
        "final_third_total": final_third_total,
        "final_third_wins": final_third_successful,
        "final_third_rate": final_third_rate,
        "progressive_total": progressive_total,
        "progressive_wins": progressive_successful,
        "progressive_rate": progressive_rate,
        "area_total": area_total,
        "area_wins": area_successful,
        "area_rate": area_rate,
        "fouls": 0,
    }


# ==========================
# Sidebar Configuration
# ==========================
st.sidebar.header("📋 Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

# Additional filter
filter_duel_type = st.sidebar.multiselect(
    "Pass Type",
    ["Successful", "Unsuccessful"],
    default=["Successful", "Unsuccessful"]
)

st.sidebar.divider()
st.sidebar.caption("Match filtered by selected options above")

# Get selected data
df = full_data[selected_match].copy()

# Apply pass type filter
if not all(x in filter_duel_type for x in ["Successful", "Unsuccessful"]):
    mask = pd.Series([False] * len(df))
    if "Successful" in filter_duel_type:
        mask |= df['type'].str.contains('SUCCESSFUL', case=False)
    if "Unsuccessful" in filter_duel_type:
        mask |= df['type'].str.contains('UNSUCCESSFUL', case=False)
    df = df[mask]

# Compute stats always from full match data
stats = compute_stats(full_data[selected_match])

# ==========================
# Main Layout
# ==========================
col_map, col_vid = st.columns([1, 1])

with col_map:
    st.subheader("Interactive Pitch Map")
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#f8f8f8', line_color='#4a4a4a')
    fig, ax = pitch.draw(figsize=(10, 7))

    for _, row in df.iterrows():
        has_vid = row["video"] is not None
        marker, color, size, lw = get_style(row["type"], has_vid)
        # For passes, draw arrows
        dx = row['end_x'] - row['start_x']
        dy = row['end_y'] - row['start_y']
        if "UNSUCCESSFUL" in row["type"].upper():
            head_w, head_l = 2, 2
        else:
            head_w, head_l = 1, 1
        ax.arrow(row['start_x'], row['start_y'], dx, dy, head_width=head_w, head_length=head_l, fc=color, ec=color, alpha=0.9)

    # Attack Arrow
    ax.annotate('', xy=(70, 83), xytext=(50, 83),
        arrowprops=dict(arrowstyle='->', color='#4a4a4a', lw=1.5))
    ax.text(60, 86, "Attack Direction", ha='center', va='center',
        fontsize=9, color='#4a4a4a', fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], color=(0.1, 0.95, 0.1, 1.0), lw=4, label='Successful Pass'),
        Line2D([0], [0], color=(0.95, 0.1, 0.1, 1.0), lw=4, label='Unsuccessful Pass'),
    ]

    # Apply legend to graphic
    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor='white',
        edgecolor='#333333',
        fontsize='small',
        title="Match Events",
        title_fontsize='medium',
        labelspacing=1.2,
        borderpad=1.0,
        framealpha=0.95
    )

    legend.get_title().set_fontweight('bold')

    # Convert plot to image for coordinate tracking
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_obj = Image.open(buf)

    # Use fixed width to ensure coordinate scaling works
    click = streamlit_image_coordinates(img_obj, width=700)

# ==========================
# Interaction Logic
# ==========================
selected_event = None

if click is not None:
    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    # Map pixel click to actual image pixels
    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)

    # Invert Y for Matplotlib logic and transform to pitch data coordinates
    mpl_pixel_y = real_h - pixel_y
    coords = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
    field_x, field_y = coords[0], coords[1]

    # Calculate distance to start or end
    df["dist_start"] = np.sqrt((df["start_x"] - field_x)**2 + (df["start_y"] - field_y)**2)
    df["dist_end"] = np.sqrt((df["end_x"] - field_x)**2 + (df["end_y"] - field_y)**2)
    df["dist"] = np.minimum(df["dist_start"], df["dist_end"])

    # Radius threshold for easier selection
    RADIUS = 5
    candidates = df[df["dist"] < RADIUS]

    if not candidates.empty:
        selected_event = candidates.loc[candidates["dist"].idxmin()]

# ==========================
# Video Display & Stats
# ==========================
with col_vid:
    st.subheader("Event Details")
    if selected_event is not None:
        st.success(f"**Selected Event:** {selected_event['type']}")
        st.info(f"**Start:** X: {selected_event['start_x']:.2f}, Y: {selected_event['start_y']:.2f}")
        st.info(f"**End:** X: {selected_event['end_x']:.2f}, Y: {selected_event['end_y']:.2f}")

        if selected_event["video"]:
            try:
                st.video(selected_event["video"])
            except:
                st.error(f"Video file not found: {selected_event['video']}")
        else:
            st.warning("No video footage available for this specific event.")
    else:
        st.info("Select an arrow on the pitch to view event details.")

    st.divider()
    st.subheader("Performance Statistics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Passes", f"{stats['pass_wins']}/{stats['pass_total']}", f"{stats['pass_rate']:.1f}% Success")
    col2.metric("Passes in the Final Third", f"{stats['final_third_wins']}/{stats['final_third_total']}", f"{stats['final_third_rate']:.1f}% Success")
    col3.metric("Progressive Passes", f"{stats['progressive_wins']}/{stats['progressive_total']}", f"{stats['progressive_rate']:.1f}% Success")
    col4.metric("Passes to the Area", f"{stats['area_wins']}/{stats['area_total']}", f"{stats['area_rate']:.1f}% Success")
