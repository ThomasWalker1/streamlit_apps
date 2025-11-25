import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import glob

# --- 1. Page Config ---
st.set_page_config(layout="wide", page_title="Centroid Affinity")

# --- 2. CSS for layout ---
st.markdown("""
    <style>
        .main > div { max-width: 1600px; margin: 0 auto; }
        h1, h2, h3, p { text-align: center; }
        .stCheckbox { padding-top: 5px; }
        /* CHANGED padding-top from 2rem to 5rem */
        .block-container { padding-top: 5rem; padding-bottom: 5rem; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER: Find available datasets ---
def get_dataset_names():
    files = glob.glob("cah_app/data/xs-*.npy")
    names = []
    for f in files:
        clean_name = os.path.basename(f)[3:-4]
        names.append(clean_name)
    if not names:
        return ["MetFacesSample"]
    return sorted(list(set(names)))

# --- 3. Data Loading ---
@st.cache_data
def load_data(dataset_name):
    xs_file = f"cah_app/data/xs-{dataset_name}.npy"
    cs_file = f"cah_app/data/cs-{dataset_name}.npy"
    feat_file = f"cah_app/data/features-{dataset_name}.npy"

    if os.path.exists(xs_file) and os.path.exists(cs_file):
        xs = np.load(xs_file)
        cs = np.load(cs_file)
    else:
        xs = np.random.randn(1000, 2)
        cs = np.random.randn(1000, 2) + 2

    if os.path.exists(feat_file):
        features = np.load(feat_file)
    else:
        features = np.random.choice(range(16), size=xs.shape[0])

    f = features.squeeze()
    unique_feats = np.unique(f)
    all_indices = []
    for feat in unique_feats:
        idx = np.where(f == feat)[0]
        if len(idx) > 2000:
            idx = np.random.permutation(idx)[:10_000]
        all_indices.append(idx)
    
    final_idx = np.concatenate(all_indices)
    xs = xs[final_idx]
    cs = cs[final_idx]
    features = features[final_idx]
    
    features = np.unique(features, return_inverse=True)[1] + 1

    # Rotation Logic
    xs_rotated = np.zeros_like(xs)
    xs_rotated[:, 0] = xs[:, 1] 
    xs_rotated[:, 1] = -xs[:, 0]
    xs = xs_rotated

    cs_rotated = np.zeros_like(cs)
    cs_rotated[:, 0] = cs[:, 1]
    cs_rotated[:, 1] = -cs[:, 0]
    cs = cs_rotated

    bounds = {
        'x_left':  [float(xs[:,0].min()), float(xs[:,0].max())],
        'y_left':  [float(xs[:,1].min()), float(xs[:,1].max())],
        'x_right': [float(cs[:,0].min()) - 0.5, float(cs[:,0].max()) + 0.5],
        'y_right': [float(cs[:,1].min()) - 0.5, float(cs[:,1].max()) + 0.5],
    }
    return xs, cs, features, bounds

# --- 4. Fast Filter & Sample ---
#@st.cache_data
def get_display_dataframe(selected, _xs, _cs, _cats, max_points=2000):
    if not selected:
        return pd.DataFrame(columns=['id', 'category', 'x_left', 'y_left', 'x_right', 'y_right'])

    mask = np.isin(_cats, selected)
    indices = np.where(mask)[0]
    
    if len(indices) > max_points:
        indices = np.random.choice(indices, size=max_points, replace=False)
    
    df = pd.DataFrame({
        'id': indices,
        'category': _cats[indices],
        'x_left': _xs[indices, 0],
        'y_left': _xs[indices, 1],
        'x_right': _cs[indices, 0],
        'y_right': _cs[indices, 1],
    })
    return df

# ==========================================
#      LAYOUT SETUP
# ==========================================

# 1. Dataset Selector (Small, Centered)
available_datasets = get_dataset_names()
c_top_1, c_top_2, c_top_3 = st.columns([2,1,2])
with c_top_2:
    selected_dataset = st.selectbox("Experiment", available_datasets, label_visibility="collapsed")

# Load Data
IMAGE_PATH = f"cah_app/data/output-{selected_dataset}.png"
xs, cs, features, bounds = load_data(selected_dataset)
unique_features = sorted(list(set(features)))

# 2. Define Visual Containers (Order of appearance)
charts_container = st.container()
st.write("---")
controls_container = st.container()

# ==========================================
#      LOGIC: Controls (Run First)
# ==========================================

with controls_container:
    selected_cats = []
    
    # Grid Layout for Checkboxes (8 columns for compactness)
    cols = st.columns(8)
    
    for i, category in enumerate(unique_features):
        col_idx = i % 8
        with cols[col_idx]:
            # Default: Select first 2 if nothing else selected yet
            default_val = (category in unique_features[:2])
            
            # Simple Checkbox
            if st.checkbox(f"{category}", value=default_val, key=f"c_{category}_{selected_dataset}"):
                selected_cats.append(category)

# ==========================================
#      LOGIC: Charts (Run Second)
# ==========================================

with charts_container:
    df_display = get_display_dataframe(selected_cats, xs, cs, features)
    
    # REDUCED HEIGHT HERE (300px)
    chart_height = 580 
    point_size = 30 
    opacity = 1.0
    color_scale = alt.Scale(scheme='category10', domain=unique_features)
    clean_axis = alt.Axis(labels=False, ticks=False, title=None, domain=True, grid=True, gridWidth=1)

    # --- LEFT CHART (Fixed Bounds) ---
    left_chart = alt.Chart(df_display).mark_circle(size=point_size, opacity=opacity).encode(
        x=alt.X('x_left', scale=alt.Scale(domain=bounds['x_left']), axis=clean_axis),
        y=alt.Y('y_left', scale=alt.Scale(domain=bounds['y_left']), axis=clean_axis),
        color=alt.Color('category:N', scale=color_scale, legend=None), 
        tooltip=['id', 'category']
    ).properties(height=chart_height)

    # --- RIGHT CHART (Dynamic Bounds) ---
    # We removed 'domain=bounds['x_right']' and replaced it with 'zero=False'
    # This tells Altair to auto-scale based on the data in df_display, 
    # without forcing the (0,0) origin to be visible.
    right_chart = alt.Chart(df_display).mark_circle(size=point_size, opacity=opacity).encode(
        x=alt.X('x_right', scale=alt.Scale(zero=False, padding=1), axis=clean_axis),
        y=alt.Y('y_right', scale=alt.Scale(zero=False, padding=1), axis=clean_axis),
        color=alt.Color('category:N', scale=color_scale, legend=None),
        tooltip=['id', 'category']
    ).properties(height=chart_height)

    if df_display.empty:
        st.warning("No categories selected.")
    else:
        c_img, c_left, c_right = st.columns(3)

        with c_img:
            st.caption(f"Network Output")
            if os.path.exists(IMAGE_PATH):
                st.image(IMAGE_PATH, use_container_width=True)
            else:
                # Reduced placeholder height
                st.markdown(
                    f"""<div style="width:100%; height:{chart_height}px; background-color:#f0f2f6; 
                    display:flex; justify-content:center; align-items:center; border-radius:5px; color:#555;">
                    Image Not Found
                    </div>""", 
                    unsafe_allow_html=True
                )

        with c_left:
            st.caption("Input Samples (Fixed View)")
            st.altair_chart(left_chart, theme="streamlit", use_container_width=True)

        with c_right:
            st.caption("Centroids (Dynamic View)")
            st.altair_chart(right_chart, theme="streamlit", use_container_width=True)