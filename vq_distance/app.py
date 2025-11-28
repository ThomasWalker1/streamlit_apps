import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

st.set_page_config(layout="wide", page_title="VQ Distance")

def get_model_structure():
    N_HIDDEN = 128
    model = nn.Sequential(
        nn.Linear(2, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 1)
    )
    return model.double()

def get_model_codes(model, x):
    with torch.no_grad():
        out0 = torch.sign(model[0](x))
        out1 = torch.sign(model[:3](x))
    return out0, out1

@st.cache_resource
def load_data():
    """
    Loads data and Pre-computes geometry and model codes for efficiency.
    Returns a dictionary containing all static assets.
    """
    model = get_model_structure()
    regions = None
    status = "init"
    
    regions_path = 'vq_distance/regions.pt'
    model_path = 'vq_distance/model.pt'


    regions = torch.load(regions_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    status = "loaded"

    verts = [r.numpy() for r in regions]
    centroids = torch.vstack([r.mean(dim=0) for r in regions])
    c_code0, c_code1 = get_model_codes(model, centroids)

    all_points = torch.vstack(regions)
    minval, _ = all_points.min(0)
    maxval, _ = all_points.max(0)
    
    return {
        "model": model,
        "verts": verts,
        "centroids": centroids,
        "c_codes": (c_code0, c_code1),
        "bounds": (minval, maxval),
        "status": status
    }

def plot_partition_optimized(verts, colors, ax, xlims, ylims, edgecolor='w', linewidth=.3, alpha=1):
    """
    Uses PolyCollection for fast rendering of many polygons.
    """
    # Create the collection once
    pc = PolyCollection(verts, facecolors=colors, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
    ax.add_collection(pc)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if xlims is not None:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

def main():
    st.title("VQ Distance Comparison to Euclidean Distance")

    data = load_data()

    minval, maxval = data["bounds"]
    centroids = data["centroids"]
    
    st.sidebar.header("Settings")
    
    with st.sidebar.form("settings_form"):
        st.write("Adjust Target Point")
        x_val = st.slider("Point X", float(minval[0]), float(maxval[0]), 0.25)
        y_val = st.slider("Point Y", float(minval[1]), float(maxval[1]), 0.1)
        submit_button = st.form_submit_button("Update Plot")
    
    point_np = np.array([x_val, y_val])
    point_t = torch.tensor(point_np).double()
    
    with st.spinner("Calculating distances..."):
        t_code0, t_code1 = get_model_codes(data["model"], point_t.unsqueeze(0))
        d0 = (t_code0 - data["c_codes"][0]).abs().sum(dim=1)
        d1 = (t_code1 - data["c_codes"][1]).abs().sum(dim=1)
        vq_distances = (d0 + d1).tolist()
        euc_distances = (point_t - centroids).norm(dim=1).tolist()
    
    plot_params = {
        "xlims": [minval[0].item(), maxval[0].item()],
        "ylims": [minval[1].item(), maxval[1].item()],
        "alpha": 0.5,
        "edgecolor": "black",
        "linewidth": 0.15,
        "verts": data["verts"]
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("VQ Distance")
        fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=150)
        
        max_vq = max(vq_distances) if max(vq_distances) > 0 else 1
        colors_vq = [plt.cm.jet(n / max_vq) for n in vq_distances]
        
        plot_partition_optimized(colors=colors_vq, ax=ax1, **plot_params)
        ax1.scatter(point_np[0], point_np[1], color='black', s=20)
        st.pyplot(fig1)

    with col2:
        st.subheader("Euclidean Distance")
        fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=150)
        
        max_euc = max(euc_distances) if max(euc_distances) > 0 else 1
        colors_euc = [plt.cm.jet(n / max_euc) for n in euc_distances]
        
        plot_partition_optimized(colors=colors_euc, ax=ax2, **plot_params)
        ax2.scatter(point_np[0], point_np[1], color='black', s=20)
        st.pyplot(fig2)

if __name__ == "__main__":
    main()