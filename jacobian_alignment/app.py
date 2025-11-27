import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Model ---
class JacobianGluer(nn.Module):
    def __init__(self, points, jacobians, values, radius=2.0):
        super().__init__()
        self.points = points
        self.jacobians = jacobians
        self.values = values
        self.radius = radius

    def bump_function(self, dist_sq):
        safe_mask = dist_sq < (self.radius**2)
        r2 = dist_sq / (self.radius**2)
        weights = torch.zeros_like(dist_sq)
        if safe_mask.any():
            valid_r2 = r2[safe_mask]
            exponent = 1 - (1 / (1 - valid_r2))
            weights[safe_mask] = torch.exp(exponent)
        return weights

    def forward(self, x):
        total_output = torch.zeros(x.shape[0], self.values.shape[1], device=x.device)
        for i, center in enumerate(self.points):
            diff = x - center
            dist_sq = torch.sum(diff**2, dim=1)
            w_i = self.bump_function(dist_sq).unsqueeze(1)
            linear_term = self.values[i] + (diff @ self.jacobians[i].T)
            total_output += w_i * linear_term
        return total_output

# --- 2. Helper Functions ---

def normalize_vector(vec_tensor):
    norm = torch.norm(vec_tensor, p=2)
    if norm > 1e-6:
        return vec_tensor / norm
    return vec_tensor

def generate_base_directions(strategy, points_list):
    """
    Generates the initial 3 unit vectors (one per point).
    Returns tensor shape (3, 2).
    """
    points_t = torch.tensor(points_list)
    base_dirs = torch.zeros(3, 2)
    
    for p_idx in range(3):
        raw_vec = torch.zeros(2)
        if strategy == "Jacobian Aligned":
            raw_vec = points_t[p_idx]
        elif strategy == "Random":
            raw_vec = torch.randn(2)
            
        base_dirs[p_idx] = normalize_vector(raw_vec)
    
    return base_dirs

def apply_rotation_and_scale(base_dirs, angle_deg, scale):
    """
    Rotates base directions by angle_deg and scales them.
    Returns full Jacobian tensor (3 points, 3 dims, 2 coords).
    """
    # Create Rotation Matrix
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    
    # Rotate the base vectors
    rotated_dirs = base_dirs @ R.T
    
    # Scale
    scaled_dirs = rotated_dirs * scale
    
    # Construct Full Jacobian Tensor (Rank 1 Logic)
    new_jacs = torch.zeros(3, 3, 2)
    for p_idx in range(3):
        v = scaled_dirs[p_idx]
        for dim_idx in range(3):
            if p_idx == dim_idx:
                new_jacs[p_idx, dim_idx] = v   # Active (+v)
            else:
                new_jacs[p_idx, dim_idx] = -v  # Inactive (-v)
                
    return new_jacs.tolist()

# --- 3. Streamlit App ---

st.set_page_config(layout="wide", page_title="Jacobian Alignment")
st.title("Classifiers with Rank One Jacobians")
st.markdown("In the paper [What Deep Networks Want to Learn and How to Get There](https://arxiv.org/abs/2506.12284),\
            we show that optimal classifiers have Jacobians at the training data that are rank one. In particular, \
            the top singular vector of this rank one Jacobian is aligned with the training data -- a property we call\
            Jacobian aligned. Jacobian aligned deep networks are optimally robust amongst classifiers with rank one\
            Jacobians. In this demo we validate this theoretical result.")
st.markdown("Below is a function mapping a two-dimensional input space to a three-dimensional output space.\
            A point of the input space is classified according to which output dimension is maximized.\
            We construct a classifier from three input samples, one from each class.\
            At each input sample we place a bumpy function such that the classifier correctly classifies that point\
            and has a Jacobian which is rank one and aligned to some direction. Choosing 'Jacobian Aligned' will set this\
            direction equal to the input sample. Choosing 'Random' will choose a random direction. In the demo you can rotate\
            these directions as well as change the magnitude of the corresponding Jacobian, change the locations of the\
            three input samples, and change the radius of the bump function.")
st.markdown("The three plots show contour maps of each output dimension, the boundary at which the predicted class of the classifier\
            changes, and the distance of the input sample to that boundary (this is quantified exactly below the plot).")

# --- State Initialization ---
if 'points' not in st.session_state:
    st.session_state['points'] = [[0.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

# Helper function for the callback
def update_base_vectors():
    st.session_state['base_directions'] = generate_base_directions(
        st.session_state['strategy_selection'], 
        st.session_state['points']
    )

if 'base_directions' not in st.session_state:
    # Initialize manually first time
    st.session_state['base_directions'] = generate_base_directions("Jacobian Aligned", st.session_state['points'])

# --- Sidebar ---
st.sidebar.header("Input Samples")
pts = []
for i in range(3):
    c1, c2 = st.sidebar.columns(2)
    px = c1.number_input(f"P{i+1} X", value=st.session_state['points'][i][0], step=0.1, key=f"p{i}_x_in")
    py = c2.number_input(f"P{i+1} Y", value=st.session_state['points'][i][1], step=0.1, key=f"p{i}_y_in")
    pts.append([px, py])
st.session_state['points'] = pts

st.sidebar.markdown("---")
st.sidebar.header("Rank One Jacobian Structure")

# UPDATED: Using on_change callback for immediate updates
st.sidebar.radio(
    "Select:", 
    ("Jacobian Aligned", "Random"), 
    key="strategy_selection",
    on_change=update_base_vectors 
)

st.sidebar.markdown("---")
st.sidebar.header("Global Transform")

# ROTATION SLIDER
rotation = st.sidebar.slider("Rotate", 0, 360, 0, 5)
# NORM SLIDER
norm_val = st.sidebar.slider("Gradient Norm", 0.1, 3.0, 1.0, 0.1)

# Apply Rotation & Scale to Base Directions
st.session_state['jacobians'] = apply_rotation_and_scale(
    st.session_state['base_directions'], rotation, norm_val
)

st.sidebar.markdown("---")
st.sidebar.header("Bump")
radius = st.sidebar.slider("Radius", 0.5, 5.0, 2.5, 0.1)

# --- Calculation ---
points_t = torch.tensor(st.session_state['points'], dtype=torch.float32)
jacobians_t = torch.tensor(st.session_state['jacobians'], dtype=torch.float32)

# Targets (+1 self, -1 others)
input_values = torch.ones(3, 3) * -1.0
input_values.fill_diagonal_(1.0)

model = JacobianGluer(points_t, jacobians_t, input_values, radius=radius)

# Grid
grid_res = 120
x_np = np.linspace(-3, 3, grid_res)
y_np = np.linspace(-3, 3, grid_res)
X, Y = np.meshgrid(x_np, y_np)
flat_grid = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

with torch.no_grad():
    output = model(flat_grid)
    Z_all = output.detach().numpy().reshape(grid_res, grid_res, 3)

# --- Plotting ---
st.write("---")
cols = st.columns(3)

for dim_idx in range(3):
    Z_raw = Z_all[:, :, dim_idx]
    
    # Boundary / Margin Calculation (Restored)
    other_dims = [i for i in range(3) if i != dim_idx]
    Z_others = Z_all[:, :, other_dims]
    Z_max_others = np.max(Z_others, axis=2)
    Z_boundary = Z_raw - Z_max_others
    
    peak_pt = points_t[dim_idx].numpy()
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    # Background
    cp = ax.contourf(X, Y, Z_raw, levels=np.linspace(-2.0, 2.0, 40), cmap='coolwarm', alpha=0.6)
    
    # Boundary
    closest_pt = None
    min_dist = float('inf')

    if Z_boundary.max() >= 0 and Z_boundary.min() <= 0:
        cs = ax.contour(X, Y, Z_boundary, levels=[0], colors='blue', linewidths=2.5, linestyles='dashed')
        
        paths = cs.allsegs[0]
        if len(paths) > 0:
            all_contour_points = np.vstack(paths)
            dists = np.sqrt(np.sum((all_contour_points - peak_pt)**2, axis=1))
            min_index = np.argmin(dists)
            min_dist = dists[min_index]
            closest_pt = all_contour_points[min_index]
            
            # Margin Line
            ax.plot([peak_pt[0], closest_pt[0]], [peak_pt[1], closest_pt[1]], 
                    color='#00FF00', linewidth=2, linestyle='-')

    # Anchors
    for p_i in range(3):
        px, py = points_t[p_i].numpy()
        color = '#00CC00' if p_i == dim_idx else '#CC0000'
        size = 150 if p_i == dim_idx else 80
        ax.scatter(px, py, c=color, s=size, edgecolors='white', zorder=10)
        ax.text(px+0.1,py+0.1,f'P{p_i+1}',zorder=10000,fontweight='bold')
        
        jx, jy = jacobians_t[p_i, dim_idx].numpy()
        # Scale arrow for visibility
        if abs(jx) > 0.01 or abs(jy) > 0.01:
            ax.quiver(px, py, jx, jy, color='black', scale=1, scale_units='xy', angles='xy', width=0.01, zorder=5)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    with cols[dim_idx]:
        st.markdown(f'Output Dimension {dim_idx+1}')
        st.pyplot(fig)
        if closest_pt is not None:
             st.markdown(f"#### Margin at P{dim_idx}: {min_dist:.2f}")
             st.markdown(f"Jacobian at P{dim_idx}: {np.round(jacobians_t[dim_idx].numpy(),decimals=2)}")
        else:
             st.metric(f"Margin {dim_idx+1}", "Undefined")