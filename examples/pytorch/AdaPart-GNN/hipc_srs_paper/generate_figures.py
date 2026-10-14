#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 10

# ---------------------------------------------------------------------------
# Figure 1: Architectural Overview (Vector PDF)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), dpi=300)

# (a) Input graph with Jaccard-Eigenvector edge weighting
ax = axes[0]
ax.set_xlim(-0.2, 3.2)
ax.set_ylim(-0.2, 2.5)
ax.axis('off')
ax.set_title("(a) Spectral Edge Weighting", fontsize=8.5, fontweight='bold', pad=4)

# Nodes
nodes_a = [(0.5, 1.8), (1.5, 2.1), (2.5, 1.8), (0.8, 0.6), (1.5, 1.0), (2.2, 0.6)]
labels_a = ['A', 'B', 'C', 'D', 'E', 'F']
colors_a = ['#4A90E2', '#D0021B', '#4A90E2', '#4A90E2', '#D0021B', '#4A90E2'] # Red = high centrality

# Edges (u, v, weight, is_heavy)
edges_a = [
    (0, 1, '0.85', True), (1, 2, '0.22', False),
    (0, 3, '0.78', True), (1, 4, '0.92', True),
    (2, 5, '0.81', True), (3, 4, '0.71', True),
    (4, 5, '0.19', False)
]

for u, v, w, heavy in edges_a:
    x1, y1 = nodes_a[u]
    x2, y2 = nodes_a[v]
    lw = 2.4 if heavy else 1.0
    color = '#2E7D32' if heavy else '#B0BEC5'
    ls = '-' if heavy else '--'
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=ls, zorder=1)
    # Weight label
    mx, my = (x1 + x2)/2, (y1 + y2)/2
    ax.text(mx, my+0.07, w, fontsize=6.5, ha='center', va='bottom', color='#37474F')

for i, (x, y) in enumerate(nodes_a):
    circle = plt.Circle((x, y), 0.18, color=colors_a[i], ec='black', lw=1.0, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, labels_a[i], color='white', fontweight='bold', fontsize=7.5, ha='center', va='center', zorder=3)

ax.text(1.5, -0.15, "Solid: High Jaccard-Eigen Weight\nDashed: Low Weight (Cut Candidate)", fontsize=6.5, ha='center', color='#555')


# (b) Scale-Adaptive Partitions (K*)
ax = axes[1]
ax.set_xlim(-0.2, 3.2)
ax.set_ylim(-0.2, 2.5)
ax.axis('off')
ax.set_title("(b) Scale-Adaptive Subgraphs", fontsize=8.5, fontweight='bold', pad=4)

# Partition 1 box
p1_box = plt.Rectangle((0.15, 0.3), 1.25, 1.9, fc='#E3F2FD', ec='#1976D2', lw=1.2, ls='--', zorder=0)
ax.add_patch(p1_box)
ax.text(0.77, 2.05, r"Partition $G_1$", fontsize=7.5, fontweight='bold', color='#1565C0', ha='center')

# Partition 2 box
p2_box = plt.Rectangle((1.8, 0.3), 1.25, 1.9, fc='#FBE9E7', ec='#D84315', lw=1.2, ls='--', zorder=0)
ax.add_patch(p2_box)
ax.text(2.42, 2.05, r"Partition $G_2$", fontsize=7.5, fontweight='bold', color='#BF360C', ha='center')

# Internal edges
ax.plot([0.5, 1.1], [1.7, 1.7], color='#1976D2', lw=2.0)
ax.plot([0.5, 0.8], [1.7, 0.7], color='#1976D2', lw=2.0)
ax.plot([0.8, 1.1], [0.7, 0.7], color='#1976D2', lw=2.0)
ax.plot([1.1, 1.1], [1.7, 0.7], color='#1976D2', lw=2.0)

ax.plot([2.1, 2.7], [1.7, 1.7], color='#D84315', lw=2.0)
ax.plot([2.1, 2.7], [0.7, 0.7], color='#D84315', lw=2.0)
ax.plot([2.7, 2.7], [1.7, 0.7], color='#D84315', lw=2.0)

# Cut edge indicator
ax.annotate("", xy=(2.1, 1.7), xytext=(1.1, 1.7),
            arrowprops=dict(arrowstyle="<->", color='red', lw=1.2, ls=':'))
ax.text(1.6, 1.82, "Cut Edge", fontsize=6.5, color='red', ha='center', fontweight='bold')

# Nodes in P1
for (x, y), lbl in zip([(0.5, 1.7), (1.1, 1.7), (0.8, 0.7), (1.1, 0.7)], ['A', 'B', 'D', 'E']):
    is_bd = lbl in ['B', 'E']
    c = '#EF5350' if is_bd else '#42A5F5'
    circle = plt.Circle((x, y), 0.14, color=c, ec='black', lw=0.8, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, lbl, color='white', fontweight='bold', fontsize=7, ha='center', va='center', zorder=3)

# Nodes in P2
for (x, y), lbl in zip([(2.1, 1.7), (2.7, 1.7), (2.1, 0.7), (2.7, 0.7)], ['C', 'X', 'F', 'Y']):
    is_bd = lbl in ['C', 'F']
    c = '#EF5350' if is_bd else '#FFA726'
    circle = plt.Circle((x, y), 0.14, color=c, ec='black', lw=0.8, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, lbl, color='white', fontweight='bold', fontsize=7, ha='center', va='center', zorder=3)

ax.text(1.6, -0.15, "Red Nodes: Boundary Cut Nodes", fontsize=6.5, ha='center', color='#C62828', fontweight='bold')


# (c) Boundary EigenPrompt Bridging
ax = axes[2]
ax.set_xlim(-0.2, 3.2)
ax.set_ylim(-0.2, 2.5)
ax.axis('off')
ax.set_title("(c) Boundary EigenPrompt", fontsize=8.5, fontweight='bold', pad=4)

# Node with prompt injection
node_pos = (1.5, 1.1)
circle_main = plt.Circle(node_pos, 0.22, color='#D32F2F', ec='black', lw=1.2, zorder=2)
ax.add_patch(circle_main)
ax.text(node_pos[0], node_pos[1], r"$v_{bd}$", color='white', fontweight='bold', fontsize=8, ha='center', va='center', zorder=3)

# Feature box
ax.text(0.3, 1.9, r"Raw Feat $\mathbf{x}_v$", fontsize=7.5, color='#1565C0', ha='center', fontweight='bold')
ax.annotate("", xy=(1.35, 1.25), xytext=(0.3, 1.75),
            arrowprops=dict(arrowstyle="->", color='#1565C0', lw=1.5))

# Prompt box
ax.text(2.7, 1.9, r"Prompt $\mathbf{p}(v)$", fontsize=7.5, color='#2E7D32', ha='center', fontweight='bold')
ax.text(2.7, 1.6, r"$\gamma_v (\mathbf{x}_v \odot \mathbf{W}_p)$", fontsize=6.5, color='#2E7D32', ha='center')
ax.annotate("", xy=(1.65, 1.25), xytext=(2.7, 1.45),
            arrowprops=dict(arrowstyle="->", color='#2E7D32', lw=1.5))

# Output representation
ax.annotate("", xy=(1.5, 0.2), xytext=(1.5, 0.85),
            arrowprops=dict(arrowstyle="->", color='#37474F', lw=1.5))
ax.text(1.5, 0.05, r"$\mathbf{\tilde{h}}_v = \mathbf{x}_v + \alpha \mathbf{p}(v)$", fontsize=8.0, color='#212121', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('./fig_approach.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 generated successfully: ./fig_approach.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Empirical Effect of Boundary EigenPrompt (Vector PDF)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 1.8), dpi=300)

# (a) Generalization Accuracy & F1 Boost
datasets = ['Arxiv', 'Reddit', 'Yelp (Acc)', 'Yelp (F1)', 'IGB', 'Products']
without_p = [55.09, 94.30, 92.03, 38.22, 75.66, 78.90]
with_p    = [66.78, 94.26, 92.99, 48.15, 75.71, 78.59]

x = np.arange(len(datasets))
width = 0.35

rects1 = ax1.bar(x - width/2, without_p, width, label='w/o Prompt', color='#90CAF9', edgecolor='#1565C0', lw=0.8)
rects2 = ax1.bar(x + width/2, with_p, width, label='w/ EigenPrompt', color='#1565C0', edgecolor='#0D47A1', lw=0.8)

ax1.set_ylabel('Score (%)', fontsize=8)
ax1.set_title('(a) Accuracy / F1: Effect of Boundary Prompt', fontsize=8.5, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=7, rotation=15)
ax1.set_ylim(20, 105)
ax1.legend(loc='lower right', fontsize=7, frameon=True)
ax1.grid(axis='y', linestyle=':', alpha=0.5)

# Add highlight callouts for Arxiv and Yelp F1
ax1.annotate('+11.7%', xy=(0 + width/2, 66.78), xytext=(0, 78),
             ha='center', fontsize=6.5, fontweight='bold', color='#D32F2F',
             arrowprops=dict(arrowstyle="->", color='#D32F2F', lw=0.8))

ax1.annotate('+9.9%', xy=(3 + width/2, 48.15), xytext=(3, 58),
             ha='center', fontsize=6.5, fontweight='bold', color='#D32F2F',
             arrowprops=dict(arrowstyle="->", color='#D32F2F', lw=0.8))

# (b) Peak GPU VRAM Footprint vs Default METIS (MB)
vram_datasets = ['Arxiv', 'Reddit', 'Yelp', 'IGB', 'Products']
dgl_vram   = [114.8, 956.4, 517.9, 828.5, 1788.7]
our_vram   = [185.4, 1478.2, 426.8, 552.9, 635.9]

x2 = np.arange(len(vram_datasets))
r1 = ax2.bar(x2 - width/2, dgl_vram, width, label='Default METIS', color='#FFCC80', edgecolor='#E65100', lw=0.8)
r2 = ax2.bar(x2 + width/2, our_vram, width, label='Our Work (K*)', color='#E65100', edgecolor='#BF360C', lw=0.8)

ax2.set_ylabel('Peak VRAM (MB)', fontsize=8)
ax2.set_title('(b) Peak GPU Memory Footprint', fontsize=8.5, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(vram_datasets, fontsize=7.5)
ax2.legend(loc='upper left', fontsize=7, frameon=True)
ax2.grid(axis='y', linestyle=':', alpha=0.5)

# Highlight Products VRAM saving
ax2.annotate('-64.4%\n(1.15 GB Saved)', xy=(4 + width/2, 635.9), xytext=(3.5, 1250),
             ha='center', fontsize=6.5, fontweight='bold', color='#2E7D32',
             arrowprops=dict(arrowstyle="->", color='#2E7D32', lw=0.8))

plt.tight_layout()
plt.savefig('./fig_ablation.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 generated successfully: ./fig_ablation.pdf")
