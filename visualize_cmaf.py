import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'Cross-Modal Attention Fusion (CMAF) Architecture', 
        fontsize=20, fontweight='bold', ha='center')

# Input layer
# Video frames
video_box = FancyBboxPatch((0.5, 9.5), 2, 1, boxstyle="round,pad=0.1", 
                          facecolor='lightblue', edgecolor='navy', linewidth=2)
ax.add_patch(video_box)
ax.text(1.5, 10, 'Video Frames\n(Input)', ha='center', va='center', fontweight='bold')

# BLIP explanations
text_box = FancyBboxPatch((7, 9.5), 2, 1, boxstyle="round,pad=0.1", 
                         facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
ax.add_patch(text_box)
ax.text(8, 10, 'BLIP Explanations\n(Text Input)', ha='center', va='center', fontweight='bold')

# Feature extraction layer
# CLIP features
clip_box = FancyBboxPatch((0.5, 7.5), 2, 1, boxstyle="round,pad=0.1", 
                         facecolor='skyblue', edgecolor='blue', linewidth=2)
ax.add_patch(clip_box)
ax.text(1.5, 8, 'CLIP Features\n(512D)', ha='center', va='center', fontweight='bold')

# DistilBERT features
bert_box = FancyBboxPatch((7, 7.5), 2, 1, boxstyle="round,pad=0.1", 
                         facecolor='lightcoral', edgecolor='red', linewidth=2)
ax.add_patch(bert_box)
ax.text(8, 8, 'DistilBERT\n(768D)', ha='center', va='center', fontweight='bold')

# Projection layer
visual_proj = FancyBboxPatch((1, 5.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                            facecolor='cyan', edgecolor='darkcyan', linewidth=2)
ax.add_patch(visual_proj)
ax.text(1.75, 5.9, 'Visual\nProjection\n(512D)', ha='center', va='center', fontsize=10)

text_proj = FancyBboxPatch((7.5, 5.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='pink', edgecolor='hotpink', linewidth=2)
ax.add_patch(text_proj)
ax.text(8.25, 5.9, 'Text\nProjection\n(512D)', ha='center', va='center', fontsize=10)

# Cross-attention mechanism (central)
attention_box = FancyBboxPatch((3.5, 4), 3, 2, boxstyle="round,pad=0.2", 
                              facecolor='gold', edgecolor='orange', linewidth=3)
ax.add_patch(attention_box)
ax.text(5, 5.5, 'Cross-Modal Attention Fusion', ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Bidirectional arrows in attention box
ax.text(5, 5, 'Visual → Text\n(8-head attention)', ha='center', va='center', fontsize=10)
ax.text(5, 4.5, 'Text → Visual\n(8-head attention)', ha='center', va='center', fontsize=10)

# Fusion gate
gate_box = FancyBboxPatch((4.2, 2.5), 1.6, 0.8, boxstyle="round,pad=0.1", 
                         facecolor='violet', edgecolor='purple', linewidth=2)
ax.add_patch(gate_box)
ax.text(5, 2.9, 'Fusion Gate\n(Learnable)', ha='center', va='center', fontweight='bold')

# Enhanced features
enhanced_box = FancyBboxPatch((3.5, 1), 3, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='lightseagreen', edgecolor='seagreen', linewidth=2)
ax.add_patch(enhanced_box)
ax.text(5, 1.4, 'Enhanced Features (512D)', ha='center', va='center', fontweight='bold')

# Anomaly detection head
anomaly_box = FancyBboxPatch((4, 0), 2, 0.6, boxstyle="round,pad=0.1", 
                            facecolor='salmon', edgecolor='darkred', linewidth=2)
ax.add_patch(anomaly_box)
ax.text(5, 0.3, 'Anomaly Score', ha='center', va='center', fontweight='bold')

# Arrows
# Input to feature extraction
ax.annotate('', xy=(1.5, 9.4), xytext=(1.5, 8.6), 
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.annotate('', xy=(8, 9.4), xytext=(8, 8.6), 
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))

# Feature extraction to projection
ax.annotate('', xy=(1.75, 7.4), xytext=(1.75, 6.4), 
            arrowprops=dict(arrowstyle='->', lw=2, color='cyan'))
ax.annotate('', xy=(8.25, 7.4), xytext=(8.25, 6.4), 
            arrowprops=dict(arrowstyle='->', lw=2, color='pink'))

# Projection to attention
ax.annotate('', xy=(3.4, 5), xytext=(2.6, 5.9), 
            arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
ax.annotate('', xy=(6.6, 5), xytext=(7.4, 5.9), 
            arrowprops=dict(arrowstyle='->', lw=2, color='orange'))

# Attention to fusion
ax.annotate('', xy=(5, 3.9), xytext=(5, 3.4), 
            arrowprops=dict(arrowstyle='->', lw=3, color='purple'))

# Fusion to enhanced
ax.annotate('', xy=(5, 2.4), xytext=(5, 1.9), 
            arrowprops=dict(arrowstyle='->', lw=2, color='seagreen'))

# Enhanced to anomaly
ax.annotate('', xy=(5, 0.9), xytext=(5, 0.7), 
            arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))

# Add side annotations
ax.text(0.2, 6, 'Visual\nPathway', rotation=90, va='center', ha='center', 
        fontsize=12, fontweight='bold', color='blue')
ax.text(9.8, 6, 'Text\nPathway', rotation=90, va='center', ha='center', 
        fontsize=12, fontweight='bold', color='green')

# Add technical details box
details_box = FancyBboxPatch((0.2, 2.5), 2.8, 2, boxstyle="round,pad=0.1", 
                            facecolor='lightyellow', edgecolor='black', linewidth=1)
ax.add_patch(details_box)
ax.text(1.6, 3.8, 'Technical Details:', ha='center', fontweight='bold', fontsize=11)
ax.text(1.6, 3.5, '• Multi-head Attention (8 heads)', ha='center', fontsize=9)
ax.text(1.6, 3.25, '• Bidirectional Information Flow', ha='center', fontsize=9)
ax.text(1.6, 3, '• Learnable Fusion Gates', ha='center', fontsize=9)
ax.text(1.6, 2.75, '• Residual Connections', ha='center', fontsize=9)

# Add novelty box
novelty_box = FancyBboxPatch((6.8, 2.5), 2.8, 2, boxstyle="round,pad=0.1", 
                            facecolor='lavender', edgecolor='black', linewidth=1)
ax.add_patch(novelty_box)
ax.text(8.2, 3.8, 'Novel Contributions:', ha='center', fontweight='bold', fontsize=11)
ax.text(8.2, 3.5, '• First CLIP-BLIP Fusion', ha='center', fontsize=9)
ax.text(8.2, 3.25, '• Cross-Modal Attention Loss', ha='center', fontsize=9)
ax.text(8.2, 3, '• Interpretable Predictions', ha='center', fontsize=9)
ax.text(8.2, 2.75, '• Semantic Enhancement', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('CMAF_Architecture.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ CMAF Architecture diagram saved as 'CMAF_Architecture.png'")
