"""
Generate the 4 key figures for README display.
Run from repo root: python scripts/generate_figures.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math, warnings
warnings.filterwarnings('ignore')

from utils import setup_chinese_font, show_image_grid
from lucent.modelzoo import inceptionv1
from lucent.optvis import render, objectives

# ── Setup ──
setup_chinese_font()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STEPS = 512
OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'Device: {device} | Steps: {STEPS}')
print()

# ── Load model ──
model = inceptionv1(pretrained=True).to(device).eval()
print('InceptionV1 loaded')


def viz(layer, channel, steps=STEPS):
    obj = objectives.channel(layer, channel)
    result = render.render_vis(model, obj, thresholds=(steps,), show_inline=False, progress=False)
    return result[0][0]


def denorm(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ═══════════════════════════════════════════════════
# Figure 1: Feature Visualization Grid
# ═══════════════════════════════════════════════════
print('Generating figure 1/4: feature_viz_grid.png')
neurons = [
    ('conv2d0',  0,  'Early: color/edge'),
    ('conv2d1',  5,  'Early: color contrast'),
    ('mixed3a',  0,  'Mid: edge detector'),
    ('mixed3b',  6,  'High/low freq'),
    ('mixed3b', 379, 'Curve detector'),
    ('mixed3b', 425, 'Pooling branch'),
    ('mixed4a',  22, 'Mid: texture'),
    ('mixed4e', 254, 'Deep: complex'),
]
cache = {}
for layer, ch, desc in neurons:
    print(f'  {layer}:{ch} ({desc})', end=' ... ', flush=True)
    img = viz(layer, ch)
    cache[(layer, ch)] = (img, desc)
    print('done')

show_image_grid(
    [cache[(l, c)][0] for l, c, _ in neurons],
    [f'{l}:{c}\n{d}' for l, c, d in neurons],
    suptitle='Feature Visualization (Activation Maximization)\nShallow \u2192 Deep: Simple \u2192 Complex',
    save_path=f'{OUT_DIR}/feature_viz_grid.png',
)
print('  => saved feature_viz_grid.png')

# ═══════════════════════════════════════════════════
# Figure 2: Polar Tuning Plot
# ═══════════════════════════════════════════════════
print('Generating figure 2/4: polar_tuning.png')


def make_arc(angle, size=224, radius=60, width=5):
    img = Image.new('RGB', (size, size), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.arc(bbox, start=angle - 60, end=angle + 60, fill=(255, 255, 255), width=width)
    transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform(img).unsqueeze(0)


def measure_tuning(layer, channel, n_angles=36):
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    activations = []
    store = {}

    def hook(m, inp, out):
        store['v'] = out[0, channel].mean().item()

    h = dict(model.named_modules())[layer].register_forward_hook(hook)
    with torch.no_grad():
        for a in angles:
            model(make_arc(a).to(device))
            activations.append(store['v'])
    h.remove()
    return angles, np.array(activations)


test_neurons = [('mixed3b', 379), ('mixed3b', 425)]
tuning_results = []
for layer, ch in test_neurons:
    angles, acts = measure_tuning(layer, ch)
    tuning_results.append((angles, acts, f'{layer}:{ch}'))
    print(f'  {layer}:{ch} preferred direction = {angles[acts.argmax()]:.0f} deg')

fig, axes = plt.subplots(1, len(tuning_results), figsize=(5 * len(tuning_results), 4.5),
                         subplot_kw={'projection': 'polar'})
if len(tuning_results) == 1:
    axes = [axes]
for ax, (angles, acts, label) in zip(axes, tuning_results):
    norm_acts = (acts - acts.min()) / (acts.max() - acts.min() + 1e-8)
    rads = np.append(np.deg2rad(angles), np.deg2rad(angles[0]))
    vals = np.append(norm_acts, norm_acts[0])
    ax.plot(rads, vals, linewidth=2, color='steelblue')
    ax.fill(rads, vals, alpha=0.25, color='steelblue')
    ax.set_title(label, pad=15)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
plt.suptitle('Orientation Tuning (Polar Plot)\nPeak direction = preferred orientation', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/polar_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print('  => saved polar_tuning.png')

# ═══════════════════════════════════════════════════
# Figure 3: Circuit Diagram
# ═══════════════════════════════════════════════════
print('Generating figure 3/4: circuit_diagram.png')

target_ch = 379
branch_idx = target_ch - 320
w_bottleneck = model.mixed3b_5x5_bottleneck_pre_relu_conv.weight.data.cpu()
w_5x5 = model.mixed3b_5x5_pre_relu_conv.weight.data.cpu()
target_filter = w_5x5[branch_idx]
bottleneck_importance = target_filter.abs().sum((1, 2))
bottleneck_weight = w_bottleneck.squeeze()
mixed3a_importance = (bottleneck_importance[:, None] * bottleneck_weight.abs()).sum(0)
top_k = 4
top_indices = mixed3a_importance.topk(top_k).indices.tolist()

upstream_imgs = []
for ch in top_indices:
    print(f'  mixed3a:{ch}', end=' ... ', flush=True)
    img = viz('mixed3a', ch)
    upstream_imgs.append((ch, img))
    print('done')

curve_img, _ = cache[('mixed3b', target_ch)]
import matplotlib.gridspec as gridspec
width_ratios = [1]*top_k + [0.4] + [1]  # narrow arrow column
fig = plt.figure(figsize=((top_k + 1.4) * 3.2, 3.5))
gs = gridspec.GridSpec(1, top_k + 2, width_ratios=width_ratios, wspace=0.15)
for i, (ch, img) in enumerate(upstream_imgs):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Upstream #{i+1}\nmixed3a:{ch}\n(edge detector)', fontsize=8.5)
arrow_ax = fig.add_subplot(gs[0, top_k])
arrow_ax.text(0.5, 0.5, 'Weighted\nSum\n\u279c', ha='center', va='center',
              fontsize=13, color='gray', transform=arrow_ax.transAxes)
arrow_ax.axis('off')
target_ax = fig.add_subplot(gs[0, top_k + 1])
target_ax.imshow(curve_img)
target_ax.axis('off')
target_ax.set_title(f'Target: Curve Detector\nmixed3b:{target_ch}', fontsize=8.5)
plt.suptitle('Circuit: Edge Detectors (mixed3a) \u2192 Curve Detector (mixed3b)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/circuit_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print('  => saved circuit_diagram.png')

# ═══════════════════════════════════════════════════
# Figure 4: Universality Comparison
# ═══════════════════════════════════════════════════
print('Generating figure 4/4: universality_comparison.png')

resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(device).eval()
print('  ResNet-18 loaded')

resnet_neurons = [
    ('layer1', 0, 'ResNet layer1:0'),
    ('layer1', 8, 'ResNet layer1:8'),
    ('layer1', 16, 'ResNet layer1:16'),
    ('layer2', 0, 'ResNet layer2:0'),
    ('layer3', 0, 'ResNet layer3:0'),
]
resnet_results = []
for layer, ch, label in resnet_neurons:
    print(f'  {label}', end=' ... ', flush=True)
    obj = objectives.channel(layer, ch)
    result = render.render_vis(resnet, obj, thresholds=(STEPS,), show_inline=False, progress=False)
    resnet_results.append((result[0][0], label))
    print('done')

inception_compare = [
    (cache[('conv2d0', 0)][0], 'InceptionV1\nconv2d0:0\n(early)'),
    (cache[('mixed3a', 0)][0], 'InceptionV1\nmixed3a:0\n(edge detector)'),
    (cache[('mixed3b', 379)][0], 'InceptionV1\nmixed3b:379\n(curve detector)'),
    (cache[('mixed3b', 6)][0], 'InceptionV1\nmixed3b:6\n(high/low freq)'),
]

n_cols = max(len(inception_compare), len(resnet_results))
fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6.5))
for i, (img, label) in enumerate(inception_compare):
    axes[0, i].imshow(img)
    axes[0, i].axis('off')
    axes[0, i].set_title(label, fontsize=8)
for j in range(len(inception_compare), n_cols):
    axes[0, j].axis('off')
for i, (img, label) in enumerate(resnet_results):
    axes[1, i].imshow(img)
    axes[1, i].axis('off')
    axes[1, i].set_title(label, fontsize=8)
for j in range(len(resnet_results), n_cols):
    axes[1, j].axis('off')
plt.suptitle('Universality: InceptionV1 (top) vs ResNet-18 (bottom)\nDifferent architectures, independent training, similar early features', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/universality_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  => saved universality_comparison.png')

print()
print('All 4 figures saved to figures/')
