import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.perfect_pack_generator import PerfectPackGenerator
from scripts.visualization import ContainerVisualizer


def build_height_map(length, width, items, positions):
    height_map = np.zeros((length, width), dtype=np.int32)
    for (l, w, h), (x, y, z) in zip(items, positions):
        top = z + h
        current = height_map[x:x + l, y:y + w]
        height_map[x:x + l, y:y + w] = np.maximum(current, top)
    return height_map


def main():
    parser = argparse.ArgumentParser(description='Visualize perfect pack dataset with ground-truth positions')
    parser.add_argument('--bin-width', type=int, default=23, help='Bin width (length dimension)')
    parser.add_argument('--bin-height', type=int, default=23, help='Bin height (width dimension)')
    parser.add_argument('--sigma', type=float, default=2.0, help='Gaussian sigma for dimension sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--size-bias', type=float, default=0.0, help='Bias to smaller sizes (>0) or larger (<0)')
    parser.add_argument('--mean-ratio', type=float, default=0.5, help='Gaussian mean ratio (0-1)')
    parser.add_argument('--layered', action='store_true', help='Use layered 3D perfect pack')
    parser.add_argument('--container-height', type=int, default=26, help='Container height for layered mode')
    parser.add_argument('--min-layer-height', type=int, default=2, help='Minimum layer thickness')
    parser.add_argument('--max-layer-height', type=int, default=6, help='Maximum layer thickness')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations/perfect_pack', help='Output directory')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle items and positions for display')

    args = parser.parse_args()

    generator = PerfectPackGenerator(
        bin_width=args.bin_width,
        bin_height=args.bin_height,
        sigma=args.sigma,
        seed=args.seed,
        size_bias=args.size_bias,
        mean_ratio=args.mean_ratio,
    )

    if args.layered:
        items, positions = generator.generate_layered_perfect_pack_with_positions(
            container_height=args.container_height,
            min_layer_height=args.min_layer_height,
            max_layer_height=args.max_layer_height,
            num_attempts=3,
            shuffle=args.shuffle,
        )
        container_height = args.container_height
    else:
        items, positions = generator.generate_perfect_pack_with_positions(
            num_attempts=3,
            shuffle=args.shuffle,
        )
        container_height = max(item[2] for item in items)

    if len(items) == 0:
        raise RuntimeError('Perfect pack generation failed to produce items')

    height_map = build_height_map(args.bin_width, args.bin_height, items, positions)

    os.makedirs(args.output_dir, exist_ok=True)

    visualizer = ContainerVisualizer(container_dims=(args.bin_width, args.bin_height, container_height))

    fig1 = visualizer.visualize_packing_2d(items, positions, height_map, title='Perfect Pack - Top View')
    fig1.savefig(os.path.join(args.output_dir, '01_perfect_pack_2d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2 = visualizer.visualize_packing_3d(items, positions, title='Perfect Pack - 3D View')
    fig2.savefig(os.path.join(args.output_dir, '02_perfect_pack_3d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3 = visualizer.visualize_cross_sections(height_map, title='Perfect Pack - Cross Sections')
    fig3.savefig(os.path.join(args.output_dir, '03_perfect_pack_cross_sections.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)

    print('Perfect pack visualizations saved to:', args.output_dir)


if __name__ == '__main__':
    main()
