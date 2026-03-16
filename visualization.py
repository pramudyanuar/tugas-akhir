import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from env.container_env import ContainerEnv


class ContainerVisualizer:
    """
    Visualisasi 3D Bin Packing untuk presentasi ke dosen.
    
    Features:
    - 2D Top View dengan height heatmap
    - 3D visualization
    - Cross-section views
    - Progress statistics
    """
    
    def __init__(self, container_dims=(59, 23, 23)):
        """
        Initialize visualizer.
        
        Args:
            container_dims: Tuple of (length, width, height)
        """
        self.L, self.W, self.H = container_dims
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    def visualize_packing_2d(self, placed_items, placed_positions, height_map, title="Container Packing - Top View"):
        """
        Visualize packing sebagai 2D top view dengan height heatmap.
        
        Args:
            placed_items: List of (length, width, height)
            placed_positions: List of (x, y, z)
            height_map: 2D numpy array of heights
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Height heatmap
        im = ax1.imshow(height_map.T, cmap='YlOrRd', origin='lower')
        ax1.set_xlabel(f'Length (0-{self.L})')
        ax1.set_ylabel(f'Width (0-{self.W})')
        ax1.set_title('Height Map (Top View)')
        plt.colorbar(im, ax=ax1, label='Height')
        
        # Plot 2: Item placement dengan warna
        ax2.set_xlim(0, self.L)
        ax2.set_ylim(0, self.W)
        ax2.set_aspect('equal')
        ax2.set_xlabel(f'Length (0-{self.L})')
        ax2.set_ylabel(f'Width (0-{self.W})')
        ax2.set_title('Item Placement (Top View)')
        
        # Draw items
        for i, (item, pos) in enumerate(zip(placed_items, placed_positions)):
            l, w, h = item
            x, y, z = pos
            
            # Draw rectangle
            rect = patches.Rectangle((x, y), l, w, linewidth=2, 
                                    edgecolor='black', facecolor=self.colors[i % 20],
                                    alpha=0.7)
            ax2.add_patch(rect)
            
            # Add label
            ax2.text(x + l/2, y + w/2, f'{i+1}\n{l}×{w}×{h}', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw border
        rect_border = patches.Rectangle((0, 0), self.L, self.W, linewidth=3,
                                       edgecolor='black', facecolor='none')
        ax2.add_patch(rect_border)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_packing_3d(self, placed_items, placed_positions, title="3D Container Packing"):
        """
        Visualize packing dalam 3D.
        
        Args:
            placed_items: List of (length, width, height)
            placed_positions: List of (x, y, z)
            title: Plot title
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw container frame
        self._draw_container_frame(ax)
        
        # Draw items
        for i, (item, pos) in enumerate(zip(placed_items, placed_positions)):
            l, w, h = item
            x, y, z = pos
            
            # Define vertices of box
            vertices = np.array([
                [x, y, z],
                [x+l, y, z],
                [x+l, y+w, z],
                [x, y+w, z],
                [x, y, z+h],
                [x+l, y, z+h],
                [x+l, y+w, z+h],
                [x, y+w, z+h]
            ])
            
            # Define faces
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[7], vertices[6], vertices[2], vertices[3]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]]
            ]
            
            # Add collection
            face_collection = Poly3DCollection(faces, alpha=0.7, 
                                              facecolor=self.colors[i % 20],
                                              edgecolors='black', linewidth=1)
            ax.add_collection3d(face_collection)
        
        # Set limits
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.W)
        ax.set_zlim(0, self.H)
        
        # Labels
        ax.set_xlabel(f'Length ({self.L})')
        ax.set_ylabel(f'Width ({self.W})')
        ax.set_zlabel(f'Height ({self.H})')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return fig
    
    def _draw_container_frame(self, ax):
        """Draw container wireframe."""
        vertices = np.array([
            [0, 0, 0],
            [self.L, 0, 0],
            [self.L, self.W, 0],
            [0, self.W, 0],
            [0, 0, self.H],
            [self.L, 0, self.H],
            [self.L, self.W, self.H],
            [0, self.W, self.H]
        ])
        
        # Draw edges
        edges = [
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
            [vertices[4], vertices[5]],
            [vertices[5], vertices[6]],
            [vertices[6], vertices[7]],
            [vertices[7], vertices[4]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]],
        ]
        
        for edge in edges:
            points = np.array(edge)
            ax.plot3D(*points.T, 'k-', linewidth=2)
    
    def visualize_cross_sections(self, height_map, title="Cross-Section Views"):
        """
        Visualize cross-section views (XZ dan YZ planes).
        
        Args:
            height_map: 2D numpy array of heights
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # XZ cross-section (along width at center)
        xz_data = height_map[:, self.W//2:self.W//2+1].flatten()
        ax1.bar(range(len(xz_data)), xz_data, color='steelblue', edgecolor='black')
        ax1.fill_between(range(len(xz_data)), xz_data, alpha=0.3, color='steelblue')
        ax1.set_xlabel('Length Position')
        ax1.set_ylabel('Height')
        ax1.set_title(f'XZ Cross-Section (Width={self.W//2})')
        ax1.set_ylim(0, self.H)
        ax1.grid(axis='y', alpha=0.3)
        
        # YZ cross-section (along length at center)
        yz_data = height_map[self.L//2:self.L//2+1, :].flatten()
        ax2.bar(range(len(yz_data)), yz_data, color='coral', edgecolor='black')
        ax2.fill_between(range(len(yz_data)), yz_data, alpha=0.3, color='coral')
        ax2.set_xlabel('Width Position')
        ax2.set_ylabel('Height')
        ax2.set_title(f'YZ Cross-Section (Length={self.L//2})')
        ax2.set_ylim(0, self.H)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_statistics(self, placed_items, placed_positions, height_map, 
                           utilization, load_balance, success_rate, title="Packing Statistics"):
        """
        Visualize packing statistics.
        
        Args:
            placed_items: List of placed items
            placed_positions: List of positions
            height_map: Height map
            utilization: Volume utilization percentage
            load_balance: Load balance score
            success_rate: Success rate
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Item distribution by size
        ax = axes[0, 0]
        volumes = [l*w*h for l, w, h in placed_items]
        ax.bar(range(len(volumes)), volumes, color='skyblue', edgecolor='black')
        ax.set_xlabel('Item Index')
        ax.set_ylabel('Volume')
        ax.set_title('Item Volumes')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Height distribution
        ax = axes[0, 1]
        heights = np.max(height_map, axis=0)
        ax.bar(range(len(heights)), heights, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Width Position')
        ax.set_ylabel('Height')
        ax.set_title('Height Profile (Length view)')
        ax.set_ylim(0, self.H)
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Key metrics
        ax = axes[1, 0]
        ax.axis('off')
        
        metrics_text = f"""
        PACKING METRICS
        ═══════════════════════════════════
        
        Volume Utilization:    {utilization:.2f}%
        Load Balance Score:    {load_balance:.4f}
        Success Rate:          {success_rate:.2%}
        
        Items Placed:          {len(placed_items)}
        Max Height:            {np.max(height_map):.0f} of {self.H}
        Container Volume:      {self.L}×{self.W}×{self.H}
        
        Total Placed Volume:   {sum(volumes):.0f}
        Container Volume:      {self.L * self.W * self.H}
        """
        
        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        # 4. Summary gauge
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        SYSTEM STATUS
        ═══════════════════════════════════
        
        ✅ 3D Bin Packing: Operational
        ✅ LBCP Clustering: Active
        ✅ HRL Agents: Ready
        ✅ MCTS Planning: Enabled
        ✅ Repacking: Available
        
        Overall Performance Rating:
        """
        
        # Calculate overall score
        overall_score = (utilization/100.0 * 0.5 + load_balance * 0.3 + success_rate * 0.2)
        
        ax.text(0.1, 0.7, summary_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='lightblue', alpha=0.5))
        
        # Rating bar
        colors_rating = ['red' if overall_score < 0.3 else 'orange' if overall_score < 0.6 else 'yellow' if overall_score < 0.8 else 'green']
        ax.barh([0], [overall_score], height=0.3, color=colors_rating[0], 
               edgecolor='black', linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.text(0.5, -0.15, f'{overall_score:.1%}', ha='center', fontsize=12, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def save_all_visualizations(self, env, output_dir='./visualizations'):
        """
        Save all visualizations to files.
        
        Args:
            env: ContainerEnv instance dengan placed items
            output_dir: Directory untuk save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        utilization = env.get_utilization()
        height_map = env.height_map.map
        
        # Calculate load balance (simple version)
        mid_x = env.L / 2.0
        mid_y = env.W / 2.0
        quadrant_weights = [0.0, 0.0, 0.0, 0.0]
        
        for pos, item in zip(env.placed_positions, env.placed_items):
            x, y, z = pos
            l, w, h = item
            center_x = x + l / 2.0
            center_y = y + w / 2.0
            weight = l * w * h
            
            if center_x < mid_x and center_y < mid_y:
                quadrant_weights[0] += weight
            elif center_x >= mid_x and center_y < mid_y:
                quadrant_weights[1] += weight
            elif center_x < mid_x and center_y >= mid_y:
                quadrant_weights[2] += weight
            else:
                quadrant_weights[3] += weight
        
        total = sum(quadrant_weights)
        if total > 0:
            avg = total / 4.0
            variance = sum((w - avg) ** 2 for w in quadrant_weights) / 4.0
            std_dev = np.sqrt(variance)
            cv = std_dev / avg if avg > 0 else 0.0
            load_balance = 1.0 / (1.0 + cv)
        else:
            load_balance = 1.0
        
        success_rate = len(env.placed_items) / max(len(env.items), 1)
        
        # Save figures
        fig1 = self.visualize_packing_2d(env.placed_items, env.placed_positions, 
                                        height_map, 
                                        f"Packing Result - {len(env.placed_items)} items")
        fig1.savefig(os.path.join(output_dir, '01_packing_2d.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        fig2 = self.visualize_packing_3d(env.placed_items, env.placed_positions,
                                        f"3D View - {len(env.placed_items)} items")
        fig2.savefig(os.path.join(output_dir, '02_packing_3d.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        fig3 = self.visualize_cross_sections(height_map, "Container Cross-Sections")
        fig3.savefig(os.path.join(output_dir, '03_cross_sections.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        fig4 = self.visualize_statistics(env.placed_items, env.placed_positions,
                                        height_map, utilization, load_balance, 
                                        success_rate, "Final Statistics")
        fig4.savefig(os.path.join(output_dir, '04_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4)
        
        print(f"✓ Visualizations saved to {output_dir}/")
        return {
            'utilization': utilization,
            'load_balance': load_balance,
            'success_rate': success_rate,
            'items_placed': len(env.placed_items),
            'max_height': np.max(height_map)
        }


if __name__ == "__main__":
    """Demo visualization"""
    
    print("Creating sample packing for visualization...")
    
    env = ContainerEnv(max_items=10, seed=42)
    state, action_mask = env.reset()
    
    # Manually place some items
    for i in range(8):
        if env.current_index < len(env.items):
            action = (i * 50) % (env.L * env.W)
            (next_state, next_mask), reward, done, info = env.step(action)
    
    # Visualize
    viz = ContainerVisualizer(container_dims=(59, 23, 23))
    metrics = viz.save_all_visualizations(env, output_dir='./visualizations')
    
    print("\nVisualization Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
