import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

def generate_feature_importance(model, feature_names):
    """Generate a horizontal bar chart showing feature importance."""
    importance = model.feature_importances_
    
    sorted_idx = np.argsort(importance)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_features)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_features, sorted_importance, color=colors, edgecolor='white', linewidth=0.7)
    
    for bar, val in zip(bars, sorted_importance):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9, color='#333')
    
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(str(STATIC_DIR / "feature_importance.png"), dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()


def generate_radar_chart(input_data, feature_names):
    """Generate a radar chart showing the input values."""
    labels = ['Cost', 'Time', 'Resource', 'Risk', 'Environ.', 'Deviation', 'Stakeholder', 'Complexity']
    
    max_vals = [1000000, 365, 10, 10, 10, 50, 10, 3]
    normalized = [min(v / m, 1) for v, m in zip(input_data, max_vals)]
    
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    normalized += normalized[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    ax.plot(angles, normalized, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, normalized, alpha=0.25, color='#667eea')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, fontweight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8, color='gray')
    
    ax.set_title('Input Parameters Overview', size=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(str(STATIC_DIR / "radar_chart.png"), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def generate_gauge_chart(risk_score, result):
    """Generate a gauge chart showing overall risk/feasibility level."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    theta = np.linspace(0, np.pi, 100)
    
    colors_sections = ['#eb3349', '#f5af19', '#11998e']
    for i, color in enumerate(colors_sections):
        start = i * np.pi / 3
        end = (i + 1) * np.pi / 3
        theta_section = np.linspace(start, end, 30)
        for j, t in enumerate(theta_section[:-1]):
            ax.fill_between([theta_section[j], theta_section[j+1]], [0.6, 0.6], [1, 1], 
                          color=color, alpha=0.7 + 0.1 * (j % 2))
    
    normalized_score = min(max(risk_score / 10, 0), 1)
    
    needle_angle = np.pi * (1 - normalized_score)
    ax.annotate('', xy=(needle_angle, 0.9), xytext=(np.pi/2, 0),
                arrowprops=dict(arrowstyle='->', color='#333', lw=3))
    
    circle = plt.Circle((np.pi/2, 0), 0.08, color='#333', transform=ax.transData)
    
    ax.text(0, 0.45, 'High\nRisk', ha='center', va='center', fontsize=10, color='#eb3349', fontweight='bold')
    ax.text(np.pi/2, 0.45, 'Medium', ha='center', va='center', fontsize=10, color='#f5af19', fontweight='bold')
    ax.text(np.pi, 0.45, 'Low\nRisk', ha='center', va='center', fontsize=10, color='#11998e', fontweight='bold')
    
    result_colors = {'Feasible': '#11998e', 'Not Feasible': '#eb3349', 'Uncertain': '#f5af19'}
    ax.text(np.pi/2, -0.15, f'{result}', ha='center', va='center', 
            fontsize=16, fontweight='bold', color=result_colors.get(result, '#333'))
    
    ax.set_xlim(-0.1, np.pi + 0.1)
    ax.set_ylim(-0.3, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Risk Assessment Gauge', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(str(STATIC_DIR / "gauge_chart.png"), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def generate_distribution_chart(input_data, feature_names):
    """Generate a bar chart showing the distribution of input scores."""
    labels = ['Cost\n(norm)', 'Time\n(norm)', 'Resource', 'Risk', 'Environ.', 'Deviation\n(norm)', 'Stakeholder', 'Complexity']
    
    max_vals = [100000, 100, 10, 10, 10, 20, 10, 3]
    normalized = [min(v / m * 10, 10) for v, m in zip(input_data, max_vals)]
    
    colors = []
    risk_indices = [3, 5, 7]
    for i, val in enumerate(normalized):
        if i in risk_indices:
            colors.append(plt.cm.RdYlGn_r(val / 10))
        else:
            colors.append(plt.cm.RdYlGn(val / 10))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, normalized, color=colors, edgecolor='white', linewidth=1.5)
    
    for bar, val, orig in zip(bars, normalized, input_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                f'{orig:.1f}', ha='center', va='bottom', fontsize=8, color='#333')
    
    ax.set_ylabel('Normalized Score (0-10)', fontsize=11, fontweight='bold')
    ax.set_title('Input Score Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Average')
    
    plt.tight_layout()
    plt.savefig(str(STATIC_DIR / "distribution_chart.png"), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def generate_all_visualizations(model, feature_names, input_data, result):
    """Generate all visualization charts."""
    generate_feature_importance(model, feature_names)
    generate_radar_chart(input_data, feature_names)
    generate_gauge_chart(input_data[3], result)
    generate_distribution_chart(input_data, feature_names)
