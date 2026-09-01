import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt错误
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

# 可调字体大小
Y_LABEL_FONT_SIZE = 20
X_TICK_FONT_SIZE = 16
Y_TICK_FONT_SIZE = 15

# 数据路径（自动获取脚本所在目录的父目录）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]

data_paths = {
    'PBM': project_root / 'outputs' / 'PBM测试集结果.csv',
    'SnowNN': project_root / 'outputs' / 'ablation_combined' / 'ablation_snowNN' / 'station_performance_real_runoff.csv',
    'ETNN': project_root / 'outputs' / 'ablation_combined' / 'ablation_etNN' / 'station_performance_real_runoff.csv',
    'RunoffNN': project_root / 'outputs' / 'ablation_combined' / 'ablation_runoffNN' / 'station_performance_real_runoff.csv',
    'DrainageNN': project_root / 'outputs' / 'ablation_combined' / 'ablation_drainageNN' / 'station_performance_real_runoff.csv',
    'HydroMoE': project_root / 'outputs' / 'MoE' / 'station_performance_real_runoff.csv'
}

# 读取数据并提取R²值
r2_data = []
model_names = []

for model, path in data_paths.items():
    df = pd.read_csv(str(path))
    
    # 检查字段名（PBM用r2小写，其他用R2）
    if 'r2' in df.columns:
        r2_values = df['r2'].dropna()
    elif 'R2' in df.columns:
        r2_values = df['R2'].dropna()
    else:
        print(f"Warning: R2 field not found in {model}")
        continue
    
    r2_data.extend(r2_values)
    model_names.extend([model] * len(r2_values))

# 创建DataFrame
plot_df = pd.DataFrame({
    'Model': model_names,
    'R2': r2_data
})

# 设置模型顺序
model_order = ['PBM', 'SnowNN', 'ETNN', 'RunoffNN', 'DrainageNN', 'HydroMoE']
plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=model_order, ordered=True)

# 设置颜色（参考原图的配色）
colors = ['#6495ED', '#87CEEB', '#B0C4DE', '#FFB6C1', '#FFA07A', '#CD853F']
color_dict = dict(zip(model_order, colors))


def create_boxplot(font_family: str, output_filename: str) -> None:
    rc_update = {
        'font.family': font_family,
        'font.size': 15
    }

    with plt.rc_context(rc_update):
        fig, ax = plt.subplots(figsize=(12, 8))
        positions = np.arange(len(model_order))
        r2_values = [plot_df[plot_df['Model'] == model]['R2'].values for model in model_order]

        bp = ax.boxplot(r2_values,
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='o', markerfacecolor='red', markersize=8,
                                      markeredgecolor='black', label='Mean', zorder=10),
                        medianprops=dict(color='black', linewidth=2, linestyle='--'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, model in enumerate(model_order):
            model_data = r2_values[i]
            if len(model_data) > 1:
                kde = gaussian_kde(model_data)
                y_vals = np.linspace(model_data.min(), model_data.max(), 200)
                density = kde(y_vals)
                density_scaled = density / density.max() * 0.25
                ax.plot(i + density_scaled, y_vals, color='black', linestyle='--', linewidth=1)

            x = np.random.normal(i, 0.04, size=len(model_data))
            ax.scatter(x, model_data, alpha=0.4, s=30, color=color_dict[model],
                       edgecolors='black', linewidth=0.5, zorder=2)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=11)
        ax.axhline(y=0.6, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=11)
        ax.set_xticks(positions)
        ax.set_xticklabels(model_order, fontsize=X_TICK_FONT_SIZE, fontweight='bold')
        ax.set_ylabel('R²', fontsize=Y_LABEL_FONT_SIZE, fontweight='bold')
        ax.tick_params(axis='y', labelsize=Y_TICK_FONT_SIZE)
        ax.set_ylim(-0.5, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        mean_legend = plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='red', markersize=8,
                                 markeredgecolor='black', label='Mean')
        median_legend = plt.Line2D([0], [0], color='black', linewidth=2,
                                   linestyle='--', label='Median')
        ax.legend(handles=[mean_legend, median_legend], loc='upper left', fontsize=11, framealpha=0.9)

        plt.tight_layout()

        output_path = script_dir / output_filename
        plt.savefig(output_path, dpi=600, format='jpeg', bbox_inches='tight',
                    pil_kwargs={'quality': 95})
        plt.close(fig)

    print(f"✅ 图片已保存到: {script_dir / output_filename}")


create_boxplot('Times New Roman', 'R2_boxplot_comparison.jpg')
create_boxplot('Arial', 'R2_boxplot_comparison_Arial.jpg')

# 显示统计信息（在控制台）
print("\n" + "="*60)
print("各模型R²统计信息:")
print("="*60)
for model in model_order:
    model_r2 = plot_df[plot_df['Model'] == model]['R2']
    print(f"\n{model}:")
    print(f"  样本数: {len(model_r2)}")
    print(f"  均值: {model_r2.mean():.4f}")
    print(f"  中位数: {model_r2.median():.4f}")
    print(f"  标准差: {model_r2.std():.4f}")
    print(f"  最小值: {model_r2.min():.4f}")
    print(f"  最大值: {model_r2.max():.4f}")
    print(f"  25%分位数: {model_r2.quantile(0.25):.4f}")
    print(f"  75%分位数: {model_r2.quantile(0.75):.4f}")

print("\n" + "="*60)
print("✅ 绘图完成！")
# plt.show()  # 使用Agg后端，不需要显示窗口
