import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_comparison_charts(output_dir="./", save_name="comparison_analysis"):
    """
    绘制成功率和双曲性对比图
    
    Args:
        output_dir: 输出目录路径
        save_name: 保存文件名前缀
    """
    
    # 数据定义
    methods = ['L1-Regression', 'Flow Matching', 'VAE']
    success_rates = [52.2, 12.0, 3.8]  # 百分比
    hyperbolicities = [0.2458119229565531, 0.3213199705724935, 0.3231994657800171]
    original_hyperbolicity = 0.28460239832796214
    
    # 设置图形样式
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 颜色设置
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    original_color = '#8B5A2B'
    
    # 图1: 成功率柱状图
    bars1 = ax1.bar(methods, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_title('Success Rate Comparison\n(Step: 160,000)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Methods', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(success_rates) * 1.15)
    
    # 添加数值标签
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 网格和美化
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # 图2: 双曲性柱状图
    # 包含原始状态的所有数据
    all_methods = ['Original State'] + methods
    all_hyperbolicities = [original_hyperbolicity] + hyperbolicities
    all_colors = [original_color] + colors
    
    bars2 = ax2.bar(all_methods, all_hyperbolicities, color=all_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    ax2.set_title('Hyperbolicity Comparison\n(Step: 160,000)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Hyperbolicity', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Methods', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(all_hyperbolicities) * 1.15)
    
    # 添加数值标签
    for bar, hyp in zip(bars2, all_hyperbolicities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{hyp:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 添加基准线（原始状态）
    ax2.axhline(y=original_hyperbolicity, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(len(all_methods)-0.5, original_hyperbolicity + 0.02, 
             f'Baseline: {original_hyperbolicity:.3f}', 
             ha='center', va='bottom', color='red', fontweight='bold', fontsize=9)
    
    # 网格和美化
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # 旋转x轴标签以避免重叠
    ax2.tick_params(axis='x', rotation=15)
    
    # 整体布局调整
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    # 保存图片
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    save_path = output_path / f"{save_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图表已保存至: {save_path}")
    
    # 显示图片
    plt.show()
    
    # 打印数据摘要
    print("\n=== 数据摘要 ===")
    print(f"训练步数: 160,000")
    print("\n成功率排名:")
    for i, (method, rate) in enumerate(zip(methods, success_rates), 1):
        print(f"{i}. {method}: {rate}%")
    
    print(f"\n双曲性基准 (Original State): {original_hyperbolicity:.6f}")
    print("双曲性变化:")
    for method, hyp in zip(methods, hyperbolicities):
        change = hyp - original_hyperbolicity
        direction = "↑" if change > 0 else "↓"
        print(f"• {method}: {hyp:.6f} ({direction}{abs(change):.6f})")

def plot_separate_charts(output_dir="./"):
    """
    绘制两张独立的图表
    
    Args:
        output_dir: 输出目录路径
    """
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 数据定义
    methods = ['L1-Regression', 'Flow Matching', 'VAE']
    success_rates = [52.2, 12.0, 3.8]
    hyperbolicities = [0.2458119229565531, 0.3213199705724935, 0.3231994657800171]
    original_hyperbolicity = 0.28460239832796214
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 图1: 成功率
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.title('Success Rate Comparison (Step: 160,000)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Methods', fontsize=14, fontweight='bold')
    plt.ylim(0, max(success_rates) * 1.2)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{rate}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 保存成功率图
    success_path = output_path / "success_rate_comparison.png"
    plt.savefig(success_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"成功率图表已保存至: {success_path}")
    plt.show()
    plt.close()  # 关闭当前图形
    
    # 图2: 双曲性
    plt.figure(figsize=(10, 6))
    all_methods = ['Original State'] + methods
    all_hyperbolicities = [original_hyperbolicity] + hyperbolicities
    all_colors = ['#8B5A2B'] + colors
    
    bars = plt.bar(all_methods, all_hyperbolicities, color=all_colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    plt.title('Hyperbolicity Comparison (Step: 160,000)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Hyperbolicity', fontsize=14, fontweight='bold')
    plt.xlabel('Methods', fontsize=14, fontweight='bold')
    plt.ylim(0, max(all_hyperbolicities) * 1.15)
    
    for bar, hyp in zip(bars, all_hyperbolicities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{hyp:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 基准线
    plt.axhline(y=original_hyperbolicity, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(len(all_methods)-0.5, original_hyperbolicity + 0.02, 
             f'Baseline: {original_hyperbolicity:.3f}', 
             ha='center', va='bottom', color='red', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # 保存双曲性图
    hyp_path = output_path / "hyperbolicity_comparison.png"
    plt.savefig(hyp_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"双曲性图表已保存至: {hyp_path}")
    plt.show()
    plt.close()  # 关闭当前图形

if __name__ == "__main__":
    # 选择绘图模式
    mode = input("选择绘图模式 (1: 双子图, 2: 独立图表): ").strip()
    
    if mode == "1":
        plot_comparison_charts(output_dir="./charts", save_name="step_160k_comparison")
    elif mode == "2":
        plot_separate_charts(output_dir="./charts")
    else:
        print("无效选择，默认绘制双子图")
        plot_comparison_charts(output_dir="./charts", save_name="step_160k_comparison")