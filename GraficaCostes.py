import matplotlib.pyplot as plt
import numpy as np

labels = ['Model 1 (3 Layers)', 'Model 2 (5 Layers)', 'Model 3 (4 Layers)']
real_costs = [497.00, 480.20, 481.18]
theoretical_costs = [184320, 307200, 245760]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, theoretical_costs, width, label='Sequential Execution (Estimated)', color='#d62728')
rects2 = ax.bar(x + width/2, real_costs, width, label='QCRAFT Scheduler (Multiplexed)', color='#1f77b4')

ax.set_ylabel('Execution Cost in USD ($) - Log Scale', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend()

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Add labels on top of bars
def autolabel(rects, is_real=False):
    for rect in rects:
        height = rect.get_height()
        if is_real:
            ax.annotate(f'${height:,.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        else:
            ax.annotate(f'${int(height):,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2, is_real=True)

fig.tight_layout()
plt.savefig('cost_comparison.png', dpi=300)
print("Plot generated successfully.")