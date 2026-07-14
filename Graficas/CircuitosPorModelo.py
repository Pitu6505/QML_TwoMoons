import matplotlib.pyplot as plt
import numpy as np

labels = ['Model 1 (3 Layers)', 'Model 3 (4 Layers)', 'Model 2 (5 Layers)']

# ==========================================
# 1. GRÁFICA DE ESCALADO DE CIRCUITOS (75 Tareas)
# ==========================================
total_circuits = [28800, 38400, 48000]

fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
x1 = np.arange(len(labels))
bars = ax1.bar(x1, total_circuits, color=['#2ca02c', '#1f77b4', '#d62728'], width=0.5)

ax1.set_ylabel('Total Generated Circuits', fontweight='bold', fontsize=14)
ax1.set_xticks(x1)
ax1.set_xticklabels(labels, fontweight='bold', fontsize=14)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f'{int(yval):,}', 
             va='bottom', ha='center', fontweight='bold')

fig1.tight_layout()
fig1.savefig('circuit_scaling.png', bbox_inches='tight')
print("Gráfica 1 guardada como circuit_scaling.png")

# ==========================================
# 2. GRÁFICA DE COSTES (75 Tareas)
# ==========================================
real_costs = [514.50, 458.97, 601.07]
theoretical_costs = [184320, 245760, 307200]

fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
width = 0.35

rects1 = ax2.bar(x1 - width/2, theoretical_costs, width, label='Sequential Execution (Estimated)', color='#d62728')
rects2 = ax2.bar(x1 + width/2, real_costs, width, label='QCRAFT Scheduler (Multiplexed)', color='#1f77b4')

ax2.set_ylabel('Execution Cost in USD ($) - Log Scale', fontweight='bold', fontsize=14)
ax2.set_xticks(x1)
ax2.set_xticklabels(labels, fontweight='bold', fontsize=14)
ax2.legend(fontsize=12)
ax2.set_yscale('log')

def autolabel(rects, is_real=False):
    for rect in rects:
        height = rect.get_height()
        if is_real:
            ax2.annotate(f'${height:,.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            ax2.annotate(f'${int(height):,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

autolabel(rects1)
autolabel(rects2, is_real=True)

fig2.tight_layout()
fig2.savefig('cost_comparison.png', bbox_inches='tight')
print("Gráfica 2 guardada como cost_comparison.png")