from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual style for academic figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

BASE_DIR = Path(__file__).resolve().parent
PROYECTO_DIR = BASE_DIR.parent
NOMBRE_CSV = "resultados_entrenamiento_qdataset_barbecho3ºModelo.csv"


def resolver_csv():
    candidatos = [
        BASE_DIR / NOMBRE_CSV,
        PROYECTO_DIR / NOMBRE_CSV,
        PROYECTO_DIR / "EntrenamientoDatasetCuantico" / NOMBRE_CSV,
    ]

    for ruta in candidatos:
        if ruta.exists():
            return ruta

    return candidatos[-1]

def generar_graficas_entrenamiento(csv_file):
    # Load data
    try:
        csv_file = Path(csv_file)
        df = pd.read_csv(csv_file, sep=';')
    except Exception as e:
        print(f"Error loading the file: {e}")
        return

    # Create a figure with 3 subplots (Loss, Bias, Time)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Continuous X axis based on the total number of batches
    df['Iteracion_Global'] = range(1, len(df) + 1)

    # 1. Loss curve
    sns.lineplot(data=df, x='Iteracion_Global', y='Loss', ax=axes[0], color='crimson', linewidth=2)
    axes[0].set_title('Loss Function Evolution (MSE) on the QPU', fontweight='bold')
    axes[0].set_ylabel('Loss')

    # 2. Bias evolution curve
    sns.lineplot(data=df, x='Iteracion_Global', y='Bias', ax=axes[1], color='dodgerblue', linewidth=2)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1) # Zero line
    axes[1].set_title('Bias Parameter Convergence', fontweight='bold')
    axes[1].set_ylabel('Bias Value')

    # 3. IBM execution times (queue congestion demonstration)
    sns.barplot(data=df, x='Iteracion_Global', y='Batch_Time_Seconds', ax=axes[2], color='darkorange')
    axes[2].set_title('Execution Time per Batch (Impact of IBM Queues)', fontweight='bold')
    axes[2].set_ylabel('Seconds')
    axes[2].set_xlabel('Global Iterations (Batches)')
    
    # Hide some X-axis labels in the bar chart to avoid visual clutter
    axes[2].set_xticks(axes[2].get_xticks()[::len(df)//10]) 

    plt.tight_layout()
    ruta_salida = PROYECTO_DIR / "EntrenamientoDatasetCuantico" / "metricas_entrenamiento_QML.png"
    plt.savefig(ruta_salida, dpi=300)
    print("✅ Figure saved as 'metricas_entrenamiento_QML.png'")
    plt.show()

if __name__ == "__main__":
    generar_graficas_entrenamiento(resolver_csv())