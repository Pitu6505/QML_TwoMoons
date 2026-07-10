from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual para artículos académicos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

BASE_DIR = Path(__file__).resolve().parent
PROYECTO_DIR = BASE_DIR.parent
NOMBRE_CSV = "resultados_entrenamiento_moons_barbecho.csv"


def resolver_csv():
    candidatos = [
        BASE_DIR / NOMBRE_CSV,
        PROYECTO_DIR / NOMBRE_CSV,
        PROYECTO_DIR / "TercerEntrenamiento" / NOMBRE_CSV,
    ]

    for ruta in candidatos:
        if ruta.exists():
            return ruta

    return candidatos[-1]

def generar_graficas_entrenamiento(csv_file):
    # Cargar datos
    try:
        csv_file = Path(csv_file)
        df = pd.read_csv(csv_file, sep=';')
    except Exception as e:
        print(f"Error cargando el archivo: {e}")
        return

    # Crear una figura con 3 subgráficos (Loss, Bias, Tiempo)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Eje X continuo basado en el número total de batches
    df['Iteracion_Global'] = range(1, len(df) + 1)

    # 1. Gráfica de la Función de Pérdida (Loss)
    sns.lineplot(data=df, x='Iteracion_Global', y='Loss', ax=axes[0], color='crimson', linewidth=2)
    axes[0].set_title('Evolución de la Función de Pérdida (MSE) en QPU', fontweight='bold')
    axes[0].set_ylabel('Loss')

    # 2. Gráfica de la Evolución del Sesgo (Bias)
    sns.lineplot(data=df, x='Iteracion_Global', y='Bias', ax=axes[1], color='dodgerblue', linewidth=2)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1) # Línea en 0
    axes[1].set_title('Convergencia del Parámetro de Sesgo (Bias)', fontweight='bold')
    axes[1].set_ylabel('Valor del Bias')

    # 3. Gráfica de los Tiempos de IBM (Demostración de la congestión)
    sns.barplot(data=df, x='Iteracion_Global', y='Batch_Time_Seconds', ax=axes[2], color='darkorange')
    axes[2].set_title('Tiempo de Ejecución por Batch (Impacto de las colas de IBM)', fontweight='bold')
    axes[2].set_ylabel('Segundos')
    axes[2].set_xlabel('Iteraciones Globales (Batches)')
    
    # Ocultar etiquetas del eje X en el gráfico de barras para evitar saturación visual
    axes[2].set_xticks(axes[2].get_xticks()[::len(df)//10]) 

    plt.tight_layout()
    ruta_salida = PROYECTO_DIR / "EntrenamientoDatasetCuantico" / "metricas_entrenamiento_QML.png"
    plt.savefig(ruta_salida, dpi=300)
    print("✅ Gráfica guardada como 'metricas_entrenamiento_QML.png'")
    plt.show()

if __name__ == "__main__":
    generar_graficas_entrenamiento(resolver_csv())