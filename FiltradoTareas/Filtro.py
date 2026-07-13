import pandas as pd

# ==========================================
# CONFIGURACIÓN (Modifica estos valores)
# ==========================================
ARCHIVO = 'IBM_Worloads.csv'  # Nombre del archivo

# Fechas exactas del entrenamiento
CREACION_INICIO = '07/07/2026 09:41 AM'
CREACION_FIN = None 

FIN_INICIO = None    
FIN_FIN = '07/09/2026 03:37 PM'

# Límites y costes del entrenamiento QML
MAX_TAREAS = 75       # Número máximo de tareas (circuitos) correspondientes al entrenamiento
PRECIO_POR_MINUTO = 98 # Coste en dólares por minuto de ejecución
# ==========================================

def analizar_workloads_qml(file_path, created_start, created_end, completed_start, completed_end, max_tareas, precio_minuto):
    print(f"Cargando archivo: {file_path}...")
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return

    # Convertir columnas a datetime asegurando formato UTC
    df['Created'] = pd.to_datetime(df['Created'], utc=True)
    df['Completed'] = pd.to_datetime(df['Completed'], utc=True)
    
    # Filtro inicial: máquina 'ibm_fez' y estado 'completed'
    filtro = (df['Status'].str.lower() == 'completed') & (df['QPU'].str.lower() == 'ibm_fez')
    
    # Aplicar filtros de fecha y hora
    if created_start:
        filtro &= (df['Created'] >= pd.to_datetime(created_start, utc=True))
    if created_end:
        filtro &= (df['Created'] <= pd.to_datetime(created_end, utc=True))
        
    if completed_start:
        filtro &= (df['Completed'] >= pd.to_datetime(completed_start, utc=True))
    if completed_end:
        filtro &= (df['Completed'] <= pd.to_datetime(completed_end, utc=True))

    df_filtrado = df[filtro].copy()
    
    # Ordenar cronológicamente por fecha de creación para asegurar que tomamos las primeras tareas enviadas
    df_filtrado = df_filtrado.sort_values(by='Created', ascending=True)
    
    tareas_encontradas_total = len(df_filtrado)
    
    # Recortar a un máximo de MAX_TAREAS
    if len(df_filtrado) > max_tareas:
        df_filtrado = df_filtrado.head(max_tareas)
        
    num_circuitos = len(df_filtrado)
    tiempo_total_segundos = df_filtrado['Usage (seconds)'].sum()
    tiempo_total_minutos = tiempo_total_segundos / 60
    
    coste_total = tiempo_total_minutos * precio_minuto
    
    print("\n" + "="*60)
    print(" 🧠 RESULTADOS DEL ENTRENAMIENTO QML (IBM FEZ)")
    print("="*60)
    print(f"🔹 Tareas completadas encontradas en el rango: {tareas_encontradas_total}")
    if tareas_encontradas_total > max_tareas:
        print(f"⚠️  Se han descartado {tareas_encontradas_total - max_tareas} tareas sobrantes (límite ajustado a {max_tareas}).")
    
    print(f"🔹 Circuitos analizados: {num_circuitos}")
    print(f"🔹 Tiempo total de ejecución: {tiempo_total_segundos} segundos ({tiempo_total_minutos:.2f} minutos)")
    print(f"💰 Coste total estimado del entrenamiento: ${coste_total:.2f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    analizar_workloads_qml(ARCHIVO, CREACION_INICIO, CREACION_FIN, FIN_INICIO, FIN_FIN, MAX_TAREAS, PRECIO_POR_MINUTO)