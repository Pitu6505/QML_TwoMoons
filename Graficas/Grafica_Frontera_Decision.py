from pathlib import Path

import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# --- CONFIGURACIÓN ---
N_QUBITS = 2
LAYERS = 4
BASE_DIR = Path(__file__).resolve().parent
PROYECTO_DIR = BASE_DIR.parent
CHECKPOINT_FILE = PROYECTO_DIR / "Graficas" / "checkpoint_qdataset_barbecho.pth"  # Tu mejor modelo

# 1. Reconstruir el QNode (Igual que en el entrenamiento)
dev = qml.device("default.qubit", wires=N_QUBITS)

# 1. Reconstruir el QNode (A prueba de colapsos de dimensión)
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def qnode(inputs, weights):
    for i in range(LAYERS):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        # Seguro antidimensiones activado
        qml.BasicEntanglerLayers(weights[i].view(1, N_QUBITS), wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

# 2. Cargar el modelo entrenado
print(f"Cargando pesos de: {CHECKPOINT_FILE}...")
checkpoint = torch.load(CHECKPOINT_FILE)
weights = checkpoint['weights']
bias = checkpoint['bias']

def predecir_punto(x):
    with torch.no_grad():
        val = qnode(torch.tensor(x, dtype=torch.float32), weights) + bias
        return 1 if val.item() > 0 else 0

# 3. Generar 50 puntos de prueba (Caso de Uso)
# --- 3. Generar 50 puntos de prueba (DATASET CUÁNTICO) ---
np.random.seed(12) # Semilla distinta para que sean 50 puntos totalmente nuevos
X_test = np.random.uniform(0, np.pi, (50, N_QUBITS))

# Reconstruimos al Profesor para etiquetar los puntos
dev_teacher = qml.device("default.qubit", wires=N_QUBITS)
np.random.seed(42) # ⚠️ DEBE SER 42: Para mantener la receta secreta del entrenamiento
teacher_weights = np.random.uniform(0, 2 * np.pi, (2, 1, N_QUBITS))

@qml.qnode(dev_teacher)
def teacher_circuit(inputs):
    for i in range(2):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.BasicEntanglerLayers(torch.tensor(teacher_weights[i]), wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

y_test = []
for x in X_test:
    val = teacher_circuit(x) + np.random.normal(0, 0.1) # Añadimos el mismo ruido
    y_test.append(1 if val > 0 else 0)
    
y_test = np.array(y_test)
X_norm = X_test # Los datos ya nacen entre 0 y pi, no hay que normalizarlos

# 4. Crear la malla para el fondo (Decision Boundary)
print("Calculando la frontera de decisión cuántica (esto puede tardar unos segundos)...")
xx, yy = np.meshgrid(np.linspace(0, np.pi, 30), np.linspace(0, np.pi, 30))
Z = np.zeros_like(xx)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        Z[i, j] = predecir_punto([xx[i, j], yy[i, j]])

# 5. Dibujar la Gráfica
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Separar puntos por clase para colorearlos
X_class0 = X_norm[y_test == 0]
X_class1 = X_norm[y_test == 1]

plt.scatter(X_class0[:, 0], X_class0[:, 1], color='blue', edgecolors='k', label='Class 0', s=60)
plt.scatter(X_class1[:, 0], X_class1[:, 1], color='red', edgecolors='k', label='Class 1', s=60)

# Calcular aciertos en los 50 puntos
predicciones_test = [predecir_punto(x) for x in X_norm]
aciertos = sum([1 for p, r in zip(predicciones_test, y_test) if p == r])

# Calcular aciertos en los 50 puntos
predicciones_test = [predecir_punto(x) for x in X_norm]
aciertos = sum([1 for p, r in zip(predicciones_test, y_test) if p == r])

# 🟢 NUEVO: Calcular el porcentaje e imprimirlo en la terminal
porcentaje = (aciertos / len(y_test)) * 100
print("-" * 40)
print(f"🎯 RESULTADOS DE LA EVALUACIÓN:")
print(f"   Total de puntos probados: {len(y_test)}")
print(f"   Puntos acertados: {aciertos}")
print(f"   Precisión (Accuracy): {porcentaje:.2f}%")
print("-" * 40)

plt.xlabel('Feature 1 (rad)', fontsize=14)
plt.ylabel('Feature 2 (rad)', fontsize=14)
plt.legend()
plt.savefig(PROYECTO_DIR / "Graficas" / "frontera_decision_qml.png", dpi=300)
print("✅ Gráfica guardada como 'frontera_decision_qml.png'")
plt.show()