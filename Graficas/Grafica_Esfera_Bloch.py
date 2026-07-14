from pathlib import Path
import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
N_QUBITS = 2
LAYERS = 4
BASE_DIR = Path(__file__).resolve().parent
PROYECTO_DIR = BASE_DIR.parent
CHECKPOINT_FILE = PROYECTO_DIR / "Graficas" / "checkpoint_qdataset_barbecho.pth"

# --- 1. EXACT DATASET RECONSTRUCTION (No guesses) ---
print("Reconstructing the exact quantum training dataset...")
np.random.seed(42)
n_samples = 200
X_raw = np.random.uniform(0, np.pi, (n_samples, N_QUBITS))

dev_teacher = qml.device("default.qubit", wires=N_QUBITS)
teacher_weights = np.random.uniform(0, 2 * np.pi, (2, 1, N_QUBITS))

@qml.qnode(dev_teacher)
def teacher_circuit(inputs):
    for i in range(2):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.BasicEntanglerLayers(torch.tensor(teacher_weights[i]), wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

y_raw = []
for x in X_raw:
    val = teacher_circuit(x)
    val += np.random.normal(0, 0.1) # Noise identical to training
    y_raw.append(1 if val > 0 else 0)

# Isolate the EXACT test set (the last 40 samples of the array)
TRAIN_SIZE = 160
X_test = X_raw[TRAIN_SIZE:]
y_test_real = y_raw[TRAIN_SIZE:]

# --- 2. STUDENT QNODE (MODIFIED FOR 3D) ---
dev = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev)
def qnode_bloch(inputs, weights):
    for i in range(LAYERS):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.BasicEntanglerLayers(weights[i].view(1, N_QUBITS), wires=range(N_QUBITS))
    
    return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]

# --- 3. LOAD TRAINED MODEL ---
print(f"Loading student weights from: {CHECKPOINT_FILE.name}")
checkpoint = torch.load(CHECKPOINT_FILE)
weights = checkpoint['weights']
bias = checkpoint['bias']

# --- 4. EXACT EVALUATION ON THE TEST SET ---
predicciones_xyz = []
predicciones_clase = []

for x in X_test:
    with torch.no_grad():
        coords = qnode_bloch(torch.tensor(x, dtype=torch.float32), weights)
        predicciones_xyz.append(coords)
        
        val_student = coords[2].item() + bias.item()
        predicciones_clase.append(1 if val_student > 0 else 0)

aciertos = sum([1 for p, r in zip(predicciones_clase, y_test_real) if p == r])
accuracy = (aciertos / len(y_test_real)) * 100

print("-" * 40)
print(f"🎯 Recovered accuracy: {accuracy:.2f}%")
print("-" * 40)

# --- 5. DRAW THE BLOCH SPHERE ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the sphere structure
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
ax.plot_surface(xs, ys, zs, color="whitesmoke", alpha=0.15, edgecolor="silver", lw=0.5)

# Draw the equator (Z=0) and the axes
ax.plot(np.cos(u[:,0]), np.sin(u[:,0]), 0, color="gray", lw=1.5, label="Decision Boundary (Equator)")
ax.plot([0,0], [0,0], [-1,1], color="black", linestyle="--", lw=1)

# Separate points based on whether the model got them right or wrong
for i, coord in enumerate(predicciones_xyz):
    px, py, pz = coord[0].item(), coord[1].item(), coord[2].item()
    
    if predicciones_clase[i] == y_test_real[i]:
        color = 'mediumseagreen'
        marker = 'o'
    else:
        color = 'crimson'
        marker = 'x'
        
    ax.scatter(px, py, pz, color=color, marker=marker, s=90, alpha=0.8, edgecolor='k' if marker=='o' else None)

# Invisible legend entries for the info box
ax.scatter([], [], [], color='mediumseagreen', marker='o', label='Correct', s=60)
ax.scatter([], [], [], color='crimson', marker='x', label='Incorrect', s=60)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis (Classification)')
ax.legend(loc='upper right')

# Adjust limits for perfect spherical proportions
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.view_init(elev=20, azim=60)

plt.savefig(PROYECTO_DIR / "Graficas" / "esfera_bloch_qdataset.png", dpi=300, bbox_inches='tight')
print("✅ Figure saved as 'esfera_bloch_qdataset.png'")
plt.show()