import pennylane as qml
import torch
import numpy as np
import asyncio
from aiohttp import web
import aiohttp
import os
import csv
import time
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from Utiles_Scheduler import circuit_path, ensure_circuits_dir, tape_to_qiskit_script

# --- CONFIGURACIÓN DEL EXPERIMENTO MULTIPLEXADO ---
SCHEDULER_URL = "http://localhost:8082/"
MY_LOCAL_IP = "http://localhost:5005"
BATCH_SIZE = 32
EPOCHS = 15
SHOTS = 1024
N_QUBITS = 2  # 2 Qubits para datos bidimensionales (Two Moons)
LEARNING_RATE = 0.05
CHECKPOINT_FILE = "checkpoints_hero/checkpoint_moons_barbecho.pth" 
CSV_FILE = "resultados_entrenamiento_moons_barbecho.csv" 

# Variables Globales de Control Asíncrono
results_storage = {}
batch_event = asyncio.Event()
current_batch_total = 0

# --- 1. PREPARACIÓN DEL DATASET (TWO MOONS) ---
print("--- 1. Generando Dataset Two Moons ---")
# 200 muestras totales con un nivel de ruido estándar
X_raw, y_raw = make_moons(n_samples=200, noise=0.1, random_state=42)

# Normalización estricta entre 0 y PI para la codificación cuántica (AngleEmbedding)
x_min_val, x_max_val = X_raw.min(axis=0), X_raw.max(axis=0)
X_norm = np.pi * (X_raw - x_min_val) / (x_max_val - x_min_val)

# División de datos (80% Entrenamiento, 20% Validación)
TRAIN_SIZE = 160
X_train = torch.tensor(X_norm[:TRAIN_SIZE], dtype=torch.float32)
y_train = torch.tensor(y_raw[:TRAIN_SIZE], dtype=torch.long)

X_test = torch.tensor(X_norm[TRAIN_SIZE:], dtype=torch.float32)
y_test = torch.tensor(y_raw[TRAIN_SIZE:], dtype=torch.long)

print(f"✅ Dataset preparado de forma nativa.")
print(f"   Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras.")

# --- 2. DEFINICIÓN DEL QNODE (CIRCUITO) ---
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

# Parámetros entrenables: 4 pesos cuánticos y 1 bias clásico
init_weights = 0.1 * torch.randn(2, N_QUBITS) 
weights = torch.tensor(init_weights, requires_grad=True, dtype=torch.float32)
bias = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

# --- 3. SERVIDOR PUENTE ASÍNCRONO (CALLBACKS) ---
def counts_to_expval(counts):
    zeros, ones, total = 0, 0, 0
    for k, v in counts.items():
        bit = k[-1] # Tomamos el qubit objetivo de la medición
        if bit == '0': zeros += v
        else: ones += v
        total += v
    if total == 0: return 0
    return (zeros - ones) / total

async def handle_callback(request):
    try:
        data = await request.json()
        name = data.get("circuit_name")
        raw_counts = data.get("results")

        val = counts_to_expval(raw_counts)
        results_storage[name] = val
        
        # Desbloquear el bucle principal si ya han llegado todas las tareas del batch
        if len(results_storage) >= current_batch_total:
            batch_event.set()
        return web.Response(text="OK")
    except Exception as e:
        return web.Response(status=500)

async def handle_file(request):
    name = request.match_info.get('name', "Anon")
    path = circuit_path(name)
    if os.path.exists(path):
        return web.FileResponse(path)
    return web.Response(status=404)

# --- 4. MÉTRICAS Y PREDICCIÓN ---
def predict_dataset(X_input, current_weights, current_bias):
    preds = []
    with torch.no_grad():
        for x in X_input:
            val = qnode(x, current_weights) + current_bias
            preds.append(1 if val.item() > 0 else 0)
    return preds

# --- 5. GESTIÓN DE CHECKPOINTS ---
def save_checkpoint(epoch, batch_idx, optimizer, loss, accuracy):
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'weights': weights, 
        'bias': bias,
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, CHECKPOINT_FILE)

def load_checkpoint(optimizer):
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE)
        with torch.no_grad():
            weights.data = checkpoint['weights'].data
            bias.data = checkpoint['bias'].data
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"⏩ Reanudando entrenamiento desde Epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'], checkpoint['batch_idx'] + 1
    return 0, 0

# --- 6. NÚCLEO DE ENTRENAMIENTO ---
async def train_moons_barbecho():
    global weights, bias, current_batch_total
    
    # Inicialización del servidor HTTP local para recibir las respuestas de la QPU
    app = web.Application()
    app.router.add_get('/circuits/{name}', handle_file)
    app.router.add_post('/callback', handle_callback)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 5005)
    await site.start()
    print("🌐 Servidor Puente local activo en el puerto 5005")

    ensure_circuits_dir()
    os.makedirs("checkpoints_hero", exist_ok=True)

    optimizer = torch.optim.Adam([weights, bias], lr=LEARNING_RATE)
    start_epoch, start_batch_global = load_checkpoint(optimizer)

    # 🟢 MODIFICACIÓN 1: Añadida la columna 'Batch_Time_Seconds' a la cabecera
    if start_epoch == 0 and start_batch_global == 0:
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["Epoch", "Batch", "Loss", "Grad_Norm", "Bias", "Batch_Time_Seconds"])

    print(f"\n🚀 INICIANDO EXPERIMENTO QML: TWO MOONS + POLÍTICA BARBECHO 🚀")

    for epoch in range(start_epoch, EPOCHS):
        start_time_epoch = time.time()
        epoch_loss = 0
        batches_done = 0
        
        perm = torch.randperm(X_train.size(0))
        batch_counter = 0
        batch_start_from = start_batch_global if epoch == start_epoch else 0

        for i in range(0, len(X_train), BATCH_SIZE):
            if batch_counter < batch_start_from:
                batch_counter += 1
                continue 

            # 🟢 MODIFICACIÓN 2: Iniciamos el cronómetro justo antes de preparar el batch
            batch_start_time = time.time() 

            idx = perm[i:i+BATCH_SIZE]
            x_batch = X_train[idx]
            y_target = y_train[idx]
            y_target_pm = (y_target.float() * 2) - 1 

            tapes_to_send = []
            tape_map = [] 
            
            for j in range(len(x_batch)):
                x_val = x_batch[j].detach().numpy()
                with qml.tape.QuantumTape() as tape:
                    qml.AngleEmbedding(x_val, wires=range(N_QUBITS))
                    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
                    qml.expval(qml.PauliZ(0))

                g_tapes, fn = qml.gradients.param_shift(tape)
                start_idx = len(tapes_to_send)
                tapes_to_send.extend(g_tapes)
                tape_map.append((j, start_idx, len(g_tapes), fn))

            results_storage.clear()
            batch_event.clear()
            current_batch_total = len(tapes_to_send)
            
            print(f"  Epoch {epoch+1} - Batch {batch_counter+1}: Transmitiendo {current_batch_total} circuitos al Scheduler...", end="\r")
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for k, tape in enumerate(tapes_to_send):
                    fname = f"e{epoch}_b{batch_counter}_t{k}.py"
                    tape_to_qiskit_script(tape, fname, SHOTS)
                    with open(circuit_path(fname), "r") as circuit_file:
                        code = circuit_file.read()
                    
                    payload = {
                        "url": f"{MY_LOCAL_IP}/circuits/{fname}", 
                        "shots": SHOTS,
                        "provider": ['ibm'],
                        "policy": "barbecho", 
                        "criterio": 0,
                        "callback_url": f"{MY_LOCAL_IP}/callback",
                        "circuit_name": fname,
                        "code": code
                    }
                    task = session.post(SCHEDULER_URL + 'circuit', json=payload)
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            
            print(f"  Epoch {epoch+1} - Batch {batch_counter+1}: Esperando procesamiento multiplexado en QPU...", end="\r")
            await batch_event.wait()
            
            # 🟢 MODIFICACIÓN 3: Paramos el cronómetro en cuanto el Scheduler nos devuelve todo
            batch_duration = time.time() - batch_start_time
            
            grad_w_accum = torch.zeros_like(weights)
            grad_b_accum = torch.tensor(0.0) 
            
            for j, start, count, fn in tape_map:
                res_list = []
                for k in range(start, start+count):
                    fname = f"e{epoch}_b{batch_counter}_t{k}.py"
                    val = results_storage.get(fname, 0.0) 
                    res_list.append(val)
                
                grad_per_sample = fn(res_list)
                g_w = torch.tensor(grad_per_sample[0], dtype=torch.float32)
                
                pred_sim = qnode(x_batch[j], weights) + bias 
                error = pred_sim - y_target_pm[j]
                
                grad_w_accum += 2 * error * g_w
                grad_b_accum += 2 * error 

            grad_w_accum /= len(x_batch)
            grad_b_accum /= len(x_batch) 
            
            weights.grad = grad_w_accum
            bias.grad = grad_b_accum 
            
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                preds_batch = torch.stack([qnode(x_batch[j], weights) + bias for j in range(len(x_batch))])
                batch_loss = torch.mean((preds_batch - y_target_pm)**2)
            
            epoch_loss += batch_loss.item()
            grad_norm = grad_w_accum.norm().item()
            
            print(f"\n  📈 Batch {batch_counter+1} | Loss: {batch_loss.item():.4f} | Grad Norm: {grad_norm:.4f} | Tiempo: {batch_duration:.1f}s")
            
            # 🟢 MODIFICACIÓN 4: Guardamos en el CSV inmediatamente después de cada batch
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow([epoch+1, batch_counter+1, batch_loss.item(), grad_norm, bias.item(), round(batch_duration, 2)])
            
            # Salvaguardamos el checkpoint (se actualiza en cada batch para mayor seguridad)
            save_checkpoint(epoch, batch_counter, optimizer, batch_loss.item(), 0)

            for k in range(len(tapes_to_send)):
                fname = f"e{epoch}_b{batch_counter}_t{k}.py"
                fpath = circuit_path(fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
            
            batches_done += 1
            batch_counter += 1
        
        # --- EVALUACIÓN DE PRECISIÓN (ACCURACY) AL FINAL DE LA ÉPOCA ---
        y_pred_test = predict_dataset(X_test, weights, bias)
        epoch_accuracy = accuracy_score(y_test.numpy(), y_pred_test) * 100

        avg_loss = epoch_loss / batches_done if batches_done > 0 else 0
        duration_epoch = time.time() - start_time_epoch
        
        print(f"\n✨ ÉPOCA {epoch+1} COMPLETADA ✨")
        print(f"   Loss Promedio: {avg_loss:.4f} | Accuracy en Test: {epoch_accuracy:.2f}% | Tiempo Total Época: {duration_epoch:.1f}s")
        print("=" * 80)

    await runner.cleanup()
    print("🏆 ¡ENTRENAMIENTO MULTIPLEXADO COMPLETADO CON ÉXITO!")

if __name__ == "__main__":
    asyncio.run(train_moons_barbecho())