import pennylane as qml
import os
import sys

CIRCUITS_DIR = "circuitos"

def ensure_circuits_dir():
    os.makedirs(CIRCUITS_DIR, exist_ok=True)
    return CIRCUITS_DIR

def circuit_path(filename):
    return os.path.join(ensure_circuits_dir(), filename)

# --- GENERADOR NATIVO QISKIT ---
def tape_to_qiskit_script(tape, filename, shots=1024):
    """
    Convierte un QuantumTape de PennyLane a un script de Qiskit nativo (Python puro).
    Evita QASM y 'circuit_to_qiskit' para ser 100% compatible con los parsers regex del Scheduler.
    """
    try:
        # 1. Expansión para desglosar plantillas (como AngleEmbedding) en puertas básicas
        expanded_tape = tape.expand()
        num_qubits = len(expanded_tape.wires)

        # 2. Generar el preámbulo exacto que busca tu parser en scheduler_policies.py
        lineas_codigo = [
            "from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister",
            "import numpy as np",
            "",
            f"qreg_q = QuantumRegister({num_qubits}, 'q')",
            f"creg_c = ClassicalRegister({num_qubits}, 'c')",
            "circuit = QuantumCircuit(qreg_q, creg_c)",
            ""
        ]

        # 3. Mapeo directo de operaciones PennyLane a comandos nativos Qiskit
        for op in expanded_tape.operations:
            nombre_puerta = op.name
            qubits = op.wires.tolist()
            parametros_crudos = op.parameters

            # Extraer parámetros de los tensores de PyTorch de forma segura
            p_vals = []
            for p in parametros_crudos:
                if hasattr(p, 'item'):
                    p_vals.append(float(p.item()))
                else:
                    p_vals.append(float(p))

            # Traducción de puertas
            if nombre_puerta == "RX":
                lineas_codigo.append(f"circuit.rx({p_vals[0]:.6f}, qreg_q[{qubits[0]}])")
            elif nombre_puerta == "RY":
                lineas_codigo.append(f"circuit.ry({p_vals[0]:.6f}, qreg_q[{qubits[0]}])")
            elif nombre_puerta == "RZ":
                lineas_codigo.append(f"circuit.rz({p_vals[0]:.6f}, qreg_q[{qubits[0]}])")
            elif nombre_puerta in ["CNOT", "CX"]:
                lineas_codigo.append(f"circuit.cx(qreg_q[{qubits[0]}], qreg_q[{qubits[1]}])")
            elif nombre_puerta == "Hadamard":
                lineas_codigo.append(f"circuit.h(qreg_q[{qubits[0]}])")
            elif nombre_puerta == "PauliX":
                lineas_codigo.append(f"circuit.x(qreg_q[{qubits[0]}])")
            elif nombre_puerta == "PauliY":
                lineas_codigo.append(f"circuit.y(qreg_q[{qubits[0]}])")
            elif nombre_puerta == "PauliZ":
                lineas_codigo.append(f"circuit.z(qreg_q[{qubits[0]}])")
            elif nombre_puerta == "PhaseShift":
                lineas_codigo.append(f"circuit.p({p_vals[0]:.6f}, qreg_q[{qubits[0]}])")
            else:
                lineas_codigo.append(f"# WARNING: Puerta no soportada - {nombre_puerta}")

        # 4. Añadir mediciones explícitas al final
        lineas_codigo.append("")
        for q in range(num_qubits):
            lineas_codigo.append(f"circuit.measure(qreg_q[{q}], creg_c[{q}])")

        # 5. Guardar el archivo (.py que leerá el Scheduler)
        filepath = circuit_path(filename)
        with open(filepath, "w") as f:
            f.write("\n".join(lineas_codigo))
            
        return filepath
        
    except Exception as e:
        print(f"❌ Error en tape_to_qiskit_script: {e}")
        return None

# --- GENERADOR NATIVO AWS BRAKET ---
def tape_to_braket_script(tape, filename, shots):
    """
    Convierte un QuantumTape de PennyLane a un script de AWS Braket SDK.
    Compatible con tensores de PyTorch y el Scheduler de multiplexación.
    """
    lineas_codigo = [
        "import numpy as np",
        "from braket.devices import LocalSimulator",
        "from braket.circuits import Circuit",
        "from collections import Counter",
        "",
        f"shots = {shots}",
        "",
        "circuit = Circuit()"
    ]
    
    for op in tape.operations:
        nombre_puerta = op.name
        qubits = op.wires.tolist()
        parametros_crudos = op.parameters
        
        p_vals = []
        for p in parametros_crudos:
            if hasattr(p, 'item'):
                p_vals.append(float(p.item()))
            else:
                p_vals.append(float(p))
        
        if nombre_puerta == "RX":
            lineas_codigo.append(f"circuit.rx({qubits[0]}, {p_vals[0]:.6f})")
        elif nombre_puerta == "RY":
            lineas_codigo.append(f"circuit.ry({qubits[0]}, {p_vals[0]:.6f})")
        elif nombre_puerta == "RZ":
            lineas_codigo.append(f"circuit.rz({qubits[0]}, {p_vals[0]:.6f})")
        elif nombre_puerta == "CNOT":
            lineas_codigo.append(f"circuit.cnot({qubits[0]}, {qubits[1]})")
        elif nombre_puerta == "Hadamard":
            lineas_codigo.append(f"circuit.h({qubits[0]})")
        elif nombre_puerta == "PauliX":
            lineas_codigo.append(f"circuit.x({qubits[0]})")
        elif nombre_puerta == "PauliZ":
            lineas_codigo.append(f"circuit.z({qubits[0]})")
        else:
            lineas_codigo.append(f"# WARNING: Puerta no soportada - {nombre_puerta}")

    filepath = circuit_path(filename)
    with open(filepath, "w") as f:
        f.write("\n".join(lineas_codigo))

    return filepath