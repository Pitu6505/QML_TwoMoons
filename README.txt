===============================================================================
GUÍA DE EJECUCIÓN Y ESPECIFICACIÓN TÉCNICA
Experimento QML: Dataset Two Moons + QCRAFT-Scheduler (Política Barbecho)
===============================================================================

Este documento contiene las instrucciones de despliegue, especificaciones de 
comunicación y directrices metodológicas para el artículo científico asociado al 
entrenamiento del modelo Quantum Machine Learning (QML) utilizando el planificador 
multiplexado QCRAFT-Scheduler.

-------------------------------------------------------------------------------
1. INSTRUCCIONES DE EJECUCIÓN SEGURA (EVITAR BLOQUEO DE PUERTOS)
-------------------------------------------------------------------------------
Al trabajar con arquitecturas asíncronas basadas en servidores puente (como aiohttp)
en entornos locales, es frecuente que la terminal se interrumpa o aborte de golpe 
debido a conflictos de red. Esto ocurre porque un proceso previo no liberó el socket, 
dejando el puerto en estado "zombie".

Para garantizar una ejecución estable y sin interrupciones, siga este protocolo:

A. Liberación manual de puertos (Windows PowerShell / CMD):
   Si el script aborta al inicio, verifique qué proceso tiene retenido el puerto 5005 
   (o el puerto configurado en MY_LOCAL_IP) e interrúmpalo:
   
   1. Buscar el identificador de proceso (PID):
      netstat -ano | findstr :5000
      
   2. Matar el proceso de forma forzada (sustituya <PID> por el número obtenido):
      taskkill /PID <PID> /F

B. Secuencia de arranque recomendada:
   1. Inicie primero el backend del QCRAFT-Scheduler (asegúrese de que responde 
      en la dirección configurada, p. ej., http://localhost:8082/).
   2. Ejecute el script de entrenamiento cuántico.

C. Ejecución en una PowerShell para no depender del entorno:
   Para evitar que el entrenamiento muera si se cierra la terminal o se corta la 
   conexión SSH, lance el script en segundo plano redirigiendo el flujo de salida 
   a un archivo de log:
   
   Start-Process python -ArgumentList "Quantum_IBM_Moons.py"
   
   Puede monitorizar el progreso en tiempo real utilizando:
   tail -f salida_experimento.log

-------------------------------------------------------------------------------
2. FORMATOS DE COMUNICACIÓN DE CIRCUITOS (API REST - JSON)
-------------------------------------------------------------------------------
El script y el planificador QCRAFT-Scheduler interactúan mediante un flujo asíncrono 
de peticiones HTTP POST (Estrategia de Callback).

A. Formato de Envío (Del Script de Entrenamiento al Scheduler)
   Ruta destino: SCHEDULER_URL + 'circuit' (e.g., http://localhost:8082/circuit)
   Método: POST
   Payload (JSON):
   {
     "url": "http://localhost:5005/circuits/e0_b0_t0.py",
     "shots": 1024,
     "provider": ["ibm"],
     "policy": "barbecho",
     "criterio": 0,
     "callback_url": "http://localhost:5000/callback",
     "circuit_name": "e0_b0_t0.py",
     "code": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n..." 
   }
   
   Campos clave:
   - policy: Directiva de optimización fijada en "barbecho".
   - callback_url: Endpoint del servidor puente local que espera los resultados.
   - code: Cadena de texto con el código fuente estructurado en Qiskit/OpenQASM.

B. Formato de Recepción (Del Scheduler al Servidor Puente Local)
   Ruta destino: MY_LOCAL_IP + '/callback' (e.g., http://localhost:5005/callback)
   Método: POST
   Payload (JSON):
   {
     "circuit_name": "e0_b0_t0.py",
     "results": {
       "00": 512,
       "01": 12,
       "10": 8,
       "11": 492
     }
   }
   
   El diccionario "results" contiene los conteos puros (counts) de la medición.
   El script procesa estos bits de forma automática mapeándolos al valor esperado 
   inter-qubit en el rango [-1, 1] necesario para derivar el gradiente (Parameter-Shift).

-------------------------------------------------------------------------------
3. CARACTERÍSTICAS CLAVE PARA EL ARTÍCULO CIENTÍFICO
-------------------------------------------------------------------------------
Al redactar la sección metodológica y de resultados del artículo sobre este modelo, 
es fundamental justificar por qué la combinación de QML con el QCRAFT-Scheduler 
aporta una ventaja competitiva en el paradigma QCaaS (Quantum Computing as a Service).

Puntos críticos a desarrollar y defender en el manuscrito:

A. El Cuello de Botella del Gradiente Cuántico:
   - Exponga matemáticamente el coste del optimizador basado en derivadas. Con la 
     regla Parameter-Shift, cada iteración requiere ejecutar (2 * Parámetros) 
     circuitos por cada muestra del lote (Batch). 
   - Para un Batch de 32 con 4 parámetros cuánticos, esto se traduce en 256 circuitos 
     lógicos por iteración. Enfoque el Scheduler no como un componente opcional, 
     sino como una infraestructura de software de integración necesaria para 
     hacer viable el entrenamiento en hardware real NISQ sin colapsar las colas.

B. Justificación Teórica de la Política "Barbecho":
   - Mitigación del Crosstalk (Diafonía): Explique que la política actúa como un 
     mecanismo de "aislamiento espacial". Al situar los circuitos lógicos en 
     regiones de la QPU físicamente distantes (separadas por una distancia x), 
     se evita la contaminación de fases inducida por puertas paralelas activas.
   - Selección Dinámica por Fidelidad: Detalle cómo el algoritmo lee la última 
     calibración de la máquina cuántica para mapear el circuito sobre los qubits 
     con menores tasas de error de lectura (Readout Error) y de compuertas CNOT.
   - Optimización Vertical (Reset): Resalte el uso de barreras y operaciones de 
     medición y reinicio rápido de registros (barrier reset), lo cual compacta 
     la ejecución temporal dentro del hardware sin necesidad de reencolar el 
     dispositivo para evaluaciones independientes del Parameter-Shift.

C. Explotación de Datos del CSV (Métricas de Rendimiento):
   - Curva de Convergencia (Loss vs. Épocas): Demuestre que mitigar el crosstalk 
     con el modo barbecho estabiliza los gradientes lógicos, permitiendo que el 
     optimizador clásico (Adam) converja de forma suave sin las oscilaciones 
     salvajes provocadas por el ruido cuántico correlacionado.
   - Porcentaje de Precisión (Accuracy): Al validar el modelo con el dataset 
     no lineal "Two Moons", se demuestra la capacidad de generalización y la 
     fidelidad del cómputo.
   - Factor de Compresión de Tiempo (Batch_Time_Seconds): Utilice esta columna 
     del CSV para contrastar los tiempos reales de procesamiento multiplexado. 
     Construya un argumento sólido sobre el ahorro económico y temporal en la 
     infraestructura en la nube, demostrando una reducción drástica respecto a 
     un flujo secuencial síncrono clásico.

===============================================================================
