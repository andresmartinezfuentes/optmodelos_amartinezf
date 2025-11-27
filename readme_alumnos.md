# Guía Completa: Optimización de Stable Diffusion con ONNX y TensorRT

## Instrucciones:
1. La práctica debe resolverse en la DGX utilizando Docker.
2. Agregar "_inicialyapellido" al final del nombre de carpeta de proyecto y también
al final del nombre de servicio en docker.
3. Debes completar una pequeña parte de código "TO DO" indicada en measure_trt_TODO.py y despues renombrar el fichero a measure_trt.py. Recuerda que la solución que debes implementar es un fragmento de código breve, y únicamente dentro de este "TO DO" siguiendo las instrucciones indicadas. El resto del proyecto está listo para funcionar.
4. Debes ejecutar los 5 modos de inferencia explicados en la guía para generar una muestra de imagen utilizando GPU en la DGX, manteniendo todo el resto de configuración (sampler, steps etc.) iguales. Estos modos son: Torch (fp32), Torch (fp16), ONNX RT (modelo .onnx base), ONNX RT (modelo .onnx optimizado), TensorRT (.engine optimizado)
5. Debes entregar:
  - fichero de texto con la parte de código TO DO completa únicamente.
  - fichero .zip con carpeta de outputs (una muestra de iamgen por cada tipo de inferencia y los     ficheros .txt con metricas de latencia)
  - fichero .md con conclusiones: comparativa numérica de latencias de generación de imagen obtenida bajo los 5 modos. Comenta de manera breve pero justificada las conclusiones más relevantes que extraes de la práctica.


## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Flujos de Trabajo de Inferencia](#flujos-de-trabajo-de-inferencia)
4. [Análisis de Performance](#análisis-de-Performance)

---

## Introducción

Esta práctica muestra cómo optimizar un modelo generativo Stable Diffusion para inferencia. Se exploran varias alternativas explicadas en la teoría, comparando rendimiento y comprendiendo los trade-offs entre velocidad, portabilidad y complejidad.

### ¿Qué es Stable Diffusion?

Stable Diffusion es un modelo generativo texto-a-imagen con tres componentes:

1. **Text Encoder (CLIP)**: Convierte prompts a embeddings
2. **UNet**: Proceso de difusión iterativo (orden del 90% del tiempo de cómputo)
3. **VAE Decoder**: Convierte latents a imagen RGB

### Estrategia de optimización

Exploraremos cinco configuraciones con niveles crecientes de optimización:

| Configuración | Backend | Modificación | Speedup aprox. | Complejidad |
|---------------|---------|--------------|------------------|-------------|
| **PyTorch FP32** | PyTorch | Ninguna | 1.0x | Baja |
| **PyTorch FP16** | PyTorch | Half-precision | 1.5x | Baja |
| **ONNX Base** | ONNX Runtime | Grafo estático | 1.5x | Media |
| **ONNX Optimizado** | ONNX Runtime | Fusiones de grafo | 2.0x | Media |
| **TensorRT** | TensorRT | Compilación GPU | 2.6x | Alta |

**Diferencias Clave:**

- **FP32 vs FP16**: Half-precision reduce memoria a la mitad y aprovecha Tensor Cores, sacrificando precisión numérica (impacto mínimo en calidad visual).

- **ONNX Base vs Optimizado**: El modelo .onnx base es conversión directa de PyTorch a grafo estático. El modelo .onnx optimizado aplica fusiones de operadores (attention, layer norm) reduciendo kernel launches.

- **ONNX vs TensorRT**: ONNX es portátil entre GPUs. TensorRT compila específicamente para tu GPU, seleccionando kernels óptimos mediante profiling real y aplicando precision mixing automático (FP16/FP32 por capa).

## Requisitos Previos: Hardware y Software

- GPU NVIDIA con CUDA (Compute Capability >= 7.0)
- Mínimo 12GB VRAM
- Docker con NVIDIA Container Toolkit
- CUDA 12.5+, TensorRT 10.x, Python 3.10

---

## Estructura del proyecto

### Estructura de archivos

```
practica_optim/
├── export_onnx.py           # PyTorch → ONNX
├── optimize_onnx.py         # Optimización grafo ONNX
├── measure_trt.py           # Benchmarking multi-backend
├── TRTUNetWrapper.py        # Wrapper engine TensorRT
├── trt_inference.py         # Inferencia TensorRT
├── run_trtexec_auto.sh      # Compilación TensorRT
├── Dockerfile               # Configuración contenedor
└── docker-compose.yml       # Orquestación
```

### Componentes clave

#### export_onnx.py

Convierte Stable Diffusion de PyTorch a ONNX mediante "tracing": ejecuta el modelo con inputs de ejemplo y registra operaciones, generando un grafo estático. PyTorch usa grafos dinámicos (construidos en ejecución) que ofrecen flexibilidad pero dificultan optimizaciones. ONNX representa el modelo como grafo estático donde todas las operaciones se conocen anticipadamente, permitiendo optimizaciones.

Utiliza la librería Optimum de Hugging Face con `export=True` para automatizar el proceso. Descarga el modelo, traza cada componente (UNet, text encoder, VAE) y genera archivos ONNX con pesos en float16, reduciendo tamaño a la mitad. Resultado: directorio `stable_diffusion_onnx/` (~2.5GB).

#### optimize_onnx.py

Aplica transformaciones al grafo ONNX mejorando rendimiento mediante fusión de operadores (combina operaciones consecutivas en operadores compuestos optimizados) y constant folding (pre-calcula expresiones constantes). Por ejemplo, MatMul + Add + ReLU se fusionan en un solo operador `FusedMatMul`, que luego ONNX Runtime ejecutará con un kernel optimizado.

ONNX optimiza el grafo de forma portátil mediante fusiones y simplificaciones a nivel de operadores (funciona en cualquier GPU compatible). TensorRT (que se utilizará más adelante) compila para tu GPU específica, probando múltiples implementaciones de kernel para cada operación y seleccionando la más rápida mediante profiling real. TensorRT conoce detalles arquitectónicos (CUDA cores, cache, Tensor Cores) y aplica optimizaciones de bajo nivel imposibles sin ese conocimiento, logrando ~2x mejora adicional pero requiriendo recompilación para GPUs diferentes.

Por defecto optimiza solo el UNet (cuello de botella: 70-80% del tiempo).

#### measure_trt.py

Herramienta de benchmarking que compara las cinco configuraciones. Selecciona automáticamente GPU con más memoria libre (importante en sistemas multi-GPU), controla reproducibilidad con seeds, y mide latencia total de generación de imagen (comparativa se realizará manteniendo resto de configuración idéntica)
En el caso de maxima optimización, mediante TensorRT, se toman métricas más detalladas de latencia por componentes (UNet, VAE, CLIP encoder).

Ejecuta warmup iterations (calentar GPU, compilar kernels JIT) antes de mediciones cronometradas. Guarda imágenes generadas y archivos `.metrics.txt` con tiempos, configuración y estadísticas.

**Casos Implementados:**

1. **PyTorch FP32**: Baseline sin optimizaciones, precisión completa (32 bits), ejecución eager
2. **PyTorch FP16**: Half-precision aprovechando Tensor Cores
3. **ONNX Base**: Grafo estático sin optimizaciones adicionales
4. **ONNX Optimizado**: Con fusiones de operadores aplicadas
5. **TensorRT**: Motor compilado para GPU específica, solo UNet (híbrido con PyTorch para text encoder y VAE)

#### TRTUNetWrapper.py

Interfaz Python para ejecutar engines TensorRT (archivos `.engine` con programas binarios optimizados). Gestiona carga de engine, buffers de memoria GPU, y ejecución de inferencia. Cuando recibe tensores de entrada, copia datos CPU→GPU, ejecuta el motor, y devuelve resultados como tensores PyTorch.

Usa buffers "pinned" en CPU (optimizados para transferencias rápidas) y streams CUDA para ejecución asíncrona (solapa transferencias con computación). Gestiona contextos CUDA con `push()`/`pop()` para correcta ejecución en multi-GPU.

#### trt_inference.py

Implementa pipeline híbrido: UNet con TensorRT (cuello de botella, 70-80% del tiempo, ejecutado 25-50 veces por imagen) y text encoder/VAE en PyTorch (ejecutados una unica vez y con rendimiento aceptable, si se desea se pueden optimizar también pero no tienen peso tan significativo en latencia).

Flujo: obtiene embeddings del prompt con text encoder PyTorch → inicializa latents con ruido → loop de difusión con UNet TensorRT prediciendo ruido en cada timestep → decodifica latents limpios a RGB con VAE PyTorch. Incluye mediciones precisas por componente.

#### run_trtexec_auto.sh

Compila ONNX a motor TensorRT mediante "building": análisis de grafo + transformaciones automáticas + profiling exhaustivo. TensorRT prueba múltiples implementaciones (kernels CUDA) para cada operación en tu GPU real, midiendo cuál es más rápida. Si ONNX de entrada partiera de FP32, TensorRT podría hacer precision mixing automático FP16/FP32 por capa para maximizar velocidad manteniendo precisión aceptable.

Las optimizaciones dependen de características específicas de tu GPU (CUDA cores, cache, Tensor Cores, ancho de banda). El resultado (`.engine`) solo funciona en el modelo exacto de GPU donde fue compilado. Tiempo de compilación: 8-10 minutos (una sola vez), ejecuciones futuras cargan instantáneamente.


---

## Flujos de Trabajo de Inferencia

### Configuración Inicial

```bash
cd workspace/nombrepractica_inicialyapellido/
docker compose build
docker compose up -d
docker exec -it sd-tensorrt_inicialyapellido bash
```

---

### 1. Inferencia PyTorch FP32 (Baseline)

**Comando:**
```bash
python measure_trt.py \
  --prompt "a photograph of an astronaut riding a horse" \
  --backend torch \
  --model_id runwayml/stable-diffusion-v1-5 \
  --steps 25 \
  --height 512 \
  --width 512 \
  --guidance_scale 7.5 \
  --sampler dpmpp_2m \
  --seed 1 \
  --runs 5 \
  --warmup 2 \
  --output outputs/torch_output_fp32.png \
  --precision fp32
```
Backend baseline sin optimizaciones, usando precisión completa (32 bits). Descarga modelo desde Hugging Face (se cachea), ejecuta 2 warmups para calentar GPU, luego 5 runs cronometrados. Guarda imagen en `outputs/torch_output_fp32.png` y métricas en `outputs/torch_output_fp32.metrics.txt`.

Esta configuración no aprovecha Tensor Cores y usa doble de memoria que FP16. Sirve como referencia para medir todas las mejoras.

**Como valor orientativo, medido en GPU H200 conectada a DGX, la generación de cada muestra puede tomar aproximadamente 0.898ss**

---

### 2. Inferencia PyTorch FP16

**Comando:**
```bash
python measure_trt.py \
  --prompt "a photograph of an astronaut riding a horse" \
  --backend torch \
  --model_id runwayml/stable-diffusion-v1-5 \
  --steps 25 \
  --height 512 \
  --width 512 \
  --guidance_scale 7.5 \
  --sampler dpmpp_2m \
  --seed 1 \
  --runs 5 \
  --warmup 2 \
  --output outputs/torch_output_fp16.png \
  --precision fp16
```
Convierte modelo a half-precision, aprovechando Tensor Cores (hardware especializado en GPUs modernas). Reduce memoria a la mitad y mejora velocidad significativamente con impacto mínimo en calidad visual.

**Como valor orientativo, medido en GPU H200 conectada a DGX, la generación de cada muestra puede tomar aproximadamente 0.545s**

---

### 3. Inferencia ONNX Runtime (Base)

**3.1. Paso Previo a Inferencia: Exportar a ONNX**

```bash
python export_onnx.py
```

Este script descarga Stable Diffusion (~5GB, cachea en `/app/.cache/huggingface/`) y realiza tracing: ejecuta cada componente con tensores de ejemplo, registrando operaciones para generar grafo ONNX estático. Convierte pesos a float16. Resultado: `stable_diffusion_onnx/` (~2.5GB) con UNet, text encoder, y VAE en formato ONNX.

Warnings esperados durante la exportación (pero no preocupantes):

<sub><i> - **TracerWarning - Converting tensor to Python boolean**: ONNX no puede representar control flow dinámico  (if/else con condiciones que dependen de datos). Durante el tracing, estas condiciones se evalúan con los inputs de ejemplo y se "congelan" en el grafo resultante. Por ejemplo, un `if seq_length > 77:` se convierte en un camino fijo (true o false) basado en el valor durante el tracing. Esto es esperado y no afecta la funcionalidad para los inputs que usaremos.</i></sub>

<sub><i> - **Memcpy nodes added for CUDAExecutionProvider**: Algunas operaciones (especialmente relacionadas con shapes y dimensiones) son más eficientes en CPU. ONNX Runtime automáticamente inserta copias de memoria para mover datos entre CPU y GPU cuando es necesario. El overhead de estas copias es mínimo para operaciones pequeñas como cálculos de dimensiones.</i></sub>

<sub><i> - **Some nodes not assigned to preferred execution provider**: ONNX Runtime optimiza la asignación de operaciones entre CPU y GPU. Operaciones triviales o que requieren sincronización pueden ejecutarse en CPU incluso con `CUDAExecutionProvider` activo. El runtime toma esta decisión para minimizar transferencias innecesarias y maximizar rendimiento global.</i></sub>

**3.2. Ejecutar inferencia y medir latencia con `measure_trt.py`:**
```bash
python measure_trt.py \
  --prompt "a photograph of an astronaut riding a horse" \
  --backend onnxrt \
  --onnx_dir stable_diffusion_onnx \
  --steps 25 \
  --height 512 \
  --width 512 \
  --guidance_scale 7.5 \
  --sampler dpmpp_2m \
  --seed 1 \
  --runs 5 \
  --warmup 2 \
  --output outputs/onnx_output.png
```
Carga modelo ONNX base (sin optimizaciones adicionales) y ejecuta con `CUDAExecutionProvider`, aún no hay optimización mediante fusiones de operadores.

**Como valor orientativo, medido en GPU H200 conectada a DGX, generación de cada muestra puede tomar aproximadamente 0.606s**

---

### 4. Inferencia ONNX Runtime (Optimizado)

**4.1. Paso Previo a inferencia: Optimizar ONNX (tras haber exportado el modelo ONNX)**

```bash
python optimize_onnx.py
```

Aplica `onnxruntime.transformers.optimizer.optimize_model()` al UNet. Analiza grafo identificando patrones fusionables (attention multi-head, layer norm), aplica constant folding (pre-calcula constantes), elimina código muerto, y simplifica expresiones algebraicas. Resultado: `stable_diffusion_onnx_opt/` con grafo optimizado (menos nodos, operaciones más eficientes).

**4.2. Ejecutar inferencia y medir latencia con `measure_trt.py`:**
```bash
python measure_trt.py \
  --prompt "a photograph of an astronaut riding a horse" \
  --backend onnxrt \
  --onnx_dir stable_diffusion_onnx_opt \
  --steps 25 \
  --height 512 \
  --width 512 \
  --guidance_scale 7.5 \
  --sampler dpmpp_2m \
  --seed 1 \
  --runs 5 \
  --warmup 2 \
  --output outputs/onnx_opt_output.png
```

Carga modelo con fusiones aplicadas. Las operaciones fusionadas reducen kernel launches y mejoran acceso a memoria. Mejora notable sobre ONNX base manteniendo portabilidad.

**Como valor orientativo, medido en GPU H200 conectada a DGX, generación de cada muestra puede tomar aproximadamente 0.439s**

---

### 5. Inferencia TensorRT

**5.1. Paso Previo a inferencia: Compilar Engine TensorRT**

```bash
run_trtexec_auto.sh
```

Compila UNet ONNX a motor TensorRT optimizado. 

Como valor orientativo, medido en GPU H200 conectada a DGX, el proceso puede tomar aproximadamente 500s, pero solo habrá que ejecutarlo una vez, para generar el .engine, en posteriores inferencias se utilizará el .engine ya generado.

1. **Análisis**: TensorRT examina grafo ONNX y aplica transformaciones (fusiones, simplificaciones)
2. **Profiling**: Prueba múltiples implementaciones de kernels CUDA para cada operación, midiendo cuál es más rápida en tu GPU
3. **Precision Mixing**: En caso de que modelo input esté en precisión fp32, decide automáticamente qué capas ejecutar en FP16/FP32 para equilibrar velocidad y precisión
4. **Serialización**: Guarda `unet_fp16.engine` (~1.7GB) con kernels optimizados

El engine está compilado específicamente para el modelo de GPU (CUDA cores, cache, Tensor Cores), requiere recompilación para GPUs diferentes.

**5.2. Ejecutar inferencia y medir latencia con `measure_trt.py`:**
```bash
python measure_trt.py \
  --prompt "a photograph of an astronaut riding a horse" \
  --backend trt \
  --engine stable_diffusion_onnx/unet_fp16.engine \
  --model_id runwayml/stable-diffusion-v1-5 \
  --steps 25 \
  --height 512 \
  --width 512 \
  --guidance_scale 7.5 \
  --sampler dpmpp_2m \
  --seed 1 \
  --runs 5 \
  --warmup 2 \
  --output outputs/trt_opt_output.png
```

Carga motor TensorRT para UNet (cuello de botella) manteniendo text encoder y VAE en PyTorch (aproximación híbrida pragmática). Durante ejecución, UNet optimizado se ejecuta múltiples veces (una por timestep) aprovechando kernels CUDA específicos de tu GPU. Output incluye desglose de tiempos por componente.

**Como valor orientativo, medido en GPU H200 conectada a DGX, la generación de cada muestra puede tomar aproximadamente 0.341s**

---

## Análisis de Performance

### Comparación de Resultados (referencia medida en GPU H200 conectada a DGX)

| Configuración | Tiempo (s) | Speedup | Memoria | Portabilidad |
|---------------|------------|---------|---------|--------------|
| PyTorch FP32 | 0.898 | 1.0x | Alta | Total |
| PyTorch FP16 | 0.545 | 1.5x | Media | Total |
| ONNX Base | 0.606 | 1.5x | Media | Total |
| ONNX Optimizado | 0.439 | 2.0x | Media | Total |
| TensorRT | 0.341 | 2.6x | Media | Limitada* |

\* TensorRT requiere recompilación para GPUs diferentes

## Conclusiones

Este proyecto demuestra el espectro de optimizaciones para modelos de difusión: desde precisión reducida (fp32 a fp16), pasando por optimizaciones portátiles con ONNX (equilibrio rendimiento-flexibilidad), hasta compilación específica con TensorRT (máximo rendimiento, menor flexibilidad).

La elección debe considerar velocidad, facilidad de deployment, mantenimiento y requisitos de portabilidad del proyecto.
