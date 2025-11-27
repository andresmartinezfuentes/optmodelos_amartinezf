# ============================================================================
# Dockerfile para Optimización de Stable Diffusion
# ============================================================================
# Imagen base: PyTorch 24.08 con CUDA, cuDNN, y TensorRT preinstalados
# Incluye: Python 3.10, PyTorch 2.4, TensorRT 10.x, CUDA 12.5
# ============================================================================
FROM nvcr.io/nvidia/pytorch:24.08-py3

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================
WORKDIR /app

# Variables de entorno para Hugging Face cache
ENV HF_HOME=/app/.cache/huggingface
#ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# ============================================================================
# INSTALACIÓN DE DEPENDENCIAS
# ============================================================================

# Actualizar pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PASO 1: Instalar todas las dependencias sin restricción de numpy
# (Deja que pip resuelva dependencias libremente)
RUN python -m pip install --no-cache-dir \
    "diffusers[torch]>=0.30.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "optimum[onnxruntime-gpu]>=1.21.0" \
    "onnxruntime-gpu>=1.18.0" \
    onnx \
    opencv-python-headless


# Instalar pycuda desde pip
RUN python -m pip uninstall -y pycuda || true && \
    python -m pip install --no-cache-dir pycuda
    
# PASO 2: Forzar numpy<2.0 al final (sobrescribe cualquier versión instalada)
# Razón: PyCUDA (preinstalado) y TensorRT Python bindings requieren NumPy 1.x
RUN python -m pip install --no-cache-dir --force-reinstall "numpy<2.0"

# ============================================================================
# COPIAR ARCHIVOS DE LA APLICACIÓN
# ============================================================================

# Scripts Python
COPY export_onnx.py measure_trt.py optimize_onnx.py \
     trt_inference.py TRTUNetWrapper.py \
     /app/

# Script de generación de TensorRT engine
COPY run_trtexec_auto.sh /usr/local/bin/run_trtexec_auto.sh
RUN chmod +x /usr/local/bin/run_trtexec_auto.sh

# ============================================================================
# VALIDACIÓN DE DEPENDENCIAS
# ============================================================================
# Nota: PyCUDA se valida en runtime

RUN mkdir -p /app/.cache/huggingface

RUN echo "=== Verificando instalación ===" && \
    python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}'); assert numpy.__version__.startswith('1.'), 'NumPy debe ser 1.x'" && \
    python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    python -c "import tensorrt; print(f'✅ TensorRT: {tensorrt.__version__}')" && \
    python -c "import onnxruntime; print(f'✅ ONNX Runtime: {onnxruntime.__version__}')" && \
    echo "✅ Todas las dependencias instaladas correctamente" && \
    echo "" && \
    echo "⚠️  Nota: PyCUDA se validará en runtime (requiere --gpus)" && \
    echo "   Comando de prueba: docker run --gpus all <imagen> python -c 'import pycuda.driver as cuda; cuda.init(); print(f\"GPUs: {cuda.Device.count()}\")'"

# Punto de entrada por defecto
CMD ["bash"]