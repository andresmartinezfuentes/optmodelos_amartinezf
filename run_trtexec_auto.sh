#!/bin/bash
#
# =====================================================================
# RUN_TRTEXEC_AUTO.SH - Compilación automática de ONNX a TensorRT
# =====================================================================
#
# Compila el modelo UNet de Stable Diffusion desde formato ONNX a un
# motor TensorRT altamente optimizado (.engine) para inferencia rápida.
#
# 1. Selección automática de GPU (elige la menos ocupada)
# 2. Configuración de precisión FP16 (Tensor Cores)
# 3. Definición de shapes de entrada para optimización
# 4. Generación de engine optimizado específico para tu GPU
#
# Requiere modelo ONNX exportado previamente en stable_diffusion_onnx/unet/model.onnx
#
# Salida: stable_diffusion_onnx/unet_fp16.engine (~1.7GB)
# (archivo serializado listo para inferencia ultra-rápida)

# Tiempo estimado: 8-10 minutos en H200

# ─────────────────────────────────────────────────────────────────────
# SELECCIÓN AUTOMÁTICA DE GPU
# ─────────────────────────────────────────────────────────────────────
# En sistemas multi-GPU, selecciona automáticamente la GPU con menor
# uso de memoria para evitar interferencias con otros procesos.

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Detectando GPU óptima para compilación..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used \
            --format=csv,noheader,nounits \
            | sort -t, -k2 -n \
            | head -n1 \
            | cut -d',' -f1)

echo "✓ GPU seleccionada: $BEST_GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
           --format=csv,noheader \
           | grep "^$BEST_GPU,"
echo ""

# ─────────────────────────────────────────────────────────────────────
# COMPILACIÓN CON TRTEXEC
# ─────────────────────────────────────────────────────────────────────
# trtexec es la herramienta CLI de TensorRT para:
# 1. Parsear modelos ONNX
# 2. Aplicar optimizaciones (layer fusion, precision selection, etc.)
# 3. Seleccionar kernels CUDA óptimos mediante profiling
# 4. Serializar el engine compilado a disco
#
# Este proceso toma varios minutos pero solo se hace UNA VEZ.
# El engine resultante es específico para:
# - Tu GPU exacta (H200 en este caso)
# - Las shapes de entrada configuradas
# - La versión de TensorRT utilizada

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Iniciando compilación TensorRT..."
echo "  ⏱️  Tiempo estimado: 8-10 minutos"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

trtexec \
  `# ═══════════════════════════════════════════════════════════════` \
  `# CONFIGURACIÓN DE DISPOSITIVO` \
  `# ═══════════════════════════════════════════════════════════════` \
  --device=$BEST_GPU \
  \
  `# ═══════════════════════════════════════════════════════════════` \
  `# ARCHIVOS DE ENTRADA Y SALIDA` \
  `# ═══════════════════════════════════════════════════════════════` \
  --onnx=stable_diffusion_onnx/unet/model.onnx \
  --saveEngine=stable_diffusion_onnx/unet_fp16.engine \
  \
  `# ═══════════════════════════════════════════════════════════════` \
  `# PRECISIÓN: FP16 (Half Precision)` \
  `# ═══════════════════════════════════════════════════════════════` \
  `# Activa Tensor Cores en GPUs modernas (Volta+)` \
  `# Beneficios:` \
  `#   - 2× throughput computacional` \
  `#   - 50% menos uso de memoria` \
  `#   - Pérdida de precisión imperceptible en generación de imágenes` \
  --fp16 \
  \
  `# ═══════════════════════════════════════════════════════════════` \
  `# CONFIGURACIÓN DE SHAPES DINÁMICOS` \
  `# ═══════════════════════════════════════════════════════════════` \
  `# TensorRT necesita conocer el rango de tamaños posibles de entrada` \
  `# para optimizar correctamente. Especificamos min/opt/max.` \
  `#` \
  `# INPUT 1: sample (latents ruidosos)` \
  `#   Shape: [batch=1, channels=4, height=64, width=64]` \
  `#   Nota: 64×64 latents = 512×512 pixels (factor 8x del VAE)` \
  `#` \
  `# INPUT 2: encoder_hidden_states (text embeddings)` \
  `#   Shape: [batch=1, tokens=77, embedding_dim=768]` \
  `#   Nota: 77 es el max length del tokenizer de CLIP` \
  `#` \
  `# INPUT 3: timestep (paso actual de denoising)` \
  `#   Shape: [1] (escalar)` \
  `#   Rango: [0, num_steps-1], típicamente [0, 999]` \
  --minShapes=sample:1x4x64x64,encoder_hidden_states:1x77x768,timestep:1 \
  --optShapes=sample:1x4x64x64,encoder_hidden_states:1x77x768,timestep:1 \
  --maxShapes=sample:1x4x64x64,encoder_hidden_states:1x77x768,timestep:1 \
  \
  `# ═══════════════════════════════════════════════════════════════` \
  `# OPTIMIZACIONES ADICIONALES` \
  `# ═══════════════════════════════════════════════════════════════` \
  `# --skipInference: No ejecutar inferencia de prueba después de` \
  `# compilar. Esto acelera la compilación ya que solo nos interesa` \
  `# generar el engine, no medir su rendimiento aquí.` \
  --skipInference

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Compilación completada exitosamente"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Engine guardado en: stable_diffusion_onnx/unet_fp16.engine"
echo "📊 Tamaño esperado: ~1.7GB"