#!/usr/bin/env python3
"""
=====================================================================
EXPORT_ONNX.PY - Exportador de Stable Diffusion a formato ONNX
=====================================================================

Este script convierte un modelo de Stable Diffusion (formato PyTorch) al
formato ONNX (Open Neural Network Exchange), que es un formato intermedio
estándar para representar modelos de deep learning, empleado tambien
como punto de partida para optimización con TensorRT.

1. Descarga el modelo pre-entrenado desde Hugging Face Hub
2. Exporta los componentes (UNet, VAE, Text Encoder) a formato ONNX
3. Guarda los archivos ONNX en disco para uso posterior

USO: python export_onnx.py
SALIDA: Directorio 'stable_diffusion_onnx/' con todos los componentes
=====================================================================
"""

#import os
#import warnings
from pathlib import Path

import torch
from optimum.onnxruntime import ORTStableDiffusionPipeline

# Configuración del modelo
# -----------------------
# model_id: Identificador del modelo en Hugging Face Hub
# Stable Diffusion v1.5 es la versión estable y ampliamente utilizada
model_id = "runwayml/stable-diffusion-v1-5"

# Directorio donde se guardarán los archivos ONNX exportados
onnx_dir = "stable_diffusion_onnx"

# Crear directorio de salida si no existe
onnx_dir_path = Path(onnx_dir)
onnx_dir_path.mkdir(parents=True, exist_ok=True)

# Verificar si ya existe el componente principal (UNet)
# El UNet es el componente más pesado y crítico del pipeline
unet_onnx = onnx_dir_path / "unet" / "model.onnx"

if not unet_onnx.exists():
    print("\n--- Exportando modelo a ONNX ---")
    
    # Cargar y exportar el pipeline completo
    # ---------------------------------------
    # from_pretrained: Descarga el modelo desde Hugging Face Hub
    # export=True: Activa la exportación automática a ONNX
    # torch_dtype=float16: Usa precisión reducida (16 bits) para:
    #   - Reducir tamaño del modelo (50% menos memoria)
    #   - Acelerar inferencia en GPUs modernas (Tensor Cores)
    #   - Mínima pérdida de calidad en la generación
    # provider="CUDAExecutionProvider": Configura ONNX Runtime para usar GPU
    pipe = ORTStableDiffusionPipeline.from_pretrained(
        model_id,
        export=True,
        torch_dtype=torch.float16,
        provider="CUDAExecutionProvider",
    )
    
    # Guardar todos los componentes ONNX en disco
    # Estructura generada:
    # stable_diffusion_onnx/
    #   ├── unet/model.onnx          (~1.7GB) - Red de difusión
    #   ├── vae_encoder/model.onnx   (~150MB) - Codificador de imágenes
    #   ├── vae_decoder/model.onnx   (~150MB) - Decodificador de imágenes
    #   ├── text_encoder/model.onnx  (~500MB) - Encoder de texto (CLIP)
    #   ├── tokenizer/                        - Tokenizador de texto
    #   └── scheduler/scheduler_config.json   - Configuración del sampler
    pipe.save_pretrained(onnx_dir)
    print(f"Modelo ONNX guardado en {onnx_dir}")
else:
    print("\n--- ONNX ya existe, usando carpeta:", onnx_dir, "---")
    print("Si necesitas re-exportar, elimina el directorio y vuelve a ejecutar.")
