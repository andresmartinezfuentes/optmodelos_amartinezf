"""
Funciones de inferencia específicas para el backend TensorRT.

Gestiona el proceso completo de generación de imágenes usando TensorRT
para el UNet, manteniendo text encoder y VAE en PyTorch.
"""

from typing import Tuple

import torch
from PIL import Image
import time

from diffusers import StableDiffusionPipeline
from TRTUNetWrapper import TRTUNetWrapper

# Constantes
DEFAULT_LATENT_CHANNELS = 4  # Canales en espacio latente
VAE_SCALE_FACTOR = 0.18215   # Factor de escala para normalización


def get_text_embeddings(
    pipe: StableDiffusionPipeline, 
    prompt: str, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Obtener embeddings de texto del prompt usando CLIP.
    
    Genera dos tipos de embeddings:
      - text_emb: Condicional (basado en el prompt)
      - uncond_emb: No condicional (prompt vacío)
    
    Args:
        pipe: Pipeline de Stable Diffusion
        prompt: Descripción de la imagen
        device: Dispositivo donde ejecutar
    
    Returns:
        Tuple de (text_embeddings, uncond_embeddings, tiempo_transcurrido)
    """
    # Tokenizar prompt y prompt vacío
    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_inputs = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # Generar embeddings con el text encoder (CLIP)
    t0 = time.perf_counter()
    with torch.no_grad():
        text_emb = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_emb = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]
    
    # Sincronizar GPU para medición precisa
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - t0
    
    return text_emb, uncond_emb, elapsed


def latents_to_image(
    latents: torch.Tensor, 
    vae, 
    device: torch.device
) -> Tuple[Image.Image, float]:
    """
    Decodificar latents a imagen usando el VAE decoder.
    
    Convierte representación de espacio latente (64x64x4) a imagen RGB (512x512x3).
    
    Args:
        latents: Tensor en espacio latente
        vae: Decodificador VAE
        device: Dispositivo donde ejecutar
    
    Returns:
        Tuple de (imagen_PIL, tiempo_transcurrido)
    """
    # Des-normalizar latents
    latents = latents / VAE_SCALE_FACTOR
    
    # Decodificar con VAE
    t0 = time.perf_counter()
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    # Sincronizar GPU
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - t0
    
    # Post-procesamiento: tensor → imagen PIL
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalizar a [0, 1]
    image = image.cpu().permute(0, 2, 3, 1).numpy()  # BCHW → BHWC
    image = (image * 255).round().astype("uint8")[0]  # A uint8 y remover batch
    
    return Image.fromarray(image), elapsed


def run_trt_inference(
    prompt: str,
    steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    trt_unet: TRTUNetWrapper,
    pipe: StableDiffusionPipeline,
    device: torch.device
) -> Tuple[Image.Image, dict]:
    """
    Ejecutar inferencia completa con backend TensorRT.
    
    Usa motor TensorRT optimizado para el UNet, manteniendo
    text encoder y VAE decoder en PyTorch.
    
    Proceso:
      1. Text encoding (CLIP) → embeddings
      2. Loop de difusión (TensorRT UNet) → latents limpios
      3. VAE decoding → imagen final
    
    Args:
        prompt: Descripción de la imagen a generar
        steps: Número de pasos de difusión
        height: Alto de la imagen en píxeles
        width: Ancho de la imagen en píxeles
        guidance_scale: Fuerza del guidance
        trt_unet: Wrapper del motor TensorRT para UNet
        pipe: Pipeline PyTorch (para text encoder y VAE)
        device: Dispositivo donde ejecutar
    
    Returns:
        Tuple de (imagen_PIL, diccionario_tiempos)
        diccionario_tiempos contiene:
          - 'text_encoder': tiempo de encoding
          - 'unet': tiempo del loop de difusión
          - 'vae_decoder': tiempo de decodificación
          - 'total': tiempo total
    """
    timings = {}

    # 1. Text encoding (CLIP)
    text_emb, uncond_emb, timings['text_encoder'] = get_text_embeddings(
        pipe, prompt, device
    )

    # 2. Inicializar latents con ruido
    in_channels = getattr(pipe.unet.config, "in_channels", DEFAULT_LATENT_CHANNELS)
    latent_shape = (1, in_channels, height // 8, width // 8)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float16)
    latents = latents * pipe.scheduler.init_noise_sigma

    # Configurar scheduler
    pipe.scheduler.set_timesteps(steps, device=device)

    # 3. Loop de difusión (UNet con TensorRT)
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in pipe.scheduler.timesteps:
            # Predicción sin condición (guidance negativo)
            noise_uncond = trt_unet(latents, t, uncond_emb)
            
            # Predicción con condición (guidance positivo)
            noise_text = trt_unet(latents, t, text_emb)
            
            # Classifier-free guidance: combinar ambas predicciones
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            
            # Actualizar latents según el scheduler
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    # Sincronizar GPU
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    timings['unet'] = time.perf_counter() - t0

    # 4. VAE decoding
    image, timings['vae_decoder'] = latents_to_image(latents, pipe.vae, device)
    
    # Calcular tiempo total
    timings['total'] = timings['text_encoder'] + timings['unet'] + timings['vae_decoder']
    
    return image, timings