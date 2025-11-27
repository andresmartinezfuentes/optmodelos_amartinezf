#!/usr/bin/env python3
"""
Benchmarking de Stable Diffusion con múltiples backends de optimización.

Compara el rendimiento de tres backends:
  - PyTorch: Baseline sin optimizaciones
  - ONNX Runtime: Optimizaciones a nivel de grafo
  - TensorRT: Optimizaciones máximas para GPU NVIDIA

Uso:
python measure_trt.py \
  --prompt "texto descriptivo"           # Descripción de la imagen
  --backend [torch|onnxrt|trt]           # Backend a usar
  --steps 25                             # Pasos de muestreo en difusión (↑ = mejor calidad, ↓ = más rápido)
  --height 512 --width 512               # Resolución (mantener en 512x512)
  --guidance_scale 7.5                   # Escala de guiado del prompt (mantener en 7.5)
  --sampler dpmpp_2m \                   # Algoritmo de muestreo (mantener en ddmpp_2m)
  --seed 42                              # Reproducibilidad
  --runs 10                              # Numero de repeticiones de muestreo para calcular estadísticas
  --warmup 3                             # Iteraciones de calentamiento
  --precision [fp16|fp32]                # Probar ambas en PyTorch, utilizar fp16 en versiones optimizadas en ONNX y TRT
  --onnx_dir stable_diffusion_onnx_opt   # Solo cuando usamos inferencia con ONNX Runtime (ruta .onnx)
  --engine unet_fp16.engine              # Solo cuando usamos inferencia con TensorRT (ruta .engine)
  --output outputs/imagen.png            # Ruta de salida para la imagen generada (metricas mismo nombre con extension .txt)
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from optimum.onnxruntime import ORTStableDiffusionPipeline
from PIL import Image

# Wrapper para UNet con TensorRT
from TRTUNetWrapper import TRTUNetWrapper

# Funciones para inferencia con TensorRT en modelo Stable Diffusion
from trt_inference import run_trt_inference

# Compatibilidad con NumPy >= 2.0
if not hasattr(np, "bool"):
    np.bool = np.bool_

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Samplers/Schedulers disponibles para el proceso de difusión
SAMPLERS = {
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
}

# ============================================================================
# CLASES DE DATOS
# ============================================================================

@dataclass
class ProfilingStats:
    """
    Estadísticas de tiempo por componente del pipeline.
    
    Registra cuánto tiempo toma cada parte del proceso de generación:
      - text_encoder: CLIP (codificar el prompt)
      - unet: Proceso de difusión (denoising)
      - vae_decoder: Decodificar latents a imagen

    Nota: solo implementamos profiling detallado para inferencia con TensorRT,
    en el resto (PyTorch, ONNX Runtime) medimos tiempo total sin desglose.
    Si se desea, implementar profiling detallado en esos backends tambien
    """
    text_encoder_time: float = 0.0
    unet_time: float = 0.0
    vae_decoder_time: float = 0.0
    
    @property
    def total_time(self) -> float:
        """Tiempo total sumando todos los componentes."""
        return self.text_encoder_time + self.unet_time + self.vae_decoder_time


@dataclass
class BenchmarkConfig:
    """
    Configuración completa para ejecutar un benchmark.
    
    Encapsula todos los parámetros necesarios para generar una imagen
    y medir su rendimiento de forma reproducible.
    """
    prompt: str
    model_id: str
    steps: int
    height: int
    width: int
    guidance_scale: float
    sampler: str
    seed: int
    backend: str
    runs: int
    warmup: int
    use_fp16: bool = True


# ============================================================================
# UTILIDADES
# ============================================================================

def gpu_with_most_free_mem():
    """
    Encuentra la GPU con más memoria libre.
    
    Returns:
        int or None: Índice de la GPU con más memoria, o None si no hay CUDA
    """
    if not torch.cuda.is_available():
        return None
    
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free, _ = torch.cuda.mem_get_info()
        if free > best_free:
            best_free, best_idx = free, i
    
    return best_idx


def set_seed(seed: int):
    """
    Configura todas las semillas para reproducibilidad.
    
    Args:
        seed: Valor entero para inicializar RNGs
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# CARGA DE MODELOS
# ============================================================================

def load_pipeline(
    model_id: str, 
    sampler: str, 
    device: torch.device, 
    gpu_id: Optional[int] = None,
    use_fp16: bool = True
) -> StableDiffusionPipeline:
    """
    Cargar pipeline de Stable Diffusion con PyTorch.
    
    Args:
        model_id: ID del modelo en Hugging Face
        sampler: Tipo de scheduler a usar
        device: Dispositivo donde cargar el modelo
        gpu_id: ID específico de GPU (opcional)
        use_fp16: Usar float16 en vez de float32
    
    Returns:
        StableDiffusionPipeline configurado
    """
    print(f"Cargando modelo {model_id}...")
    
    if use_fp16 and device.type == "cuda":
        torch_dtype = torch.float16
        print("  Usando precisión: float16")
    else:
        torch_dtype = torch.float32
        print("  Usando precisión: float32")
    
    if gpu_id is not None and device.type == "cuda":
        device = torch.device(f"cuda:{gpu_id}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype
    ).to(device)
    
    pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)
    
    return pipe


# ============================================================================
# FUNCIONES DE INFERENCIA POR BACKEND
# ============================================================================

def run_torch_backend(
    config: BenchmarkConfig, 
    pipe: StableDiffusionPipeline
) -> Tuple[Image.Image, float, ProfilingStats]:
    """
    Ejecutar inferencia con backend PyTorch (baseline).
    
    Args:
        config: Configuración del benchmark
        pipe: Pipeline de Stable Diffusion
    
    Returns:
        Tuple de (imagen, tiempo_total, stats_vacias)
    """
    set_seed(config.seed)
    
    t0 = time.perf_counter()
    out = pipe(
        config.prompt,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
    )
    elapsed = time.perf_counter() - t0
    
    stats = ProfilingStats()
    
    return out.images[0], elapsed, stats


def run_onnx_backend(
    config: BenchmarkConfig, 
    pipe: ORTStableDiffusionPipeline
) -> Tuple[Image.Image, float, ProfilingStats]:
    """
    Ejecutar inferencia con backend ONNX Runtime.
    
    Args:
        config: Configuración del benchmark
        pipe: Pipeline ONNX de Stable Diffusion
    
    Returns:
        Tuple de (imagen, tiempo_total, stats_vacias)
    """
    set_seed(config.seed)
    
    t0 = time.perf_counter()
    out = pipe(
        config.prompt,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
    )
    elapsed = time.perf_counter() - t0
    
    stats = ProfilingStats()
    
    return out.images[0], elapsed, stats


def run_trt_backend(
    config: BenchmarkConfig, 
    trt_unet: TRTUNetWrapper, 
    pipe: StableDiffusionPipeline, 
    device: torch.device
) -> Tuple[Image.Image, float, ProfilingStats]:
    """
    Ejecutar inferencia con backend TensorRT.
    
    Wrapper simple que llama a la función de inferencia en trt_inference.py
    y adapta el resultado al formato esperado.
    
    Args:
        config: Configuración del benchmark
        trt_unet: Wrapper del motor TensorRT para UNet
        pipe: Pipeline PyTorch (para text encoder y VAE)
        device: Dispositivo donde ejecutar
    
    Returns:
        Tuple de (imagen, tiempo_total, stats_profiling)
    """
    set_seed(config.seed)
    
    # Llamar a la función de inferencia en módulo separado
    image, timings = run_trt_inference(
        prompt=config.prompt,
        steps=config.steps,
        height=config.height,
        width=config.width,
        guidance_scale=config.guidance_scale,
        trt_unet=trt_unet,
        pipe=pipe,
        device=device
    )
    
    # Convertir diccionario de timings a ProfilingStats
    stats = ProfilingStats(
        text_encoder_time=timings['text_encoder'],
        unet_time=timings['unet'],
        vae_decoder_time=timings['vae_decoder']
    )
    
    return image, timings['total'], stats


# ============================================================================
# EJECUCIÓN DE BENCHMARK
# ============================================================================

def run_benchmark(
    config: BenchmarkConfig, 
    device: torch.device, 
    engine_path: Optional[str] = None, 
    onnx_dir: Optional[str] = None
):
    """
    Ejecutar benchmark completo con el backend seleccionado.
    
    Args:
        config: Configuración del benchmark
        device: Dispositivo base
        engine_path: Ruta al motor TensorRT (solo para backend trt)
        onnx_dir: Directorio con modelo ONNX (solo para backend onnxrt)
    
    Returns:
        Tuple de (lista_tiempos, lista_stats, imagen_final)
    """
    # Seleccionar GPU con más memoria libre
    gpu_id = gpu_with_most_free_mem()
    if gpu_id is not None:
        print(f"🎯 Usando GPU {gpu_id} (mayor memoria libre)")
        device = torch.device(f"cuda:{gpu_id}")
        
        if config.backend == "trt":
            import pycuda.driver as cuda_drv
            cuda_drv.init()
            cuda_device = cuda_drv.Device(gpu_id)
            cuda_context = cuda_device.make_context()
    else:
        print("⚠️  Sin CUDA disponible, usando CPU")
    
    pipe = None
    trt_unet = None
    
    try:
        # ====================================================================
        # TODO : CARGA DE MODELOS POR BACKEND
        # =====================================
        # Implementa la lógica para cargar el modelo según el config.backend seleccionado.
        #
        # Casos a implementar:
        #         
        if config.backend == "torch":
        # A) Backend "torch":
        #    - Cargar pipeline completo usando load_pipeline()
        #    - Pasar: model_id, sampler, device, gpu_id, config.usefp16
        # 
        # ***** TO DO *****
            print("Cargando pipeline PyTorch (baseline)...")
            pipe = load_pipeline(
                model_id=config.model_id,
                sampler=config.sampler,
                device=device,
                gpu_id=gpu_id,
                use_fp16=config.use_fp16
            )

        elif config.backend == "trt":
        # B) Backend "trt":
        #    - Cargar motor TensorRT: trt_unet = TRTUNetWrapper(engine_path)
        #    - Cargar pipeline PyTorch para text_encoder y VAE usando load_pipeline()
        #    - Forzar use_fp16=True en TensorRT (forzar último parámetro)
        #
        # ***** TO DO *****
            print("Cargando pipeline PyTorch (solo text_encoder + VAE) para integracion con TRT...")
            pipe = load_pipeline(
                model_id=config.model_id,
                sampler=config.sampler,
                device=device,
                gpu_id=gpu_id,
                use_fp16=True  # forzamos fp16 para compatibilidad con el motor TRT
            )

            # Si quieres, desligar el UNet original del pipeline para evitar uso accidental.
            try:
                pipe.unet = None
            except Exception:
                # Si por compatibilidad no se puede, simplemente continuamos; run_trt_backend ignora pipe.unet.
                pass


        elif config.backend == "onnxrt":
        # C) Backend "onnxrt":
        #    - Determinar execution provider:
        #      * Si gpu_id no es None → "CUDAExecutionProvider"
        #      * Si no → "CPUExecutionProvider"
        #      provider = "CUDAExecutionProvider" if gpu_id is not None else "CPUExecutionProvider"
        #    - Si CUDA, crear provider_options = {"device_id": gpu_id}, sino None
        #      provider_options = {"device_id": gpu_id} if gpu_id is not None else None
        #     - Cargar con ORTStableDiffusionPipeline.from_pretrained()
        #      pasando onnx_dir, provider y provider_options
        # ***** TO DO *****

    

        # ====================================================================
        # FASE DE WARMUP (calentar GPU y compilar kernels)
        # ====================================================================
        # Las primeras ejecuciones en GPU incluyen overheads de:
        #   - Compilación JIT de kernels CUDA
        #   - Inicialización de cuDNN
        #   - Allocación de buffers internos
        # Ejecutar warmup garantiza que las mediciones reflejen rendimiento real
        if config.warmup > 0:
            print(f"\nEjecutando {config.warmup} iteración(es) de warmup...")
            for _ in range(config.warmup):
                if config.backend == "torch":
                    run_torch_backend(config, pipe)
                elif config.backend == "trt":
                    run_trt_backend(config, trt_unet, pipe, device)
                else:
                    run_onnx_backend(config, pipe)

        # ====================================================================
        # EJECUCIONES CRONOMETRADAS (mediciones de performance)
        # ====================================================================
        times = []                    # Tiempos totales de cada run
        profiling_stats_list = []     # Stats detalladas por componente (solo TRT)
        final_img = None              # Última imagen generada (para guardar)

        print(f"\nEjecutando {config.runs} iteración(es) cronometradas...")
        for i in range(config.runs):
            print(f"\nRun {i+1}/{config.runs} (backend={config.backend})")
            
            # Ejecutar inferencia según backend
            if config.backend == "torch":
                img, t, stats = run_torch_backend(config, pipe)
            elif config.backend == "trt":
                img, t, stats = run_trt_backend(config, trt_unet, pipe, device)
            else:
                img, t, stats = run_onnx_backend(config, pipe)
            
            # Recopilar métricas
            times.append(t)
            profiling_stats_list.append(stats)
            final_img = img
            
            # Mostrar resultados del run actual
            print(f"  Tiempo: {t:.3f}s")
            
            # Mostrar desglose por componente si está disponible (solo TRT)
            if stats.total_time > 0:
                print(f"    ├─ Text Encoder: {stats.text_encoder_time:.3f}s ({stats.text_encoder_time/t*100:.1f}%)")
                print(f"    ├─ UNet: {stats.unet_time:.3f}s ({stats.unet_time/t*100:.1f}%)")
                print(f"    └─ VAE Decoder: {stats.vae_decoder_time:.3f}s ({stats.vae_decoder_time/t*100:.1f}%)")

        return times, profiling_stats_list, final_img
    
    finally:
        if trt_unet:
            trt_unet.cleanup()
        
        if config.backend == "trt" and gpu_id is not None:
            try:
                cuda_context.pop()
            except:
                pass


# ============================================================================
# GUARDADO DE RESULTADOS
# ============================================================================

def save_results(
    config: BenchmarkConfig, 
    times: list, 
    profiling_stats_list: list, 
    image: Image.Image, 
    output_path: str
):
    """
    Guardar imagen generada y métricas del benchmark.
    
    Args:
        config: Configuración usada
        times: Lista de tiempos de cada run
        profiling_stats_list: Lista de stats de profiling
        image: Imagen generada
        output_path: Ruta donde guardar
    """
    avg_time = sum(times) / len(times)
    
    has_profiling = any(s.total_time > 0 for s in profiling_stats_list)
    if has_profiling:
        avg_text_encoder = sum(s.text_encoder_time for s in profiling_stats_list) / len(profiling_stats_list)
        avg_unet = sum(s.unet_time for s in profiling_stats_list) / len(profiling_stats_list)
        avg_vae = sum(s.vae_decoder_time for s in profiling_stats_list) / len(profiling_stats_list)
    
    print(f"\n{'='*60}")
    print(f"Resumen de Resultados:")
    print(f"{'='*60}")
    print(f"Backend: {config.backend}")
    print(f"Tiempos individuales: {[f'{t:.3f}s' for t in times]}")
    print(f"Promedio: {avg_time:.3f}s | Mínimo: {min(times):.3f}s | Máximo: {max(times):.3f}s")
    
    if has_profiling:
        print(f"\nDesglose por Componente (promedio):")
        print(f"  Text Encoder: {avg_text_encoder:.3f}s ({avg_text_encoder/avg_time*100:.1f}%)")
        print(f"  UNet:         {avg_unet:.3f}s ({avg_unet/avg_time*100:.1f}%)")
        print(f"  VAE Decoder:  {avg_vae:.3f}s ({avg_vae/avg_time*100:.1f}%)")
    
    print(f"{'='*60}")

    image.save(output_path)
    print(f"\nImagen guardada en: {output_path}")

    metrics_path = Path(output_path).with_suffix(".metrics.txt")
    with metrics_path.open("w") as f:
        f.write(f"# Configuracion\n")
        f.write(f"prompt: {config.prompt}\n")
        f.write(f"backend: {config.backend}\n")
        f.write(f"model_id: {config.model_id}\n")
        f.write(f"steps: {config.steps}\n")
        f.write(f"size: {config.height}x{config.width}\n")
        f.write(f"guidance_scale: {config.guidance_scale}\n")
        f.write(f"sampler: {config.sampler}\n")
        f.write(f"seed: {config.seed}\n")
        f.write(f"use_fp16: {config.use_fp16}\n")
        
        f.write(f"\n# Ejecucion\n")
        f.write(f"warmup_runs: {config.warmup}\n")
        f.write(f"timed_runs: {config.runs}\n")
        
        f.write(f"\n# Resultados\n")
        f.write(f"times: {times}\n")
        f.write(f"avg_time: {avg_time:.6f}\n")
        f.write(f"min_time: {min(times):.6f}\n")
        f.write(f"max_time: {max(times):.6f}\n")
        
        if has_profiling:
            f.write(f"\n# Profiling por Componente (promedio)\n")
            f.write(f"avg_text_encoder_time: {avg_text_encoder:.6f}\n")
            f.write(f"avg_unet_time: {avg_unet:.6f}\n")
            f.write(f"avg_vae_decoder_time: {avg_vae:.6f}\n")
            f.write(f"text_encoder_pct: {avg_text_encoder/avg_time*100:.2f}\n")
            f.write(f"unet_pct: {avg_unet/avg_time*100:.2f}\n")
            f.write(f"vae_decoder_pct: {avg_vae/avg_time*100:.2f}\n")
            
            f.write(f"\n# Tiempos por Run\n")
            for i, stats in enumerate(profiling_stats_list, 1):
                f.write(f"run_{i}_text_encoder: {stats.text_encoder_time:.6f}\n")
                f.write(f"run_{i}_unet: {stats.unet_time:.6f}\n")
                f.write(f"run_{i}_vae_decoder: {stats.vae_decoder_time:.6f}\n")
    
    print(f"Métricas guardadas en: {metrics_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de Stable Diffusion con múltiples backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # PyTorch baseline
  python measure_trt.py --prompt "astronaut" --backend torch --steps 25 --precision fp16
  
  # ONNX Runtime
  python measure_trt.py --prompt "astronaut" --backend onnxrt \\
      --onnx_dir stable_diffusion_onnx --steps 25
  
  # TensorRT
  python measure_trt.py --prompt "astronaut" --backend trt \\
      --engine unet_fp16.engine --steps 25
        """
    )
    
    parser.add_argument("--prompt", type=str, required=True, help="Descripción de la imagen")
    parser.add_argument("--backend", type=str, choices=["torch", "trt", "onnxrt"], default="torch")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--engine", type=str, help="Ruta al motor TensorRT")
    parser.add_argument("--onnx_dir", type=str, help="Directorio con modelo ONNX")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sampler", type=str, choices=list(SAMPLERS.keys()), default="pndm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=str, default="outputs/sd_output.png")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16")
    
    args = parser.parse_args()

    if args.backend == "trt" and not args.engine:
        parser.error("--engine es requerido cuando backend=trt")
    if args.backend == "onnxrt" and not args.onnx_dir:
        parser.error("--onnx_dir es requerido cuando backend=onnxrt")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo disponible: {device}")

    config = BenchmarkConfig(
        prompt=args.prompt,
        model_id=args.model_id,
        steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        seed=args.seed,
        backend=args.backend,
        runs=args.runs,
        warmup=args.warmup,
        use_fp16=(args.precision == "fp16"),
    )

    times, profiling_stats_list, image = run_benchmark(
        config, 
        device, 
        args.engine, 
        args.onnx_dir
    )
    
    save_results(config, times, profiling_stats_list, image, args.output)


if __name__ == "__main__":
    main()
