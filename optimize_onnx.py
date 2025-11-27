"""
Optimización de modelos ONNX de Stable Diffusion usando onnxruntime.transformers.

Este script aplica optimizaciones específicas a los componentes de Stable Diffusion
exportados a formato ONNX, mejorando significativamente su rendimiento en inferencia.

1. Carga cada componente ONNX (UNet, Text Encoder, VAE)
2. Aplica fusiones específicas según tipo de arquitectura
3. Guarda modelos optimizados manteniendo compatibilidad con el pipeline original

Uso: python optimize_onnx.py --input stable_diffusion_onnx --output stable_diffusion_onnx_opt
donde:
    --input: Directorio con modelos ONNX exportados (default: stable_diffusion_onnx)
    --output: Directorio para guardar modelos optimizados (default: stable_diffusion_onnx_opt)
    
Salida: Directorio con modelo ONNX optimizado
"""

import argparse
import shutil
import torch
from pathlib import Path
from optimum.onnxruntime import ORTStableDiffusionPipeline
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions


def gpu_with_most_free_mem():
    """
    Encuentra la GPU con más memoria libre disponible.
    
    Útil en sistemas multi-GPU para evitar conflictos de memoria
    y aprovechar la GPU menos cargada.
    
    Returns:
        int or None: Índice de la GPU con más memoria libre, o None si no hay CUDA
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


def main():
    # ============================================================================
    # CONFIGURACIÓN Y ARGUMENTOS
    # ============================================================================
    parser = argparse.ArgumentParser(
        description="Optimizar modelos ONNX de Stable Diffusion con fusiones específicas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python optimize_onnx.py
  python optimize_onnx.py --input stable_diffusion_onnx --output sd_optimized
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        default="stable_diffusion_onnx",
        help="Directorio con modelos ONNX de entrada (default: stable_diffusion_onnx)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stable_diffusion_onnx_opt",
        help="Directorio para modelos optimizados (default: stable_diffusion_onnx_opt)"
    )
    
    args = parser.parse_args()
    
    # ============================================================================
    # INICIALIZACIÓN
    # ============================================================================
    # Seleccionar GPU con más memoria libre para el proceso de optimización
    gpu_id = gpu_with_most_free_mem()
    print(f"🎯 Usando GPU {gpu_id}" if gpu_id is not None else "⚠️  Sin CUDA disponible")
    
    # Configurar directorios de entrada y salida
    onnx_sd_dir = Path(args.input)
    optimized_dir = Path(args.output)
    optimized_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # DEFINICIÓN DE COMPONENTES A OPTIMIZAR
    # ============================================================================
    # Mapeo: nombre_componente -> tipo_arquitectura
    # El tipo de arquitectura determina qué optimizaciones se aplican
    components = {
        "unet": "unet",  # Componente más crítico (70-80% del tiempo de inferencia)
    }
    # Nota: También se pueden optimizar "text_encoder" (bert) y "vae" (vae),
    # pero la UNet es la más crítica para rendimiento en Stable Diffusion

    # ============================================================================
    # OPTIMIZACIÓN DE CADA COMPONENTE
    # ============================================================================
    for component_name, model_type in components.items():
        # Verificar que el componente existe
        src_dir = onnx_sd_dir / component_name
        if not src_dir.exists():
            print(f"⚠️  Componente {component_name} no encontrado, saltando...")
            continue
        
        # Buscar el archivo .onnx dentro del directorio del componente
        onnx_files = list(src_dir.glob("*.onnx"))
        if not onnx_files:
            print(f"⚠️  No se encontró archivo .onnx en {src_dir}, saltando...")
            continue
        
        src_model = onnx_files[0]
        
        # Preparar directorio de salida para el componente optimizado
        dst_dir = optimized_dir / component_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_model = dst_dir / "model.onnx"
        
        print(f"🔧 Optimizando {component_name} ({model_type})...")
        
        try:
            # ====================================================================
            # APLICAR OPTIMIZACIONES
            # ====================================================================
            # Configurar opciones de fusión según tipo de modelo
            # Define qué patrones fusionar: attention, layer_norm, gelu, skip_layer_norm, etc.
            # Cada tipo (unet/bert/vae) tiene optimizaciones específicas
            fusion_options = FusionOptions(model_type)
            
            # Optimizar modelo ONNX con fusiones específicas
            # Proceso:
            #   1. Analiza el grafo ONNX identificando patrones optimizables
            #   2. Aplica fusiones específicas (MultiHeadAttention, LayerNorm, etc.)
            #   3. Elimina nodos redundantes (dead code elimination)
            #   4. Pre-calcula constantes (constant folding)
            opt = optimizer.optimize_model(
                str(src_model),
                model_type=model_type,          # Tipo de arquitectura (unet, bert, vae)
                num_heads=0,                    # 0 = auto-detectar número de attention heads
                hidden_size=0,                  # 0 = auto-detectar dimensión oculta
                optimization_options=fusion_options,  # Opciones de fusión configuradas
                use_gpu=True,                   # Optimizar para GPU (kernels CUDA, permite float16)
            )
            # Ejemplo de optimización aplicada:
            # Antes: [MatMul(Q) + MatMul(K) + MatMul(V) + Softmax + ...] (8+ operaciones)
            # Después: [MultiHeadAttention] (1 operación fusionada)
            # Resultado: ~25% más rápido, menos overhead de kernel launches
         
            # Guardar modelo optimizado
            opt.save_model_to_file(str(dst_model))
            print(f"  ✅ Optimizado → {dst_model}\n")
            
        except Exception as e:
            # Si falla la optimización, usar modelo original como fallback
            print(f"⚠️  Error optimizando {component_name}: {e}")
            print(f"   Copiando modelo original como fallback...")
            shutil.copy2(src_model, dst_model)
        
        # ====================================================================
        # COPIAR ARCHIVOS AUXILIARES DEL COMPONENTE
        # ====================================================================
        # Copiar configuraciones, weights_only, etc. (todo excepto .onnx)
        for item in src_dir.iterdir():
            if item.suffix != ".onnx":
                dst = dst_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst)

    # ============================================================================
    # COPIAR ARCHIVOS GLOBALES DEL PIPELINE
    # ============================================================================
    # Copiar archivos que no son componentes específicos:
    # - tokenizer/ (vocabulario y configuración del tokenizer)
    # - scheduler/ (configuración del scheduler de difusión)
    # - model_index.json (metadatos del pipeline)
    # - feature_extractor/ (si existe)
    for item in onnx_sd_dir.iterdir():
        if item.name in components:
            continue  # Ya procesado arriba
        dst = optimized_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    # ============================================================================
    # VERIFICACIÓN Y RESUMEN
    # ============================================================================
    print("="*60)
    print("✅ Optimización completa")
    print("="*60)
    print(f"📁 Modelos optimizados: {optimized_dir}")
    print()

    # ============================================================================
    # VALIDACIÓN: PROBAR QUE EL PIPELINE OPTIMIZADO CARGA CORRECTAMENTE
    # ============================================================================
    print("🧪 Probando pipeline optimizado...")
    
    # Configurar execution provider según disponibilidad de GPU
    provider = "CUDAExecutionProvider" if gpu_id is not None else "CPUExecutionProvider"
    provider_options = {"device_id": gpu_id} if gpu_id is not None else None

    try:
        # Intentar cargar el pipeline completo con los modelos optimizados
        # Esto valida que:
        #   1. Los archivos ONNX son válidos
        #   2. La estructura del directorio es correcta
        #   3. Las configuraciones son compatibles
        pipe = ORTStableDiffusionPipeline.from_pretrained(
            optimized_dir,
            provider=provider,
            provider_options=provider_options,
        )
        print("✅ Pipeline cargado correctamente")
        print("   Puedes usar este pipeline con measure_trt.py:")
        print(f"   python measure_trt.py --backend onnxrt --onnx_dir {optimized_dir}")
    except Exception as e:
        print(f"❌ Error cargando pipeline: {e}")
        print("   Revisa los logs arriba para identificar el problema")

    # ============================================================================
    # FINALIZACIÓN
    # ============================================================================
    print("\n" + "="*60)
    print("🎉 Proceso de optimización completado")
    print("="*60)


if __name__ == "__main__":
    main()