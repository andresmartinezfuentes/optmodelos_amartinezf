"""
Wrapper de TensorRT para el UNet de Stable Diffusion.

Proporciona una interfaz simple para ejecutar inferencia con motores TensorRT
optimizados, manejando toda la gestión de buffers y transferencias GPU.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTUNetWrapper:
    """
    Envoltorio para ejecutar el UNet con TensorRT (máxima optimización GPU).
    
    ¿Qué hace TensorRT?
    -------------------
    Convierte modelos ONNX en "engines" altamente optimizados para una GPU específica.
    Similar a compilar código: el resultado es mucho más rápido que el original.
    
    Optimizaciones principales:
    ---------------------------
    1. Fusión de operaciones: Combina múltiples cálculos en un solo kernel GPU
       Ejemplo: [MatMul + Add + ReLU] → [OperaciónFusionada]
    
    2. Selección de kernels óptimos: Prueba múltiples implementaciones y elige
       la más rápida para tu GPU específica
    
    3. Gestión eficiente de memoria: Minimiza transferencias CPU↔GPU y reutiliza buffers
    
    4. Precision mixing (FP16/FP32): Usa float16 donde es seguro (más rápido)
       y float32 donde se necesita mayor precisión
    
    Gestión de recursos:
    --------------------
    - Carga el motor .engine pre-compilado desde disco
    - Asigna buffers en GPU para entradas y salidas
    - Ejecuta inferencia de forma asíncrona (no bloquea CPU)
    - Maneja transferencias de datos entre CPU y GPU
    
    Speedup típico: 2-3x más rápido que PyTorch u ONNX Runtime
    """
    
    REQUIRED_INPUTS = ["sample", "timestep", "encoder_hidden_states"]
    REQUIRED_OUTPUTS = ["out_sample"]

    def __init__(self, engine_path: str):
        """
        Carga y inicializa el motor TensorRT.
        
        Args:
            engine_path: Ruta al archivo .engine pre-compilado
        """
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Motor TensorRT no encontrado: {engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"Cargando motor TensorRT desde {engine_path}...")
        
        # Deserializar engine desde disco
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Error al deserializar motor TensorRT")

        # Crear contexto de ejecución y stream CUDA
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.ctx = cuda.Context.get_current()
        
        # Diccionarios para almacenar información de tensores
        self.inputs = {}
        self.outputs = {}

        self._initialize_tensors()
        self._validate_tensors()

    def _initialize_tensors(self):
        """Recopilar y categorizar tensores del engine."""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = list(self.engine.get_tensor_shape(name))
            info = {"dtype": dtype, "shape": shape, "host": None, "device": None}
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = info
            else:
                self.outputs[name] = info

    def _validate_tensors(self):
        """Validar que existen todos los tensores requeridos."""
        for name in self.REQUIRED_INPUTS:
            if name not in self.inputs:
                raise ValueError(f"Falta input '{name}'. Disponibles: {list(self.inputs.keys())}")
        
        for name in self.REQUIRED_OUTPUTS:
            if name not in self.outputs:
                raise ValueError(f"Falta output '{name}'. Disponibles: {list(self.outputs.keys())}")

    def _alloc_buffer(self, name: str, shape: Optional[tuple] = None):
        """
        Asignar buffers pinned (host) y device (GPU) para un tensor.
        
        Args:
            name: Nombre del tensor
            shape: Forma del tensor (None = usar forma del engine)
        """
        info = self.inputs.get(name) or self.outputs.get(name)
        shape = shape if shape is not None else info["shape"]
        size = int(np.prod(shape))
        
        # Buffer pinned para transferencias rápidas CPU→GPU
        info["host"] = cuda.pagelocked_empty(size, info["dtype"])
        # Buffer en GPU para computación
        info["device"] = cuda.mem_alloc(info["host"].nbytes)
        info["shape"] = shape
        
        # Registrar dirección del buffer en el contexto TensorRT
        self.context.set_tensor_address(name, int(info["device"]))

    def _set_dynamic_shape(self, name: str, shape: tuple):
        """
        Establecer forma de entrada si el tensor tiene dimensiones dinámicas.
        
        Args:
            name: Nombre del tensor
            shape: Forma actual del tensor
        """
        engine_shape = list(self.engine.get_tensor_shape(name))
        if any(d == -1 for d in engine_shape):
            self.context.set_input_shape(name, shape)

    def _copy_to_device(self, name: str, data: np.ndarray):
        """
        Copiar datos de entrada a GPU.
        
        Args:
            name: Nombre del tensor de entrada
            data: Array NumPy con los datos
        """
        # Asignar buffer si es la primera vez
        if self.inputs[name]["device"] is None:
            self._alloc_buffer(name, data.shape)
        
        # Configurar forma dinámica si aplica
        self._set_dynamic_shape(name, data.shape)
        
        # Copiar: NumPy → buffer pinned → GPU (asíncrono)
        np.copyto(self.inputs[name]["host"].reshape(data.shape), data)
        cuda.memcpy_htod_async(
            self.inputs[name]["device"], 
            self.inputs[name]["host"], 
            self.stream
        )

    def __call__(
        self, 
        sample: torch.Tensor, 
        timestep: torch.Tensor, 
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Ejecutar forward pass del UNet con TensorRT.
        
        Args:
            sample: Latents ruidosos [batch, channels, height, width]
            timestep: Paso de tiempo actual (0-999)
            encoder_hidden_states: Embeddings del prompt [batch, seq_len, dim]
        
        Returns:
            torch.Tensor: Predicción de ruido
        """
        self.ctx.push()
        try:
            # Preparar inputs (PyTorch → NumPy, float16)
            sample_np = sample.detach().cpu().numpy().astype(np.float16)
            encoder_np = encoder_hidden_states.detach().cpu().numpy().astype(np.float16)
            timestep_np = np.array([timestep.item()], dtype=np.float32)

            # Copiar inputs a GPU
            self._copy_to_device("sample", sample_np)
            self._copy_to_device("encoder_hidden_states", encoder_np)
            self._copy_to_device("timestep", timestep_np)

            # Asignar buffer de salida si es necesario
            if self.outputs["out_sample"]["device"] is None:
                self._alloc_buffer("out_sample", sample_np.shape)

            # Ejecutar inferencia asíncrona
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            # Recuperar resultado (GPU → buffer pinned → NumPy)
            out_info = self.outputs["out_sample"]
            cuda.memcpy_dtoh_async(out_info["host"], out_info["device"], self.stream)
            self.stream.synchronize()

            # Convertir a PyTorch tensor
            result = torch.from_numpy(
                out_info["host"].reshape(out_info["shape"])
            ).to(sample.device)
            
            return result
            
        finally:
            self.ctx.pop()

    def cleanup(self):
        """Liberar memoria GPU."""
        for tensor_dict in [self.inputs, self.outputs]:
            for info in tensor_dict.values():
                if info["device"]:
                    info["device"].free()
                info["device"] = None
                info["host"] = None

    def __del__(self):
        """Limpieza automática al destruir el objeto."""
        try:
            self.cleanup()
        except:
            pass