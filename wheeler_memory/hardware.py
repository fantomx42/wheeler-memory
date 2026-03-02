
"""Hardware detection module for Wheeler Memory.

Provides capabilities to detect CPU, GPU (NVIDIA & Integrated), NPU, RAM, and Storage
specifications to help debug environment issues and optimize performance.
"""

import platform
import shutil
import subprocess
import psutil

def get_cpu_info() -> dict:
    """Returns CPU specifications."""
    info = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
    }
    try:
        # Linux-specific frequency check
        freq = psutil.cpu_freq()
        if freq:
            info["frequency_mhz"] = {
                "current": freq.current,
                "min": freq.min,
                "max": freq.max
            }
    except Exception:
        pass
    return info

def get_memory_info() -> dict:
    """Returns RAM specifications."""
    svmem = psutil.virtual_memory()
    return {
        "total_gb": round(svmem.total / (1024**3), 2),
        "available_gb": round(svmem.available / (1024**3), 2),
        "used_gb": round(svmem.used / (1024**3), 2),
        "percent_used": svmem.percent
    }

def get_storage_info(path: str = "/") -> dict:
    """Returns storage specifications for the given path."""
    total, used, free = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
        "percent_used": round((used / total) * 100, 1)
    }

def get_gpu_info() -> dict:
    """Returns GPU and NPU information."""
    info = {"nvidia_gpu": None, "pci_devices": []}

    # 1. NVIDIA GPU via nvidia-smi (most reliable for NVIDIA)
    try:
        # Check if nvidia-smi is available
        subprocess.check_call(["which", "nvidia-smi"], stdout=subprocess.DEVNULL)
        
        # Get memory and name
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            encoding="utf-8"
        )
        gpus = []
        for line in output.strip().split('\n'):
            name, total, free = line.split(', ')
            gpus.append({
                "name": name,
                "memory_total": total,
                "memory_free": free
            })
        info["nvidia_gpu"] = gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["nvidia_gpu"] = "Not detected or driver missing"

    # 2. PCI Devices (Integrated Graphics, NPUs, other Accelerators) via lspci
    try:
        # Check if lspci is available (requires pciutils)
        subprocess.check_call(["which", "lspci"], stdout=subprocess.DEVNULL)
        
        output = subprocess.check_output(["lspci"], encoding="utf-8")
        
        relevant_keywords = ["VGA", "3D", "Display", "NPU", "Processing Accelerator", "Intelligence"]
        
        for line in output.split('\n'):
            if any(keyword.lower() in line.lower() for keyword in relevant_keywords):
                info["pci_devices"].append(line.strip())
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["pci_devices"] = "lspci not found or failed"

    return info

def get_optimal_device() -> str:
    """Auto-selects the best available accelerator (cuda/mps/cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

def check_software_hardware_mismatch() -> list[str]:
    """Returns warnings if powerful hardware is detected but unused by software."""
    warnings = []
    gpu_info = get_gpu_info()
    
    # Check for likely discrete GPUs via lspci that Torch might miss (e.g. AMD without ROCm)
    has_discrete_gpu = False
    if isinstance(gpu_info["pci_devices"], list):
        for device in gpu_info["pci_devices"]:
            # Simple heuristic: "VGA" or "3D" usually implies a GPU. 
            # If it's not Intel (often integrated), it might be discrete AMD/NVIDIA.
            device_lower = device.lower()
            if "vga" in device_lower or "3d controller" in device_lower:
                if "nvidia" in device_lower or "amd" in device_lower or "radeon" in device_lower:
                    has_discrete_gpu = True

    try:
        import torch
        torch_device = get_optimal_device()
        
        if has_discrete_gpu and torch_device == "cpu":
            warnings.append(
                "Discrete GPU detected but PyTorch is using CPU. "
                "Verify CUDA/ROCm installation matches your hardware."
            )
    except ImportError:
        warnings.append("PyTorch not installed. Hardware acceleration unavailable.")

    return warnings

def get_system_summary() -> dict:
    """Aggregates all system information."""
    return {
        "os": platform.system(),
        "release": platform.release(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "storage": get_storage_info(),
        "gpu_npu": get_gpu_info(),
        "optimal_device": get_optimal_device(),
        "warnings": check_software_hardware_mismatch()
    }
