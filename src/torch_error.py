"""
Permite importar data-depth sin cargar PyTorch.

data-depth importa torch al cargarse, aunque solo lo utiliza para calcular en
GPU (opción CUDA=True). En Windows, el Control inteligente de aplicaciones
(Smart App Control) puede bloquear las DLL de torch, y entonces la importación
de data-depth falla con

    OSError: [WinError 4551] Una directiva de Control de aplicaciones bloqueó
    este archivo. Error loading "...\\torch\\lib\\c10.dll"

Si torch no se puede importar, este módulo registra en su lugar un módulo
"torch" vacío. Los cálculos en CPU (CUDA=False, el valor por defecto) no lo
usan, así que los resultados de data-depth son exactamente los mismos. Si torch
sí funciona, el módulo no hace nada.
"""

import sys
import types


class _SinTorch:
    """Ocupa el lugar de cualquier clase de torch (Tensor, device, dtype...)."""

    def __init__(self, *args, **kwargs):
        pass


def _no_disponible(*args, **kwargs):
    return False


def _error(*args, **kwargs):
    raise RuntimeError("PyTorch no está disponible: usa CUDA=False.")


def _registrar_torch_vacio():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=_no_disponible)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=_no_disponible))
    torch.manual_seed = _no_disponible
    torch.tensor = _error
    # Cualquier otro nombre (torch.Tensor, torch.device, torch.float32...) se
    # resuelve como una clase vacía, para que las anotaciones de tipo de
    # data-depth se puedan evaluar al importar.
    torch.__getattr__ = lambda nombre: _SinTorch

    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    functional.normalize = _error
    nn.functional = functional
    torch.nn = nn

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional


try:
    import torch  # noqa: F401
except (ImportError, OSError):
    _registrar_torch_vacio()
