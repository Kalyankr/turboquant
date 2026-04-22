"""tqtorch — GPU-accelerated TurboQuant for embeddings & RAG."""

from tqtorch.core.mse_quantizer import MSEQuantizer, mse_quantize, mse_dequantize
from tqtorch.core.prod_quantizer import InnerProductQuantizer, ip_quantize, estimate_inner_product
from tqtorch.search.index import TurboQuantIndex

__version__ = "0.1.0"
__all__ = [
    "MSEQuantizer",
    "mse_quantize",
    "mse_dequantize",
    "InnerProductQuantizer",
    "ip_quantize",
    "estimate_inner_product",
    "TurboQuantIndex",
]
