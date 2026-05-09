"""HY-Motion 1.0 — text-to-motion generation library.

Fork of Tencent-Hunyuan/HY-Motion-1.0 packaged for pip installation, with
streaming, GGUF, and quantization additions.

Top-level imports stay deliberately minimal so tooling can import the
package without forcing torch/transformers to load — pull individual
submodules as needed:

    from hymotion.utils.t2m_runtime import T2MRuntime
"""

__version__ = "1.1.0"
