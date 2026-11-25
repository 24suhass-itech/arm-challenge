"""Export Llama 3.2 1B-Instruct to ExecuTorch for on-device advice (run with the ExecuTorch environment).

    C:\\Users\\HP\\et-venv\\Scripts\\python.exe advisor\\export_llama.py

1. Downloads meta-llama/Llama-3.2-1B-Instruct (original Meta checkpoint: consolidated.00.pth, params.json,
   tokenizer.model) into advisor/llama/original. Requires accepting the Llama 3.2 licence on Hugging Face and
   `hf auth login` beforehand.
2. Exports with ExecuTorch's export_llm:
     - KV cache + SDPA custom op (fast token-by-token decoding)
     - 8da4w quantisation: 4-bit weights (group size 128), 8-bit dynamic activations; 4-bit embeddings
     - XNNPACK backend = optimised Arm CPU kernels (Android phones, Raspberry Pi 5)
     - 1024-token context (prompt ~450 tokens + answer)
   Output: advisor/llama/llama3_2_1b_instruct_8da4w.pte (~1 GB)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.join(HERE, "llama")
ORIG = os.path.join(LLAMA, "original")
REPO = "meta-llama/Llama-3.2-1B-Instruct"
NAME = "llama3_2_1b_instruct_8da4w.pte"


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, allow_patterns=["original/consolidated.00.pth", "original/params.json",
                                            "original/tokenizer.model"], local_dir=LLAMA)


def build_config(checkpoint, params, out_dir, out_name, max_len=1024, group_size=128):
    """Same settings as ExecuTorch's llama_xnnpack.yaml, built in Python (the Hydra CLI does not run on
    Python 3.14 yet)."""
    from executorch.extension.llm.export.config.llm_config import DtypeOverride, LlmConfig, ModelType
    c = LlmConfig()
    c.base.model_class = ModelType.llama3_2
    c.base.checkpoint = checkpoint
    c.base.params = params
    c.base.metadata = '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
    c.model.use_kv_cache = True
    c.model.use_sdpa_with_kv_cache = True
    c.model.dtype_override = DtypeOverride.fp32
    c.export.max_seq_length = max_len
    c.export.max_context_length = max_len
    c.export.output_dir = out_dir
    c.export.output_name = os.path.join(out_dir, out_name)  # export_llm writes output_name relative to the cwd
    c.quantization.qmode = "8da4w"
    c.quantization.group_size = group_size
    c.quantization.embedding_quantize = "4,32"
    c.backend.xnnpack.enabled = True
    c.backend.xnnpack.extended_ops = True
    return c


def _use_bundled_flatc():
    """On Windows the ExecuTorch wheel ships data/bin/flatc.exe, but its lookup searches for 'flatc' (no .exe)."""
    import executorch
    exe = os.path.join(list(executorch.__path__)[0], "data", "bin", "flatc.exe")
    if os.path.exists(exe):
        os.environ.setdefault("FLATC_EXECUTABLE", exe)


def export(max_len, checkpoint=None, params=None, out_dir=LLAMA, out_name=NAME, group_size=128):
    _use_bundled_flatc()
    from executorch.examples.models.llama.export_llama_lib import export_llama
    cfg = build_config(checkpoint if checkpoint is not None else os.path.join(ORIG, "consolidated.00.pth"),
                       params or os.path.join(ORIG, "params.json"), out_dir, out_name, max_len, group_size)
    return export_llama(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--dry_run_params", help="tiny params.json: export a randomly initialised model to test the toolchain")
    ap.add_argument("--out_dir", default=LLAMA)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.dry_run_params:
        export(256, checkpoint="", params=a.dry_run_params, out_dir=a.out_dir, out_name="tiny.pte", group_size=32)
        out = os.path.join(a.out_dir, "tiny.pte")
    else:
        if not a.skip_download:
            download()
        export(a.max_len)
        out = os.path.join(LLAMA, NAME)
    print(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB)")
