import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert, fuse_modules

def try_fuse(model):
    """Recursively fuse Conv + BN + ReLU where possible."""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Sequential):
            try_fuse(module)
        elif isinstance(module, torch.nn.Conv2d):
            children = list(model.named_children())
            idx = [i for i, (n, _) in enumerate(children) if n == name][0]
            fuse_list = [name]
            if idx + 1 < len(children):
                next_name, next_mod = children[idx + 1]
                if isinstance(next_mod, torch.nn.BatchNorm2d):
                    fuse_list.append(next_name)
                    if idx + 2 < len(children):
                        next2_name, next2_mod = children[idx + 2]
                        if isinstance(next2_mod, torch.nn.ReLU):
                            fuse_list.append(next2_name)
            if len(fuse_list) > 1:
                try:
                    fuse_modules(model, fuse_list, inplace=True)
                except Exception:
                    pass
        else:
            try_fuse(module)
    return model

def quantize_model(model, example_input, backend="fbgemm"):
    """Fuse and quantize model for minimal storage."""
    model.cpu().eval()
    torch.backends.quantized.engine = backend

    # 1. Fuse layers
    model = try_fuse(model)

    # 2. Static quantization
    model.qconfig = get_default_qconfig(backend)
    prepared_model = prepare(model)
    quantized_model = convert(prepared_model)

    # 3. Dynamic quantization for Linear layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        quantized_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # 4. Trace and return
    traced_model = torch.jit.trace(quantized_model, example_input)
    return traced_model

def save_quantized(model, example_input, path):
    traced_model = quantize_model(model, example_input)
    traced_model.save(path)
    print(f"[INFO] Quantized model saved at {path}")

def load_quantized(path, device="cpu"):
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model

