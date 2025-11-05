import io, os, lzma, torch
import torch.nn as nn

def _fold_bn_into_affine(affine, bn):
    gamma = bn.weight.detach() if bn.affine else torch.ones_like(bn.running_var)
    beta  = bn.bias.detach()   if bn.affine else torch.zeros_like(bn.running_var)
    mean  = bn.running_mean.detach(); var = bn.running_var.detach(); eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    if isinstance(affine, nn.Linear):
        W = affine.weight.detach()
        b = affine.bias.detach() if affine.bias is not None else torch.zeros_like(mean)
        W.mul_(scale.view(-1, 1)); b = (b - mean) * scale + beta
        affine.weight.data.copy_(W); affine.bias = nn.Parameter(b)
    elif isinstance(affine, nn.Conv2d):
        W = affine.weight.detach()
        b = affine.bias.detach() if affine.bias is not None else torch.zeros_like(mean)
        W.mul_(scale.view(-1, 1, 1, 1)); b = (b - mean) * scale + beta
        affine.weight.data.copy_(W); affine.bias = nn.Parameter(b)
    elif isinstance(affine, nn.ConvTranspose2d):
        W = affine.weight.detach()
        b = affine.bias.detach() if affine.bias is not None else torch.zeros_like(mean)
        # ConvTranspose2d weight: (in_channels, out_channels, kh, kw)
        W.mul_(scale.view(1, -1, 1, 1)); b = (b - mean) * scale + beta
        affine.weight.data.copy_(W); affine.bias = nn.Parameter(b)
    if bn.affine:
        bn.weight.data.fill_(1.0); bn.bias.data.zero_()
    bn.running_mean.data.zero_(); bn.running_var.data.fill_(1.0)

def fold_all_batchnorms(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            for i in range(len(module) - 1):
                a = module[i]; b = module[i + 1]
                if isinstance(b, (nn.BatchNorm1d, nn.BatchNorm2d)) and isinstance(a, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    _fold_bn_into_affine(a, b)
    return model

def _param_type_map(model: nn.Module):
    # Map fully-qualified parameter name -> module type for choosing quant axis
    m = {}
    for name, mod in model.named_modules():
        for p_name, _ in mod._parameters.items():
            if _ is None: 
                continue
            fq = f"{name}.{p_name}" if name else p_name
            m[fq] = type(mod)
    return m

def _quantize_weight(t: torch.Tensor, mtype, per_tensor_for_transpose=True):
    # returns (q_int8, scales, scheme, axis)
    if isinstance(mtype, type) and issubclass(mtype, nn.Linear):
        # weight shape: (out, in) -> per-out channel
        axis = 0
        maxv = t.abs().amax(dim=1) + 1e-12
        scales = (maxv / 127.0).clamp(min=1e-12)
        q = torch.round(t / scales.view(-1, 1)).clamp_(-127, 127).to(torch.int8)
        return q, scales.float(), "per_channel", axis
    if isinstance(mtype, type) and issubclass(mtype, nn.Conv2d):
        # weight shape: (out, in, kh, kw) -> per-out channel
        axis = 0
        maxv = t.abs().amax(dim=(1, 2, 3)) + 1e-12
        scales = (maxv / 127.0).clamp(min=1e-12)
        q = torch.round(t / scales.view(-1, 1, 1, 1)).clamp_(-127, 127).to(torch.int8)
        return q, scales.float(), "per_channel", axis
    if isinstance(mtype, type) and issubclass(mtype, nn.ConvTranspose2d):
        # weight shape: (in, out, kh, kw)
        if per_tensor_for_transpose:
            maxv = t.abs().amax() + 1e-12
            scale = torch.tensor([float(maxv / 127.0)], dtype=torch.float32)
            q = torch.round(t / scale).clamp_(-127, 127).to(torch.int8)
            return q, scale, "per_tensor", None
        else:
            # per-out channel (dim=1)
            axis = 1
            maxv = t.abs().amax(dim=(0, 2, 3)) + 1e-12  # note: dim index different due to (in,out,kh,kw)
            scales = (maxv / 127.0).clamp(min=1e-12)
            q = torch.round(t / scales.view(1, -1, 1, 1)).clamp_(-127, 127).to(torch.int8)
            return q, scales.float(), "per_channel", axis
    # fallback: per-tensor
    maxv = t.abs().amax() + 1e-12
    scale = torch.tensor([float(maxv / 127.0)], dtype=torch.float32)
    q = torch.round(t / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale, "per_tensor", None

def save_model_weights_int8_fused(light_model, path, per_tensor_convT=True):
    """Quantize weights to int8 for storage, keep biases fp32; fold BN; LZMA-compress."""
    ae = fold_all_batchnorms(light_model.model.eval())
    # Build map name->module type to choose quant scheme
    pmap = _param_type_map(ae)

    pkg = {
        "meta": {
            "model_type": type(ae).__name__,
            "n_atoms": ae.n_atoms,
            "latent_dim": ae.latent_dim,
            "n_channels": ae.n_channels,
            "dtype_runtime": "float32"
        },
        "int8": {},   # name -> tensor(int8)
        "scales": {}, # name -> (tensor(float32), scheme, axis)
        "float": {}   # biases and any non-weight tensors
    }

    with torch.no_grad():
        for name, t in ae.state_dict().items():
            if t.dtype != torch.float32:
                continue
            if name.endswith(".weight"):
                mtype = pmap.get(name.rsplit(".", 1)[0], None)
                q, scales, scheme, axis = _quantize_weight(t.detach().cpu().contiguous(), mtype, per_tensor_for_transpose=per_tensor_convT)
                pkg["int8"][name] = q
                pkg["scales"][name] = {"scales": scales, "scheme": scheme, "axis": axis}
            else:
                # keep biases and others in fp32 (small)
                pkg["float"][name] = t.detach().cpu().contiguous()

    # LZMA compress torch.save payload
    buf = io.BytesIO(); torch.save(pkg, buf)
    comp = lzma.compress(buf.getvalue(), preset=9)
    with open(path, "wb") as f:
        f.write(comp)

def load_lightae_int8(path, model_type: str, device="cpu", loss_path=None, lr=1e-4, weight_decay=0.01):
    raw = lzma.decompress(open(path, "rb").read())
    pkg = torch.load(io.BytesIO(raw), map_location="cpu")  # scales/int8 on CPU
    meta = pkg["meta"]

    # Import classes
    try:
        if model_type == "AE":
            from .AE import AE as AECls, LightAE as LightAECls
        elif model_type == "skipAE":
            from .skipAE import AE as AECls, LightAE as LightAECls
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    except Exception:
        if model_type == "AE":
            from AE import AE as AECls, LightAE as LightAECls
        elif model_type == "skipAE":
            from skipAE import AE as AECls, LightAE as LightAECls
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Reconstruct full-fp32 state_dict
    state = {}
    # dequantize weights
    for name, q in pkg["int8"].items():
        info = pkg["scales"][name]
        scales = info["scales"]
        scheme = info["scheme"]; axis = info["axis"]
        qf = q.float()
        if scheme == "per_channel":
            # broadcast scale along 'axis'
            shape = qf.shape
            if axis is None:
                raise RuntimeError("axis None for per_channel")
            view_shape = [1] * len(shape); view_shape[axis] = -1
            w = qf * scales.view(*view_shape)
        else:
            w = qf * scales.view(1)
        state[name] = w.to(torch.float32)
    # copy float tensors
    for name, t in pkg["float"].items():
        state[name] = t.to(torch.float32)

    ae = AECls(n_atoms=meta["n_atoms"], latent_dim=meta["latent_dim"], n_channels=meta["n_channels"])
    # strict=False in case BN buffers were dropped
    ae.load_state_dict(state, strict=False)
    ae.to(device).eval()

    light = LightAECls(model=ae, lr=lr, loss_path=(loss_path or os.path.join(os.getcwd(), "losses.dat")))
    light.to(device).eval()
    return light

def _inspect_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    is_skip = any(k.startswith("model.encoder1.") for k in sd.keys())
    if is_skip:
        arch = "skipAE"
        W0 = sd["model.encoder1.0.weight"]     # (C, 1, n_atoms, 1)
        C = W0.shape[0]; n_atoms_conv = W0.shape[2]
        latent_dim = sd["model.decoder1.0.weight"].shape[1]
    else:
        arch = "AE"
        W0 = sd["model.encoder.0.weight"]       # (C, 1, n_atoms, 1)
        C = W0.shape[0]; n_atoms_conv = W0.shape[2]
        latent_dim = sd["model.decoder.0.weight"].shape[1]
    return arch, int(C), int(latent_dim), int(n_atoms_conv)
