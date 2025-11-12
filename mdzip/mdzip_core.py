import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import mdtraj as md
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')

from .utils import *
from .quantize import *
from .quantize import _inspect_ckpt

set_seed()

# -------------------------------------------- 
#                  T R A I N                  
# -------------------------------------------- 
def train(traj:str, top:str, stride:int=1, out:str=os.getcwd(), fname:str='', epochs:int=100, batchSize:int=128, lat:int=20, memmap:bool=False, model_type:str='AE'):
    r'''
Train the AutoEncoder model
-----------------------------
traj (str) : Path to the trajectory file [netccdf]
top (str) : Path to the topology file
stride (int) : Stride to read the trajectory [Default=1]
out (str) : Output directory to save trained model and logs [Default=current directory]
fname (str) : Prefix for all generated files [Default='']
epochs (int) : Number of epochs to train AE model [Default=100]
batchSize (int) : Batch size for training [Default=128]
lat (int) : Latent dimension size [Default=20]
memmap (bool) : Whether to use memory mapping when reading the trajectory [Default=False]
model_type (str) : Type of model to use ['AE'|'skipAE'] [Default='AE']
        '''
    pathExists(traj)
    pathExists(top)

    if model_type == 'AE':
        from .AE import Loss, AE, LightAE
    elif model_type == 'skipAE':
        from .skipAE import Loss, AE, LightAE

    if fname:
        fname += '_'

    comp_dir = os.path.join(out, f'{fname}compressed')
    if os.path.exists(comp_dir):
        shutil.rmtree(comp_dir)
    os.mkdir(comp_dir)
    out = comp_dir

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)
    n_atoms = traj_.shape[2]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(traj_.reshape(-1, 3)).reshape(traj_.shape)
    pickle.dump(scaler, open(os.path.join(out, f"{fname}scaler.pkl"), 'wb'))
    
    traj_dl = DataLoader(X, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=4)

    print('_'*70+'\n')

    model = AE(n_atoms=n_atoms, latent_dim=lat)
    model = LightAE(model=model, lr=1e-4, loss_path=os.path.join(out, f'{fname}losses.dat'))

    checkpoint_callback = ModelCheckpoint(dirpath=out, filename=f'{fname}checkpoint', save_top_k=1)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=fname+"logs/")

    trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator, devices=devices,
                         callbacks=[checkpoint_callback], logger=tb_logger)
    trainer.fit(model, traj_dl)

    ckpt_path = os.path.join(out, f'{fname}checkpoint.ckpt')
    torch.save(ckpt_path, os.path.join(out, f'{fname}checkpoint.pt'))
    save_model_weights_int8_fused(model, os.path.join(out, f'{fname}model_weights.pt.xz'))

    if os.path.exists(f"{fname}logs/"):
        shutil.rmtree(f"{fname}logs/")
    if os.path.exists('temp_traj.dat'):
        os.remove('temp_traj.dat')
    print('\n')


# -------------------------------------------- 
#        C O N T I N U E  T R A I N                  
# -------------------------------------------- 
def cont_train(traj:str, top:str, model:str, model_type:str, checkpoint:str, stride:int=1, epochs:int=100, batchSize:int=128, memmap:bool=False):
    r'''
Continue training the AutoEncoder model from a checkpoint
-----------------------------
traj (str) : Path to the trajectory file [netccdf]
top (str) : Path to the topology file
model (str) : Path to the trained model file [.pt.xz]
model_type (str) : Type of model to use ['AE'|'skipAE']
checkpoint (str) : Path to the checkpoint file [*checkpoint.pt]
stride (int) : Stride to read the trajectory [Default=1]
epochs (int) : Number of epochs to continue training AE model [Default=100]
batchSize (int) : Batch size for training [Default=128]
memmap (bool) : Whether to use memory mapping when reading the trajectory [Default=False]
'''
    pathExists(traj); pathExists(top); pathExists(checkpoint)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = torch.load(checkpoint)  # pointer to .ckpt
    pathExists(ckpt_path)
    out = os.path.dirname(ckpt_path)
    base = os.path.basename(ckpt_path)
    fname = base.replace('checkpoint.ckpt', '')

    # Rebuild exact architecture from checkpoint
    arch, C_ckpt, lat_ckpt, n_atoms_ckpt = _inspect_ckpt(ckpt_path)
    if arch == 'AE':
        from .AE import AE, LightAE
    else:
        from .skipAE import AE, LightAE

    traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)

    # Load scaler saved during train and apply transform (do not refit)
    scaler_path = os.path.join(out, f"{fname}scaler.pkl") if fname else os.path.join(out, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, 'rb'))
        X = scaler.transform(traj_.reshape(-1, 3)).reshape(traj_.shape)
    else:
        X = traj_

    # Build model that matches checkpoint; let Lightning restore weights/optimizer from ckpt
    ae = AE(n_atoms=traj_.shape[2], n_channels=C_ckpt, latent_dim=lat_ckpt)
    model = LightAE(model=ae, lr=1e-4, loss_path=os.path.join(out, f'{fname}losses.dat') if fname else os.path.join(out, 'losses.dat'))
    model = model.to(device)

    dl = DataLoader(X, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=4)
    print('_'*70+'\n')

    # Seed existing losses for continuous write-out
    if os.path.exists(model.loss_path):
        model.epoch_losses = list(np.loadtxt(model.loss_path, usecols=1))
    else:
        model.epoch_losses = []

    checkpoint_callback = ModelCheckpoint(dirpath=out, filename=f'{fname}checkpoint', save_top_k=1)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=fname+"logs/")

    trainer = pl.Trainer(max_epochs=len(model.epoch_losses)+epochs, accelerator=accelerator, devices=devices,
                         callbacks=[checkpoint_callback], logger=tb_logger)

    # Resume strictly from checkpoint
    trainer.fit(model, dl, ckpt_path=ckpt_path)

    # Save updated pointer and compact weights for runtime
    best_ckpt = trainer.checkpoint_callback.best_model_path or ckpt_path
    torch.save(best_ckpt, os.path.join(out, f'{fname}checkpoint.pt'))
    save_model_weights_int8_fused(model, os.path.join(out, f'{fname}model_weights.pt.xz'))

    if os.path.exists(f"{fname}logs/"):
        shutil.rmtree(f"{fname}logs/")
    if os.path.exists('temp_traj.dat'):
        os.remove('temp_traj.dat')
    print('\n')

# -------------------------------------------- 
#             C O M P R E S S                  
# -------------------------------------------- 
def compress(traj:str, top:str, model:str, model_type:str, stride:int=1, out:str|None=None, fname:str|None=None, memmap:bool=False):
    r'''
Compress trajectory
-----------------------------
traj (str) : Path to the trajectory file [netccdf]
top (str) : Path to the topology file
model (str) : Path to the trained model file [.pt.xz]
model_type (str) : Type of model to use ['AE'|'skipAE']
stride (int) : Stride to read the trajectory [Default=1]
out (str) : Output directory to save compressed latents [Default=directory of model]
fname (str) : Prefix for the compressed latent file [Default=derived from model name]
memmap (bool) : Whether to use memory mapping when reading the trajectory [Default=False]
'''
    import io, lzma

    pathExists(traj); pathExists(top); pathExists(model)

    if out is None:
        out = os.path.dirname(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        if model.endswith('.xz'):
            model = load_lightae_int8(model, model_type=model_type, device=device)
        else:
            model = torch.load(model)
    except Exception:
        model = load_lightae_int8(model, model_type=model_type, device=device)

    if fname is not None:
        fname += '_'
    else:
        fname = '' if os.path.basename(model.loss_path) == 'losses.dat' else os.path.basename(model.loss_path).split('_')[0] + '_'

    print('_'*70+'\n')

    traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)
    dl = DataLoader(traj_, batch_size=1, shuffle=False, drop_last=False, num_workers=4)

    encoder = model.model.encoder.to(device)
    latents = []
    with torch.no_grad():
        encoder.eval()
        for batch in tqdm(dl, desc="Compressing "):
            batch = batch.to(device=device, dtype=torch.float32)
            z = encoder(batch).view(1, -1)
            latents.append(z.cpu())

    Z = torch.cat(latents, dim=0).contiguous()

    z_min = Z.min(dim=0).values
    z_max = Z.max(dim=0).values
    scale = (z_max - z_min).clamp(min=1e-12)
    Q = torch.round((Z - z_min.view(1, -1)) * (255.0 / scale.view(1, -1))).clamp(0, 255).to(torch.uint8).contiguous()

    pkg = {
        "dtype": "uint8",
        "n_frames": int(Z.shape[0]),
        "latent_dim": int(Z.shape[1]),
        "q": Q,
        "min": z_min.float(),
        "scale": scale.float()
    }
    buf = io.BytesIO()
    torch.save(pkg, buf)
    comp_bytes = lzma.compress(buf.getvalue(), preset=9)
    out_file = os.path.join(out, f'{fname}compressed_lat.pt.xz')
    with open(out_file, 'wb') as f:
        f.write(comp_bytes)

    print('_'*70+'\n')
    org_size = os.path.getsize(traj)
    comp_size = os.path.getsize(out_file)
    compression = 100*(1 - comp_size/org_size)
    template = "{string:<20} :{value:15.3f}"
    print(template.format(string='Original Size [MB]', value=round(org_size*1e-6,3)))
    print(template.format(string='Compressed Lat [MB]', value=round(comp_size*1e-6,3)))
    print(template.format(string='Compression %', value=round(compression,3)))
    print('Saved:', out_file)
    print('---')


# -------------------------------------------- 
#            D E C O M P R E S S                  
# -------------------------------------------- 
def decompress(top:str, model:str, compressed:str, out:str, model_type:str, scaler:str):
    r'''
Decompress latents to trajectory
-----------------------------
top (str) : Path to the topology file
model (str) : Path to the trained model file [.pt.xz]
compressed (str) : Path to the compressed latent file [.pt.xz]
out (str) : Output trajectory file path [.nc|.xtc]
model_type (str) : Type of model to use ['AE'|'skipAE']
scaler (str) : Path to the saved scaler file [.pkl]
    '''
    import io, lzma

    pathExists(top); pathExists(model); pathExists(compressed); pathExists(os.path.dirname(out))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        if model.endswith('.xz'):
            model = load_lightae_int8(model, model_type=model_type, device=device)
        else:
            model = torch.load(model)
    except Exception:
        model = load_lightae_int8(model, model_type=model_type, device=device)

    decoder = model.model.decoder.to(device)

    if compressed.endswith('.xz'):
        raw = lzma.decompress(open(compressed, 'rb').read())
        pkg = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if pkg.get("dtype") != "uint8":
            raise ValueError("Unsupported latent package dtype")
        q = pkg["q"].to(torch.uint8)
        z_min = pkg["min"].to(torch.float32)
        scale = pkg["scale"].to(torch.float32)
        Z = (q.float() / 255.0) * scale.view(1, -1) + z_min.view(1, -1)
    else:
        Z = torch.cat(pickle.load(open(compressed, 'rb'))).float()

    scaler = pickle.load(open(scaler, 'rb'))
    with torch.no_grad():
        decoder.eval()
        if out.endswith('.nc'):
            traj_file = md.formats.netcdf.NetCDFTrajectoryFile(out, 'w')
        elif out.endswith('.xtc'):
            traj_file = md.formats.xtc.XTCTrajectoryFile(out, 'w')
        else:
            raise ValueError('Supported formats: .nc, .xtc')

        with traj_file as f:
            for i in tqdm(range(Z.shape[0]), desc='Decompressing '):
                z_i = Z[i:i+1, :]
                np_traj_frame = decoder(z_i.to(device)).detach().cpu().numpy()
                np_traj_frame = np_traj_frame.reshape(-1, np_traj_frame.shape[2], 3)
                np_traj_frame = scaler.inverse_transform(np_traj_frame.reshape(-1, 3)).reshape(np_traj_frame.shape)
                f.write(np_traj_frame*10)

    print('\nDecompression complete\n')
    
###########################################

def _build_parser():
    p = argparse.ArgumentParser(prog="mdzip", description="MDZip CLI", allow_abbrev=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Train")
    t.add_argument("-t", "--traj", required=True, type=str, help="Path to trajectory file (str)")
    t.add_argument("-T", "--top", required=True, type=str, help="Path to topology file (str)")
    t.add_argument("-s", "--stride", type=int, default=1, help="Stride to read trajectory (int) [default=1]")
    t.add_argument("-o", "--out", default=os.getcwd(), type=str, help="Output directory to save trained model and logs (str) [default=current directory]")
    t.add_argument("-f", "--fname", default="", type=str, help="Prefix for all generated files (str) [default='']")
    t.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to train AE model (int) [default=100]")
    t.add_argument("-b", "--batch-size", type=int, default=128, dest="batchSize", help="Batch size for training (int) [default=128]")
    t.add_argument("-l", "--lat", type=int, default=20, help="Latent dimension size (int) [default=20]")
    t.add_argument("-M", "--memmap", action="store_true", help="Use memory mapping when reading the trajectory")
    t.add_argument("-A", "--model-type", choices=["AE", "skipAE"], default="AE", help="Type of model to use ['AE'|'skipAE'] [default='AE']")

    # cont_train
    c = sub.add_parser("cont_train", help="Continue training")
    c.add_argument("-t", "--traj", required=True, type=str, help="Path to trajectory file (str)")
    c.add_argument("-T", "--top", required=True, type=str, help="Path to topology file (str)")
    c.add_argument("-m", "--model", default="", type=str, help="Path to trained model file (.pt.xz) (str)")
    c.add_argument("-A", "--model-type", choices=["AE", "skipAE"], required=True, help="Type of model to use")
    c.add_argument("-c", "--checkpoint", required=True, type=str, help="Path to checkpoint file (.pt) (str)")
    c.add_argument("-s", "--stride", type=int, default=1, help="Stride to read trajectory (int) [default=1]")
    c.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to continue training AE model (int) [default=100]")
    c.add_argument("-b", "--batch-size", type=int, default=128, dest="batchSize", help="Batch size for training (int) [default=128]")
    c.add_argument("-M", "--memmap", action="store_true", help="Use memory mapping when reading the trajectory")

    # compress
    x = sub.add_parser("compress", help="Compress to latents")
    x.add_argument("-t", "--traj", required=True, type=str, help="Path to trajectory file (str)")
    x.add_argument("-T", "--top", required=True, type=str, help="Path to topology file (str)")
    x.add_argument("-m", "--model", required=True, type=str, help="Path to trained model file (.pt.xz) (str)")
    x.add_argument("-A", "--model-type", choices=["AE", "skipAE"], required=True, help="Type of model to use")
    x.add_argument("-s", "--stride", type=int, default=1, help="Stride to read trajectory (int) [default=1]")
    x.add_argument("-o", "--out", default=None, type=str, help="Output directory to save compressed latents (str) [default=directory of model]")
    x.add_argument("-f", "--fname", default=None, type=str, help="Prefix for the compressed latent file (str) [default=derived from model name]")
    x.add_argument("-M", "--memmap", action="store_true", help="Use memory mapping when reading the trajectory")

    # decompress
    d = sub.add_parser("decompress", help="Decompress latents")
    d.add_argument("-T", "--top", required=True, type=str, help="Path to topology file (str)")
    d.add_argument("-m", "--model", required=True, type=str, help="Path to trained model file (.pt.xz) (str)")
    d.add_argument("-z", "--compressed", required=True, type=str, help="Path to compressed latent file (.pt.xz) (str)")
    d.add_argument("-o", "--out", required=True, type=str, help="Output trajectory file path (.nc|.xtc) (str)")
    d.add_argument("-A", "--model-type", choices=["AE", "skipAE"], type=str, required=True, help="Type of model to use")
    d.add_argument("-S", "--scaler", required=True, type=str, help="Path to the saved scaler file (.pkl) (str)")

    return p

def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.cmd == "train":
        return train(args.traj, args.top, args.stride, args.out, args.fname, args.epochs, args.batchSize, args.lat, args.memmap, args.model_type)
    if args.cmd == "cont_train":
        return cont_train(args.traj, args.top, args.model, args.model_type, args.checkpoint, args.stride, args.epochs, args.batchSize, args.memmap)
    if args.cmd == "compress":
        return compress(args.traj, args.top, args.model, args.model_type, args.stride, args.out, args.fname, args.memmap)
    if args.cmd == "decompress":
        return decompress(args.top, args.model, args.compressed, args.out, args.model_type, args.scaler)

if __name__ == "__main__":
    main()