import io
import torch
import os

def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema, save_every_epoch=200):
    """
    save the model to path
    """
    save_ckpt = {
        'epoch': epoch + 1, 
        'model': model.state_dict(), 
        'ema_state_dict': ema.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'schedule': scheduler.state_dict(), 
        'loss': train_loss,
        'wandb_id': wandb_id
    }
    
    torch.save(save_ckpt, f"{save_path}/latest.pth")

    if epoch+1 >= save_every_epoch:
        with open(f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth', "wb") as f:
            torch.save(save_ckpt, f)

def load_model(path: str):
    """
    load ckpt from path
    """
    ckpt = torch.load(path, weights_only=True)

    return ckpt


def resume_model(path: str, model, optimizer, scheduler, ema, device, strict: bool = True):
    """
    load ckpt from path

    NOTE: previous version used bare ``except:`` clauses around every load
    step. That silently swallowed key-mismatch errors AND silently broke the
    DDP-prefix fallback (which incorrectly split ``ckpt`` top-level keys like
    ``'epoch'`` / ``'model'`` instead of the inner ``ckpt['model']`` state
    dict). The net effect was: a checkpoint with a tiny architectural drift
    (e.g. ``centerline_gate`` added by Patch 4) silently re-initialised the
    model to random weights and training continued. This rewrite narrows the
    exception classes and surfaces real load failures.
    """
    path = os.path.join(path, 'latest.pth')
    ckpt = torch.load(path, weights_only=True)

    # ---- model state dict ----
    raw_model_sd = ckpt['model']
    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            raw_model_sd, strict=False
        )
    except (RuntimeError, KeyError):
        # DDP prefix fallback: strip ``module.`` and retry.
        cleaned = {k.replace('module.', '', 1): v for k, v in raw_model_sd.items()}
        missing_keys, unexpected_keys = model.load_state_dict(cleaned, strict=False)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"resume_model: state_dict mismatch (strict=True). "
            f"missing[:5]={list(missing_keys)[:5]} "
            f"unexpected[:5]={list(unexpected_keys)[:5]}"
        )
    print(
        f"Model load done (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
    )

    # ---- optimizer ----
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except KeyError:
        print("no pretrained optimizer found")

    # ---- scheduler ----
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except KeyError:
        print("no schedule found,")

    # ---- step / epoch ----
    init_epoch = ckpt.get('epoch', 0)
    if 'epoch' in ckpt:
        print("Step load done")

    # ---- wandb id ----
    wandb_id = ckpt.get('wandb_id', None)
    if 'wandb_id' in ckpt:
        print("wandb id load done")

    # ---- ema shadow ----
    try:
        ema.ema.load_state_dict({n: v for n, v in ckpt['ema_state_dict'].items()})
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)
        print("ema load done")
    except KeyError:
        print('no ema shadow found')

    return model, optimizer, scheduler, init_epoch, wandb_id, ema