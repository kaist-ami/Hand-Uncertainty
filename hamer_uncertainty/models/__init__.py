from .mano_wrapper import MANO
from .hamer import HAMER
from .hamer_diag import HAMER_DIAG
from .hamer_full import HAMER_FULL
from .hamer_ours import HAMER_OURS
from .hamer_ours_wo_linear import HAMER_OURS_WO_LINEAR
from .discriminator import Discriminator

from ..utils.download import cache_url
from ..configs import CACHE_DIR_HAMER


def download_models(folder=CACHE_DIR_HAMER):
    """Download checkpoints and files for running inference.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    download_files = {
        "hamer_demo_data.tar.gz"      : ["https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz", folder],
    }
    
    for file_name, url in download_files.items():
        output_path = os.path.join(url[1], file_name)
        if not os.path.exists(output_path):
            print("Downloading file: " + file_name)
            # output = gdown.cached_download(url[0], output_path, fuzzy=True)
            output = cache_url(url[0], output_path)
            assert os.path.exists(output_path), f"{output} does not exist"

            # if ends with tar.gz, tar -xzf
            if file_name.endswith(".tar.gz"):
                print("Extracting file: " + file_name)
                os.system("tar -xvf " + output_path)

DEFAULT_CHECKPOINT=f'{CACHE_DIR_HAMER}/hamer_ckpts/checkpoints/hamer.ckpt'
def load_hamer(checkpoint_path=DEFAULT_CHECKPOINT):
    from pathlib import Path
    from ..configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg

def load_hamer_uncertainty(model_cfg):
    from pathlib import Path

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    # Setup model
    if model_cfg.model_type == None or model_cfg.model_type == 'deterministic':
        model = HAMER.load_from_checkpoint(model_cfg.ckpt_path, strict=False, cfg=model_cfg, map_location='cuda:0')
    elif model_cfg.model_type == 'diag':
        model = HAMER_DIAG.load_from_checkpoint(model_cfg.ckpt_path, strict=False, cfg=model_cfg, map_location='cuda:0')
    elif model_cfg.model_type == 'full':
        model = HAMER_FULL.load_from_checkpoint(model_cfg.ckpt_path, strict=False, cfg=model_cfg, map_location='cuda:0')
    elif model_cfg.model_type == 'ours':
        model = HAMER_OURS.load_from_checkpoint(model_cfg.ckpt_path, strict=False, cfg=model_cfg, map_location='cuda:0')
    elif model_cfg.model_type == 'ours_wo_linear':
        model = HAMER_OURS_WO_LINEAR.load_from_checkpoint(model_cfg.ckpt_path, strict=False, cfg=model_cfg, map_location='cuda:0')
    else:
        print(f"There is no model_type: {model_cfg.model_type}")

    print(f"Model type : {model_cfg.model_type}")
    print(f"Load checkpoint from {model_cfg.ckpt_path}")
    return model, model_cfg