import os

from fairseq import checkpoint_utils
from rvc_python.lib.jit.get_hubert import get_hubert_model
def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config, lib_dir):
    hubert_model = get_hubert_model(
        model_path=os.path.join(lib_dir, "base_model", "hubert_base.pt"),
        device=config.device,
    )
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
