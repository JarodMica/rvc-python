# RVC Python

A Python implementation for using RVC (Retrieval-based Voice Conversion) via console, Python scripts, or API.

Base code is from: https://github.com/daswer123/rvc-python

I have made a few modifications for my workflow and have removed areas of the readme that I am not using.  Please use daswers if you're interested in using this library for other systems or other applications.

## Installation Python 3.11 (Windows 10/11)
1. Create a venv with python 3.11 and activate it
    ```
    py -3.11 -m venv venv
    .\venv\Scripts\activate    
    ```
2. Install necessary requirements. We first install the github repo, then download and install a 3.11 compatible fairseq wheels file (as the OG pypy at 0.12.2 is broken for 3.11), then install pytorch at 2.4.0.

    If you can't use `curl` for some reason, you can manually download the fairseq wheels file here: https://huggingface.co/Jmica/rvc/resolve/main/fairseq-0.12.4-cp311-cp311-win_amd64.whl?download=true then you can skip the `curl` command below

    ```
    pip install git+https://github.com/JarodMica/rvc-python
    curl -Uri "https://huggingface.co/Jmica/rvc/resolve/main/fairseq-0.12.4-cp311-cp311-win_amd64.whl?download=true" -OutFile "fairseq-0.12.4-cp311-cp311-win_amd64.whl"
    pip install .\fairseq-0.12.4-cp311-cp311-win_amd64.whl
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```

### Python Usage
When you first attempt to use this, hubert and rmvpe models will be downloaded from HF into the rvc_python folder in you venv.  You can find them there.

Create a new object from RVCInference with the parameters you want set. The defaults are fine, but `rvc_models` is the root directory that will be looked inside of for RVC models.

Inside root, there should be individual directories named after the desired speaker, and in each, there should be a .pth and .index(if using) file like below.

```
rvc_models/
├── name_model1/
│   ├── name_model1.pth
│   └── name_model1.index
└── name_model2/
    └── name_model2.pth

```

An example usage is shown below:

```python
from rvc_python.infer import RVCInference

rvc = RVCInference(models_dir="rvc_models", 
                   device="cuda:0",
                   f0method = "rmvpe",
                   f0up_key=0,
                   index_rate=0.5,
                   filter_radius=3,
                   resample_sr=48000,
                   rms_mix_rate=1,
                   protect=0.33)
rvc.load_model("name_of_rvc_model")
rvc.infer_file("input_path.wav", "output_path.wav")
```

## Acknowledgements

Thank you to daswer123 for creating the library for easy RVC inference: https://github.com/daswer123/rvc-python

