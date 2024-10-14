# RVC Python

A Python implementation for using RVC (Retrieval-based Voice Conversion) via console, Python scripts, or API.

Base code is from: https://github.com/daswer123/rvc-python

I have made a few modifications for my workflow and have removed areas of the readme that I am not using.  Please use daswers if you're interested in using this library for other systems or other applications.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python Module](#python-module)
  - [API](#api)
- [Model Management](#model-management)
- [Options](#options)
- [Advanced Setup (GPU Acceleration)](#advanced-setup-gpu-acceleration)
- [Changelog](#changelog)
- [Contributing](#contributing)


## Installation (Windows 10/11)

```
pip install #enter github repo here
curl -Uri "https://huggingface.co/Jmica/rvc/resolve/main/fairseq-0.12.4-cp311-cp311-win_amd64.whl?download=true" -OutFile "fairseq-0.12.4-cp311-cp311-win_amd64.whl"
pip install .\fairseq-0.12.4-cp311-cp311-win_amd64.whl
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```




### Python Module

```python
from rvc_python.infer import RVCInference

rvc = RVCInference(device="cuda:0")
rvc.load_model("path/to/model.pth")
rvc.infer_file("input.wav", "output.wav")
```

## Model Management

Models are stored in the `rvc_models` directory by default. Each model should be in its own subdirectory and contain:

- A `.pth` file (required): The main model file.
- An `.index` file (optional): For improved voice conversion quality.

Example structure:
```
rvc_models/
├── model1/
│   ├── model1.pth
│   └── model1.index
└── model2/
    └── model2.pth
```

You can add new models by:
1. Manually placing them in the `rvc_models` directory.
2. Using the `/upload_model` API endpoint to upload a zip file containing the model files.
3. Using the `/set_models_dir` API endpoint to change the models directory dynamically.

## Options

### Input/Output Options
- `-i`, `--input`: Input audio file (CLI mode)
- `-d`, `--dir`: Input directory for batch processing (CLI mode)
- `-o`, `--output`: Output file or directory

### Model Options
- `-mp`, `--model`: Path to the RVC model file (required for CLI, optional for API)
- `-md`, `--models_dir`: Directory containing RVC models (default: `rvc_models` in the current directory)
- `-ip`, `--index`: Path to the index file (optional)
- `-v`, `--version`: Model version (v1 or v2)

### Processing Options
- `-de`, `--device`: Computation device (e.g., "cpu", "cuda:0")
- `-me`, `--method`: Pitch extraction method (harvest, crepe, rmvpe, pm)
- `-pi`, `--pitch`: Pitch adjustment in semitones
- `-ir`, `--index_rate`: Feature search ratio
- `-fr`, `--filter_radius`: Median filtering radius for pitch
- `-rsr`, `--resample_sr`: Output resampling rate
- `-rmr`, `--rms_mix_rate`: Volume envelope mix rate
- `-pr`, `--protect`: Protection for voiceless consonants

### API Server Options
- `-p`, `--port`: API server port (default: 5050)
- `-l`, `--listen`: Allow external connections to API server
- `-pm`, `--preload-model`: Preload a model when starting the API server (optional)

## Changelog

For a detailed list of changes and updates, please see the [Releases page](https://github.com/daswer123/rvc-python/releases).

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for bugs and feature requests.
