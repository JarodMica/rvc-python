import subprocess
import os
import json
import glob
from tqdm import tqdm
from pathlib import Path
import requests

def download_rvc_models(this_dir):
    folder = os.path.join(this_dir,'base_model')
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    files = {
        "hubert_base.pt": "https://huggingface.co/daswer123/rvc_base/resolve/bbb6736b97a98df0a87fe3592c0a061c53f0a75f/hubert_base.pt?download=true",
        "rmvpe.pt": "https://huggingface.co/daswer123/rvc_base/resolve/bbb6736b97a98df0a87fe3592c0a061c53f0a75f/rmvpe.pt?download=true",
        "rmvpe.onnx": "https://huggingface.co/daswer123/rvc_base/resolve/bbb6736b97a98df0a87fe3592c0a061c53f0a75f/rmvpe.onnx?download=true"
    }
    
    for filename, url in files.items():
        file_path = os.path.join(folder, filename)
    
        if not os.path.exists(file_path):
            print(f'File {filename} not found, start loading...')
    
            response = requests.get(url)
    
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f'File {filename} successfully loaded.')
            else:
                print(f'f {filename}.')
