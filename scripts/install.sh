set -euo pipefail
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps 'git+https://github.com/facebookresearch/sam2.git@722d1d15111c689908aeeb82d49a57780aac5153'
python -m pip install --no-deps 'git+https://github.com/hkchengrex/Cutie.git@ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a'
python -m pip install -e . --no-deps
python -m pytest -q
