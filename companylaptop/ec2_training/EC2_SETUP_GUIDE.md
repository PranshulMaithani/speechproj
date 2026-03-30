# EC2 Training Guide — wav2vec2 Speech Classification

## Recommended Instance

| Instance | GPU | VRAM | Cost (on-demand) | Notes |
|----------|-----|------|-------------------|-------|
| **g4dn.xlarge** | T4 | 16 GB | ~$0.53/hr | Budget pick, batch_size=8 works |
| **g5.xlarge** | A10G | 24 GB | ~$1.01/hr | Faster, batch_size=16 |
| **p3.2xlarge** | V100 | 16 GB | ~$3.06/hr | Fastest training |

Use **g4dn.xlarge** unless you want faster training. T4 is fine for this model.

**AMI**: Ubuntu 22.04 LTS (search "Ubuntu 22.04" in AMI marketplace)
**Storage**: At least **80 GB** gp3 (datasets ~30GB + model + cache)
**Region**: Pick one close to you for SSH latency

---

## 1. Launch & Connect

```bash
# After launching the instance, SSH in:
ssh -i your-key.pem ubuntu@<public-ip>

# Optional: set up tmux so training survives SSH disconnects
tmux new -s train
# (Ctrl+B, D to detach; tmux attach -t train to reconnect)
```

---

## 2. Install NVIDIA Drivers + CUDA

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (Ubuntu 22.04)
sudo apt install -y nvidia-driver-535
sudo reboot
```

After reboot, reconnect and verify:
```bash
nvidia-smi
# Should show your GPU (T4/A10G/V100)
```

Install CUDA toolkit + cuDNN:
```bash
# CUDA 12.1 (matches PyTorch 2.x)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# cuDNN
sudo apt install -y libcudnn8 libcudnn8-dev

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
python3 -c "import ctypes; ctypes.CDLL('libcudnn.so')"
```

---

## 3. Setup Python Environment

```bash
# Install system dependencies
sudo apt install -y python3.10-venv python3-pip ffmpeg libsndfile1 git-lfs

# Create venv
python3 -m venv ~/wav2vec2_env
source ~/wav2vec2_env/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers datasets librosa soundfile pandas numpy scikit-learn \
            tqdm xgboost joblib onnx onnxruntime huggingface_hub

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 4. Create Working Directory & Upload Scripts

```bash
mkdir -p ~/training
cd ~/training
```

**Option A: Clone from your machine (if you push scripts to a repo)**
```bash
git clone <your-repo-url> .
```

**Option B: SCP from your local machine**
```bash
# Run this FROM YOUR LOCAL MACHINE (not EC2):
scp -i your-key.pem companylaptop/ec2_training/*.py ubuntu@<public-ip>:~/training/
```

**Option C: Copy-paste scripts** — just create the files manually on EC2.

---

## 5. Download Datasets

### 5a. Meta Casual Conversations v2 (CCv2)

CCv2 has **both scripted (read) and unscripted (spontaneous)** video recordings.
Files inside zips are named: `{participant_id}_{language}_{scripted|nonscripted}_{index}.mp4`

We only extract **English** files and convert MP4 → WAV audio.

**Step 1: Save download links**

After agreeing to Meta's license, you get a page with ~80 download links.
Copy the links table and save to a text file (tab-separated: `filename\turl`):

```bash
# Create the links file on EC2 (paste the links from Meta's page)
cat > ~/training/ccv2_links.txt << 'LINKS'
CCv2_part_1.zip	https://scontent.xx.fbcdn.net/...
CCv2_part_2.zip	https://scontent.xx.fbcdn.net/...
...
LINKS
```

Or SCP from your local machine:
```bash
scp -i your-key.pem ccv2_links.txt ubuntu@<public-ip>:~/training/
```

**Step 2: Download & process (automatic, one zip at a time)**

```bash
cd ~/training

# Downloads each zip, extracts only English MP4s, converts to WAV, deletes zip
# Stops when it has 5000 scripted + 5000 nonscripted (or runs out of parts)
python download_casual_conversations.py --links ccv2_links.txt --max-per-class 5000
```

This is disk-efficient: each zip (~30-60 GB) is downloaded, English files extracted,
then the zip is deleted before the next one. Peak disk usage: ~60 GB + extracted audio.

### 5c. LibriSpeech + AMI

```bash
cd ~/training
python download_datasets.py
```

Downloads:
- **LibriSpeech train-clean-100** (~6.3 GB) — read speech (audiobook)
- **AMI ihm** (~7 GB via HuggingFace) — spontaneous meetings

### 5d. Build Balanced Manifest

After all datasets are ready:
```bash
python download_datasets.py --manifest-only
```

**Balancing logic**:
- Casual Conv: equal scripted and unscripted (min of both)
- LibriSpeech capped so **total read ≈ total spontaneous**
```
Read:        ALLSSTAR read + LibriSpeech (capped) + CCv2 scripted
Spontaneous: ALLSSTAR spont + AMI + CCv2 unscripted
```

---

## 6. Train

```bash
cd ~/training
source ~/wav2vec2_env/bin/activate

# Use tmux so training survives SSH disconnects!
tmux new -s train

# Train (should take 2-4 hours on T4, 1-2 hours on A10G)
python train.py

# Or with custom settings:
python train.py --epochs 12 --batch-size 16 --lr 1e-5 --freeze-layers 6
```

Monitor from another terminal:
```bash
# GPU usage
watch -n 2 nvidia-smi

# Training logs
tail -f ~/training/checkpoints/training.log
```

---

## 7. Export to ONNX

```bash
python export_onnx.py
# Creates: checkpoints/wav2vec2_trained.onnx (fp32, ~361 MB)
#          checkpoints/wav2vec2_trained_quant.onnx (int8, ~117 MB)
```

---

## 8. Upload Results

```bash
# Upload trained model to HuggingFace
huggingface-cli upload Pransfrance/speechproj-models \
    checkpoints/wav2vec2_trained_quant.onnx combined_v2/wav2vec2_combined_quant.onnx

huggingface-cli upload Pransfrance/speechproj-models \
    checkpoints/wav2vec2_best.pt combined_v2/wav2vec2_combined_best.pt

huggingface-cli upload Pransfrance/speechproj-models \
    checkpoints/results.json combined_v2/results.json

huggingface-cli upload Pransfrance/speechproj-models \
    checkpoints/history.json combined_v2/history.json
```

---

## 9. Download to Company Laptop

On your company laptop:
```bash
cd companylaptop/
huggingface-cli download Pransfrance/speechproj-models \
    combined_v2/wav2vec2_combined_quant.onnx --local-dir .
```

Then update `eval_and_finetune.ipynb` to point to the new model.

---

## 10. Shut Down Instance

**IMPORTANT: Don't forget to stop/terminate the instance when done!**

```bash
# From AWS Console: Instances -> Select -> Instance State -> Stop/Terminate
# Or from CLI:
aws ec2 stop-instances --instance-ids <instance-id>
```

---

## Quick Reference — Full Flow

```bash
# 1. SSH in + tmux
ssh -i key.pem ubuntu@<ip>
tmux new -s train

# 2. Activate env
source ~/wav2vec2_env/bin/activate
cd ~/training

# 3. Download data (first time only)
python download_allsstar.py                                          # ~9 GB from HuggingFace
python download_casual_conversations.py --links ccv2_links.txt       # CCv2 English audio
python download_datasets.py                                          # LibriSpeech + AMI
python download_datasets.py --manifest-only                          # build balanced manifest

# 4. Train
python train.py

# 5. Export
python export_onnx.py

# 6. Upload
huggingface-cli upload Pransfrance/speechproj-models checkpoints/ combined_v2/

# 7. STOP THE INSTANCE
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `--batch-size` (try 4) |
| `nvidia-smi` not found | Drivers not installed, run step 2 |
| SSH disconnected during training | Use `tmux attach -t train` to reconnect |
| Slow download | Check instance region, use `aria2c` for parallel downloads |
| `libsndfile` error | `sudo apt install libsndfile1` |
| `RuntimeError: cuDNN not found` | `sudo apt install libcudnn8` |
