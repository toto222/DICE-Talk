# DICE-Talk
Disentangle Identity, Cooperate Emotion: Correlation-Aware Emotional Talking Portrait Generation.


<a href='https://toto222.github.io/DICE-Talk/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2504.18087'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href="https://raw.githubusercontent.com/toto222/DICE-Talk/refs/heads/main/LICENSE" style="margin: 0 2px;">
  <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
</a>




## 🔥🔥🔥 NEWS
**`2025/08/01`**: Our DICE-Talk is accepted by ACM MM'25!

**`2025/04/29`**: We released the initial version of the inference code and models. Stay tuned for continuous updates!



## 🎥 Demo
| Input                | Neutral                | Happy                | Angry                | Surprised
|----------------------|-----------------------|----------------------|-----------------------|-----------------------|
|<img src="examples/img/female.png" width="640">|<video src="https://github.com/user-attachments/assets/e17ccff2-12f3-4d0e-8475-ce0e2dd6bd2a" width="320"> </video>|<video src="https://github.com/user-attachments/assets/cf799a36-c489-453f-85a7-dce7f366e0f0" width="320"> </video>|<video src="https://github.com/user-attachments/assets/c30c39f8-ab5d-4382-837d-b26137edbdd8" width="320"> </video>|<video src="https://github.com/user-attachments/assets/5f24b0dc-2f43-46c9-90bd-cc9e635be014" width="320"> </video>|
|<img src="examples/img/pyy.jpg" width="640">|<video src="https://github.com/user-attachments/assets/629753bc-aad0-45f3-bc0b-b6b8eb599f17" width="320"> </video>|<video src="https://github.com/user-attachments/assets/8619ef3d-4669-45ee-9cce-3f21df6d4bb3" width="320"> </video>|<video src="https://github.com/user-attachments/assets/79bae96b-175b-4dd4-8d4e-6f325959f67f" width="320"> </video>|<video src="https://github.com/user-attachments/assets/f3f7287d-e0b9-466d-abf2-019ef44f5ace" width="320"> </video>|




For more visual demos, please visit our [**Page**](https://toto222.github.io/DICE-Talk/).



## 📜 Requirements
* It is recommended to use a GPU with `20GB` or more VRAM and have an independent `Python 3.10`.
* Tested operating system: `Linux`

## 🔑 Inference

### Installtion
- `ffmpeg` requires to be installed.
- `PyTorch`: make sure to select the appropriate CUDA version based on your hardware, for example,
```shell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```
- `Dependencies`:
```shell
pip install -r requirements.txt
```
- All models are stored in `checkpoints` by default, and the file structure is as follows:
```shell
DICE-Talk
  ├──checkpoints
  │  ├──DICE-Talk
  │  │  ├──audio_linear.pth
  │  │  ├──emo_model.pth
  │  │  ├──pose_guider.pth  
  │  │  ├──unet.pth
  │  ├──stable-video-diffusion-img2vid-xt
  │  │  ├──...
  │  ├──whisper-tiny
  │  │  ├──...
  │  ├──RIFE
  │  │  ├──flownet.pkl
  │  ├──yoloface_v5m.pt
  ├──...
```
Download by `huggingface-cli` follow
```shell
python3 -m pip install "huggingface_hub[cli]"

huggingface-cli download EEEELY/DICE-Talk --local-dir  checkpoints
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir  checkpoints/stable-video-diffusion-img2vid-xt
huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny
```

or manully download [pretrain model](https://drive.google.com/drive/folders/1l1Ojt-4yMfYQCCnNs_NgkzQC2-OoAksN?usp=drive_link), [svd-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and [whisper-tiny](https://huggingface.co/openai/whisper-tiny) to `checkpoints/`.


### Run demo
```shell
python3 demo.py --image_path '/path/to/input_image' --audio_path '/path/to/input_audio'\ 
  --emotion_path '/path/to/input_emotion' --output_path '/path/to/output_video'
```

### Run GUI
```shell
python3 gradio_app.py
```

<img width="720" alt="gradio_demo" src="https://github.com/user-attachments/assets/7cdb2e6b-53c4-43e4-b6df-2b25db10ea8d" />



On the left you need to:
* Upload an image or take a photo
* Upload or record an audio clip
* Select the type of emotion to generate
* Set the strength for identity preservation and emotion generation
* Choose whether to crop the input image

On the right are the generated videos.

 
## 🔗 Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@article{tan2025dicetalk,
  title={Disentangle Identity, Cooperate Emotion: Correlation-Aware Emotional Talking Portrait Generation}, 
  author={Tan, Weipeng and Lin, Chuming and Xu, Chengming and Xu, FeiFan and Hu, Xiaobin and Ji, Xiaozhong and Zhu, Junwei and Wang, Chengjie and Fu, Yanwei},
  journal={arXiv preprint arXiv:2504.18087},
  year={2025}
}

@article{ji2024sonic,
  title={Sonic: Shifting Focus to Global Audio Perception in Portrait Animation},
  author={Ji, Xiaozhong and Hu, Xiaobin and Xu, Zhihong and Zhu, Junwei and Lin, Chuming and He, Qingdong and Zhang, Jiangning and Luo, Donghao and Chen, Yi and Lin, Qin and others},
  journal={arXiv preprint arXiv:2411.16331},
  year={2024}
}

@article{ji2024realtalk,
  title={Realtalk: Real-time and realistic audio-driven face generation with 3d facial prior-guided identity alignment network},
  author={Ji, Xiaozhong and Lin, Chuming and Ding, Zhonggan and Tai, Ying and Zhu, Junwei and Hu, Xiaobin and Luo, Donghao and Ge, Yanhao and Wang, Chengjie},
  journal={arXiv preprint arXiv:2406.18284},
  year={2024}
}
```

## 🚀 Deploying on RunPod Serverless

This project can be deployed as a serverless endpoint on RunPod. Here's how:

1.  **Ensure you have a RunPod account.**
2.  **Create a Serverless Endpoint:**
    *   Go to `Serverless -> My Endpoints` and click `+ New Endpoint`.
    *   Configure the endpoint:
        *   **Name:** Choose a name (e.g., "dice-talk").
        *   **Select Template:** You might start with a blank template or a generic Python one if you don't have a custom template yet.
        *   **Container Image:** You will build a Docker image from the `Dockerfile` in this repository and push it to a container registry (like Docker Hub, GCP Artifact Registry, AWS ECR). Then, provide the image URI here.
        *   **GPU Configuration:** Select a GPU type with at least 20GB VRAM (e.g., RTX 3090, A100).
        *   **Min/Max Workers:** Configure as needed.
        *   **Idle Timeout:** Set an appropriate idle timeout.
        *   **Container Disk:** Ensure it's large enough for the models and dependencies (e.g., 25-30 GB might be a safe start, adjust as needed).
3.  **Build and Push Docker Image:**
    *   Clone this repository.
    *   Build the Docker image: `docker build -t your-registry/dice-talk:latest .`
    *   Push the image to your container registry: `docker push your-registry/dice-talk:latest`
    *   *(Replace `your-registry` with your actual container registry path)*
4.  **Handler:** The entry point for RunPod is `runpod_handler.py`. The `Dockerfile` is already configured to use this.
5.  **Making Requests:**
    *   Once the endpoint is active, you can send requests to its API URL.
    *   The input should be a JSON payload like this:

    ```json
    {
        "input": {
            "image_url": "URL_TO_YOUR_IMAGE.png",
            "audio_url": "URL_TO_YOUR_AUDIO.wav",
            "emotion": "happy", // Optional: e.g., "neutral", "happy", "angry", "surprised", "sad", "disgusted", "fear", "contempt" (defaults to "neutral")
            "ref_scale": 3.0,     // Optional: Identity preservation strength (default: 3.0)
            "emo_scale": 6.0,     // Optional: Emotion generation strength (default: 6.0)
            "crop": false,        // Optional: Whether to crop the input image (default: false)
            "seed": null          // Optional: Random seed for generation (default: None)
        }
    }
    ```
    *   The response will be a JSON containing either an `output_video_path` (if using direct output from RunPod) or an error message. For production, you'd typically modify `runpod_handler.py` to upload the video to a cloud storage (like S3) and return a presigned URL.

**Note on Model Downloads:** The `Dockerfile` handles downloading all necessary pre-trained models during the image build process. This means the models will be included in your Docker image.
