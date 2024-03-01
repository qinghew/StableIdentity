<div align="center">

## StableIdentity: Inserting Anybody into Anywhere at First Sight
🤗[[Paper](https://huggingface.co/papers/2401.15975)] &emsp; 🔥[[Project Page](https://qinghew.github.io/StableIdentity/)]
</div>

## News
- **[2024.03.01]**: Release codes for **StableIdentity & ModelScopeT2V (Identity-Driven Video Generation)** codes!
- **[2024.03.01]**: Release codes for **StableIdentity & LucidDreamer (Identity-Driven 3D Generation)** codes!
- **[2024.02.29]**: Release codes for **StableIdentity & ControlNet** codes!
- **[2024.02.25]**: Release **training and inference codes**!


<img src="https://qinghew.github.io/StableIdentity/static/images/first_image.svg" width="100%">

Click the GIF to access the high-resolution videos.
<table class="center">
  <td><a href="https://qinghew.github.io/StableIdentity/static/videos/video1/burger1.mp4"><img src=assets/burger.gif width="180"></td>
  <td><a href="https://qinghew.github.io/StableIdentity/static/videos/video1/golden_crown.mp4"><img src=assets/golden_crown.gif width="180"></td>
  <td><a href="https://qinghew.github.io/StableIdentity/static/videos/video1/makeup1.mp4"><img src=assets/makeup.gif width="180"></td>
  <td><a href="https://qinghew.github.io/StableIdentity/static/videos/video1/oil_painting.mp4"><img src=assets/oil_painting.gif width="180"></td>
  <tr>
</table >

More results can be found in our [Project Page](https://qinghew.github.io/StableIdentity/) and [Paper](https://arxiv.org/abs/2401.15975).


---

## Getting Started
### Installation
- Requirements (Only need 9GB VRAM for training): If you want to implement StableIdentity & LucidDreamer, you need to clone this repo by: `git clone https://github.com/qinghew/StableIdentity.git --recursive` to download submodules in `LucidDreamer/submodules/`. 
  ```bash
  conda create -n stableid python=3.8.5
  pip install -r requirements_StableIdentity.txt
  ```

- Download pretrained models: [Stable Diffusion v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1/), [face recognition ViT](https://huggingface.co/jayanta/vit-base-patch16-224-in21k-face-recognition).

- Set the paths of pretrained models as default in the Line94 of train.py or command with 
  ```bash
  --pretrained_model_name_or_path **sd2.1_path** --vit_face_recognition_model_path **face_vit_path**
  ```

- Download the [face parsing model](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) into `models/face_parsing/res/cp`.



### Train
- Train for a single test image:
  ```bash
  CUDA_VISIBLE_DEVICES=0 accelerate launch --machine_rank 0 --num_machines 1 --main_process_port 11135 --num_processes 1 --gpu_ids 0 train.py --face_img_path=datasets_face/test_data_demo/00059.png --output_dir="experiments512/save_00059" --resolution=512 --train_batch_size=1 --checkpointing_steps=50 --gradient_accumulation_steps=1 --seed=42 --learning_rate=5e-5 --l_hair_diff_lambda=0.1
  ```

- Train for your test dataset (Preprocess with [FFHQ-Alignment](https://github.com/happy-jihye/FFHQ-Alignment) or cut the headshots):
  ```bash
  bash train_for_testset.sh
  ```


### Test
- **Test StableIdentity**: We provide three test mode "test a single image with a single prompt", "test a single image with prompts" and "test all images with prompts" in [test.ipynb](https://github.com/qinghew/StableIdentity/blob/main/test.ipynb) for developers to use. The results will be generated in `results/{index}/`.


- **Test StableIdentity & ControlNet**: Download the OpenPose's `facenet.pth, body_pose_model.pth, body_pose_model.pth` in [ControlNet's Annotators](https://huggingface.co/lllyasviel/Annotators/tree/main) into `models/openpose_models` and the [ControlNet-SD21](https://huggingface.co/thibaud/controlnet-sd21-openpose-diffusers). 
  ```bash
  # Requirements for ControlNet:
  pip install controlnet_aux
  ```
  The test code is [test_with_controlnet_openpose.ipynb](https://github.com/qinghew/StableIdentity/blob/main/test_with_controlnet_openpose.ipynb). The results will be generated in `results/{index}/with_controlnet/`.


- **Test StableIdentity & LucidDreamer**: 
  ```bash
  # Requirement for LucidDreamer:
  # Clone this repo by: `git clone https://github.com/qinghew/StableIdentity.git --recursive` to download submodules in `LucidDreamer/submodules/`.
  pip install -r requirements_LucidDreamer.txt
  pip install LucidDreamer/submodules/diff-gaussian-rasterization/
  pip install LucidDreamer/submodules/simple-knn/
  
  # test 
  python LucidDreamer/train.py --opt 'LucidDreamer/configs/stableid.yaml'
  ```
  You also could refer the [LucidDreamer's preparations](https://github.com/EnVision-Research/LucidDreamer/blob/main/resources/Training_Instructions.md). We only edit the code at Line 130 in `LucidDreamer/train.py` and set the SD2.1 path and prompts in `LucidDreamer/configs/stableid.yaml` to insert the learned identity into 3D (LucidDreamer). The 3D videos will be generated in `LucidDreamer/output/stableid_{index}/videos/`. 


- **Test StableIdentity & ModelScopeT2V**: Download the ModelScopeT2V's pretrained models in [ModelScopeT2V](https://huggingface.co/ali-vilab/modelscope-damo-text-to-video-synthesis/tree/main) into `modelscope_t2v_files/`. 
  ```bash
  # Requirement for ModelScopeT2V:
  pip install -r requirements_modelscope.txt
  ```
  The test code is [test_with_modelscope.ipynb](https://github.com/qinghew/StableIdentity/blob/main/test_with_modelscope.ipynb). Since the ModelScope library lacks some functions for tokenizer and embedding layer, you need to replace the `anaconda3/envs/**your_envs**/lib/python3.8/site-packages/modelscope/models/multi_modal/video_synthesis/text_to_video_synthesis_model.py` with `modelscope_t2v_files/text_to_video_synthesis_model.py`. The videos will be generated in `results/{index}/with_modelscope/`.


## TODOs
- [x] Release training and inference codes
- [x] Release codes for StableIdentity & ControlNet
- [x] Release codes for StableIdentity & LucidDreamer for Identity-Driven 3D Generation
- [x] Release codes for StableIdentity & ModelScopeT2V for Identity-Driven Video Generation


## Acknowledgements
❤️ Thanks to all the authors of the used repos and pretrained models, let's push AIGC together!


## Citation	

```
@article{wang2024stableidentity,
  title={StableIdentity: Inserting Anybody into Anywhere at First Sight},
  author={Wang, Qinghe and Jia, Xu and Li, Xiaomin and Li, Taiqing and Ma, Liqian and Zhuge, Yunzhi and Lu, Huchuan},
  journal={arXiv preprint arXiv:2401.15975},
  year={2024}
}
```
