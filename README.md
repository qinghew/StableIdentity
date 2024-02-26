<div align="center">

## StableIdentity: Inserting Anybody into Anywhere at First Sight
🤗[[Paper](https://huggingface.co/papers/2401.15975)] &emsp; 🔥[[Project Page](https://qinghew.github.io/StableIdentity/)]
</div>

## News
- **[2024.02.25]**: Release training and inference codes!


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
- Requirements (Only need 9GB VRAM for training):
```bash
conda create -n stableid python=3.8.5
pip install -r requirements.txt
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
We provide three test mode "test a single image with a single prompt", "test a single image with prompts" and "test all images with prompts" in [test.ipynb](https://github.com/qinghew/StableIdentity/blob/main/test.ipynb) for developers to use.



## TODOs
- [x] Release training and inference codes
- [ ] Release codes for StableIdentity & Image/Video/3D generation models



### Citation	

```
@article{wang2024stableidentity,
  title={StableIdentity: Inserting Anybody into Anywhere at First Sight},
  author={Wang, Qinghe and Jia, Xu and Li, Xiaomin and Li, Taiqing and Ma, Liqian and Zhuge, Yunzhi and Lu, Huchuan},
  journal={arXiv preprint arXiv:2401.15975},
  year={2024}
}
```
