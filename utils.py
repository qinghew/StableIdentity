from PIL import Image

import torch
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask
import torch.nn as nn
from torch import autograd

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def add_noise_return_paras(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples, sqrt_alpha_prod, sqrt_one_minus_alpha_prod
                
                
def text_encoder_forward(
    text_encoder = None,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    embedding_manager = None,
    only_embedding=False,
    face_img_embeddings = None,
    timesteps = None,
):
    output_attentions = output_attentions if output_attentions is not None else text_encoder.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else text_encoder.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else text_encoder.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    # see embedding_forward() in models/celeb_embeddings.py
    hidden_states = text_encoder.text_model.embeddings(input_ids=input_ids, position_ids=position_ids,
                                                                          embedding_manager=embedding_manager,
                                                                          only_embedding=only_embedding,
                                                                          face_img_embeddings = face_img_embeddings,
                                                                          timesteps = timesteps,
                                                                          )

    if only_embedding:
        return hidden_states


    causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)

    if text_encoder.text_model.eos_token_id == 2:
        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
    else:
        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == text_encoder.text_model.eos_token_id)
            .int()
            .argmax(dim=-1),
        ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )[0]


def downsampling(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()


def image_grid(images, rows=2, cols=2):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def latents_to_images(vae, latents, scale_factor=0.18215):
    """
    Decode latents to PIL images.
    """
    scaled_latents = 1.0 / scale_factor * latents.clone()
    images = vae.decode(scaled_latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()

    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def latents_to_images_tensor(vae, latents, scale_factor=0.18215):
    """
    Decode latents to PIL images.
    """
    scaled_latents = 1.0 / scale_factor * latents.clone()
    images = vae.decode(scaled_latents).sample

    return images
