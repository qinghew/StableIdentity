import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia
import numpy as np

def embedding_forward(
        self,
        input_ids = None,
        position_ids = None,
        inputs_embeds = None,
        embedding_manager = None,
        only_embedding=True,
        face_img_embeddings = None,
        timesteps = None,
    ) -> torch.Tensor:

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)          
            if only_embedding:      
                return inputs_embeds

        if embedding_manager is not None:
            inputs_embeds = embedding_manager(input_ids, inputs_embeds, face_img_embeddings, timesteps)
  
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]        
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
    
        return embeddings


@torch.no_grad()
def _get_celeb_embeddings_basis(tokenizer, text_encoder, good_names_txt):
    
    device = text_encoder.device
    max_length = 77
    
    with open(good_names_txt, "r") as f:
        celeb_names = f.read().splitlines()


    ''' get tokens and embeddings '''
    embeddings_list = []
    for name in celeb_names:
        batch_encoding = tokenizer(name, truncation=True, return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(device)[:, 1:3]  
        embeddings = text_encoder.text_model.embeddings(input_ids=tokens, only_embedding=True)  

        embeddings_list.append(embeddings)
    all_embeddings: torch.Tensor = torch.cat(embeddings_list, dim=0)
    print('[all_embeddings loaded] shape =', all_embeddings.shape,
            'max:', all_embeddings.max(),
            'min:', all_embeddings.min())  
    
    # torch.save(all_embeddings, "all_celeb_embeddings.pt")
    name_emb_mean = all_embeddings.mean(0)
    name_emb_std = all_embeddings.std(0)        

    print('[name_emb_mean loaded] shape =', name_emb_mean.shape,
            'max:', name_emb_mean.max(),
            'min:', name_emb_mean.min()) 
    return name_emb_mean, name_emb_std


