# #### example from leloykun for image generation ####
# import torch
# from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
# import torch.nn.functional as F

# processor = ChameleonProcessor.from_pretrained("/liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo")
# model = ChameleonForConditionalGeneration.from_pretrained(
#     "/liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

# # Prepare a prompt
# prompt = "Generate an image of a snowman."

# # Preprocess the prompt
# inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)

# # Generate discrete image tokens
# output = model.generate(
#     **inputs,
#     generate_wo_quant=True,
#     multimodal_generation_mode="image-only",
#     # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
#     max_new_tokens=1026,
#     # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
#     do_sample=True,
#     repetition_penalty=1.0,
#     use_cache=True,
#     output_logits=True,
#     return_dict_in_generate=True,
# )
# generate_ids = output['sequences']

# # Only keep the tokens from the response
# response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
# print(response_ids)

# # Decode the generated image tokens
# pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
# images = processor.postprocess_pixel_values(pixel_values)

# # Save the image
# from torchvision.transforms.functional import to_pil_image
# images = [to_pil_image(img.detach().cpu()) for img in images]
# images[0].save("snowman_test.png")


# # check if the image is still ok if we dont use quantization
# T = 1
# output_logits = torch.stack(output['logits'][1:-1], dim=1)
# weighted_indices = F.softmax(output_logits / T, dim=-1)
# image_embeds_wo_quant = torch.matmul(weighted_indices[0, :, 4:8196].bfloat16(), model.model.vqmodel.quantize.embedding.weight)
# # print(image_embeds_wo_quant.shape)
# emb_dim: int = model.model.vqmodel.quantize.embedding.weight.shape[-1]
# image_embeds_wo_quant = image_embeds_wo_quant.view((1, *model.model.vqmodel.quantize.quant_state_dims, emb_dim))
# image_embeds_wo_quant = image_embeds_wo_quant.permute(0, 3, 1, 2).contiguous()
# hidden_states = model.model.vqmodel.post_quant_conv(image_embeds_wo_quant)
# pixel_values_wo_quant = model.model.vqmodel.decoder(hidden_states)
# images_wo_quant = processor.postprocess_pixel_values(pixel_values_wo_quant)

# # Save the image
# from torchvision.transforms.functional import to_pil_image
# images_wo_quant = [to_pil_image(img.detach().cpu()) for img in images_wo_quant]
# images_wo_quant[0].save("snowman_wo_quant.png")


# ##### code for image understanding #####
# from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
# import torch
# import requests
# from PIL import Image

# device = 'cuda'
# model = ChameleonForConditionalGeneration.from_pretrained("/liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo", torch_dtype=torch.bfloat16).to(device)
# processor = ChameleonProcessor.from_pretrained("/liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo")

# prompt = "Describe this image for me.<image>"
# image = Image.open("/liymai24/sjtu/siqi/IP-Adapter-main/assets/images/river.png")

# inputs = processor(prompt, images=image, return_tensors="pt").to(model.device, torch.bfloat16)

# generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False) 
# out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(out)





######## new version of image generation ########

# import torch
# from models.processing_chameleon_wo_quant import ChameleonProcessor
# from models.modeling_chameleon_wo_quant import ChameleonForConditionalGeneration
# import torch.nn.functional as F
# from safetensors.torch import load_file

# ckpt_path = "/liymai24/sjtu/siqi/leloykun/outputs/mse_only_head_codebook_trainable/checkpoint-500"
# processor = ChameleonProcessor.from_pretrained(ckpt_path)
# model = ChameleonForConditionalGeneration.from_pretrained(
#     ckpt_path,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

# if 'checkpoint' in ckpt_path:
#     # codebook_in, codebook_out are trained already
#     pass
# else:
#     # initialize codebook_in, codebook_out
#     state_dict = load_file('/liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo/model-00003-of-00003.safetensors')
#     origin_codebook = state_dict["model.vqmodel.quantize.embedding.weight"]
#     for param in model.model.codebook_in.parameters():
#         param.data.copy_(origin_codebook)
#     for param in model.codebook_out.parameters():
#         param.data.copy_(origin_codebook)

# # Prepare a prompt
# prompt = "Give me an image of a snowman."

# # Preprocess the prompt
# inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)

# # Generate discrete image tokens
# output = model.generate(
#     **inputs,
#     multimodal_generation_mode="image-only",
#     # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
#     max_new_tokens=1026,
#     # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
#     do_sample=True,
#     use_cache=False,
# )
# print(output.shape)
# image_embeds_wo_quant = output.squeeze(dim=1)
# emb_dim: int = model.model.vqmodel.quantize.embedding.weight.shape[-1]
# image_embeds_wo_quant = image_embeds_wo_quant.view((1, *model.model.vqmodel.quantize.quant_state_dims, emb_dim))
# image_embeds_wo_quant = image_embeds_wo_quant.permute(0, 3, 1, 2).contiguous()
# hidden_states = model.model.vqmodel.post_quant_conv(image_embeds_wo_quant)
# pixel_values_wo_quant = model.model.vqmodel.decoder(hidden_states)
# images_wo_quant = processor.postprocess_pixel_values(pixel_values_wo_quant)

# # Save the image
# from torchvision.transforms.functional import to_pil_image
# images_wo_quant = [to_pil_image(img.detach().cpu()) for img in images_wo_quant]
# images_wo_quant[0].save("test.png")


######## new version of image understanding ########

import torch
from models.processing_chameleon_wo_quant import ChameleonProcessor
from models.modeling_chameleon_wo_quant import ChameleonForConditionalGeneration
import torch.nn.functional as F
import requests
from PIL import Image
from safetensors.torch import load_file

device = 'cuda'
ckpt_path = "/liymai24/sjtu/siqi/leloykun/outputs/mse_only_head_codebook_trainable_T1/checkpoint-30000"
processor = ChameleonProcessor.from_pretrained(ckpt_path)
model = ChameleonForConditionalGeneration.from_pretrained(
    ckpt_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

if 'checkpoint' in ckpt_path:
    # codebook_in, codebook_out are trained already
    pass
else:
    # initialize codebook_in, codebook_out
    state_dict = load_file('/liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo/model-00003-of-00003.safetensors')
    origin_codebook = state_dict["model.vqmodel.quantize.embedding.weight"]
    for param in model.model.codebook_in.parameters():
        param.data.copy_(origin_codebook)
    for param in model.codebook_out.parameters():
        param.data.copy_(origin_codebook)

prompt = "<image>Describe this image for me."
image = Image.open("/liymai24/sjtu/siqi/IP-Adapter-main/assets/images/girl.png")

inputs = processor(prompt, images=image, return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)

generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True, use_cache=False) 
out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(out)

# from safetensors.torch import load_file

# state_dict_origin =  load_file('/liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo-vae135/model-00003-of-00003.safetensors')
# origin_codebook = state_dict_origin["model.vqmodel.quantize.embedding.weight"]
# print(origin_codebook)
# origin_lm_head = state_dict_origin["lm_head.weight"]
# print(f'origin lm_head.weight: {origin_lm_head[4:8196]}')

# state_dict_trained = load_file('/liymai24/sjtu/siqi/leloykun/outputs/mse_only_head_codebook_trainable_random_init/checkpoint-16670/model-00006-of-00006.safetensors')
# trained_lm_head = state_dict_trained['lm_head.weight']
# print(f'trained lm_head: {trained_lm_head[4:8196]}')
# trained_out = state_dict_trained['codebook_out.weight']
# print(f'trained codebook_out: {trained_out}')
# init_in = state_dict_trained['model.codebook_in.weight']
# print(f'init codebook_in: {init_in}')
