import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.models.dinov3_vit import DINOv3ViTBackbone
from transformers.models.dinov2 import Dinov2Backbone
from transformers.image_utils import load_image


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")


model = AutoModel.from_pretrained(
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    dtype=torch.bfloat16,
    device_map="auto",
    output_hidden_states=True
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
print(model.device)
print(inputs['pixel_values'].shape)
with torch.inference_mode():
    outputs = model(**inputs)

print(len(outputs.hidden_states))
a=[1,2,3,4]
print(outputs.hidden_states[0].shape)
print(outputs.hidden_states[1].shape)


# patch_size = model.config.patch_size
# batch_size, _, img_height, img_width = inputs.pixel_values.shape
# num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
# num_patches_flat = num_patches_height * num_patches_width

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape) 
# cls_token = last_hidden_states[:, 0, :]
# patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
# print(patch_features_flat.shape)