from torch import nn
from .CLIP import CLIP
from .TrafficCLIP import TrafficCLIP


def build_model(name: str, state_dict: dict,details=None):
    # 通过检查 state_dict 中是否包含 visual.proj 键来确定模型是否使用 Vision Transformer（ViT）。如果包含该键，说明模型使用的是 Vision Transformer。
    vit = "visual.proj" in state_dict
    # 根据模型类型提取和计算视觉模型参数
    if vit:
        # 'vision_width：提取第一个卷积层的输出通道数。
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    # 提取文本模型参数
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64  # Transformer 的头数（每 64 个宽度一个头）
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    # print('name', name)
    # if 'CS-' in name:


    # 构建模型
    if details is None:
        print("This is TrafficCLIP")
        model = TrafficCLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    else:
        print("This is CLIP")
        model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    #convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()

