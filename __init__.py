import torch
from torch.nn.functional import cosine_similarity
import comfy.clip_vision

class ImageCLIPSimilarityPure:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "clip_vision_model": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity_score",)
    FUNCTION = "calculate_similarity"
    CATEGORY = "Analysis/Image"

    def calculate_similarity(self, image_a, image_b, clip_vision_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 1. 预处理：确保只处理单张图片 [1, H, W, C]
            if len(image_a.shape) == 4 and image_a.shape[0] > 1:
                image_a = image_a[0].unsqueeze(0)
            if len(image_b.shape) == 4 and image_b.shape[0] > 1:
                image_b = image_b[0].unsqueeze(0)

            # 2. 编码
            print("\n[DEBUG] Encoding images...")
            embed_a = clip_vision_model.encode_image(image_a)
            embed_b = clip_vision_model.encode_image(image_b)

            # 3. 【核心修复】正确提取 Output 对象中的数据
            def extract_vector(embed_obj):
                # 检查是否是 comfy.clip_vision.Output 对象
                if hasattr(embed_obj, 'last_hidden_state'):
                    # 大多数 CLIP Vision 模型主要输出 last_hidden_state
                    # 形状通常是 [B, Seq_Len, Dim]，我们需要对 Seq_Len 做平均池化得到全局向量
                    vec = embed_obj.last_hidden_state
                    print(f"[DEBUG] Extracted 'last_hidden_state', shape: {vec.shape}")
                    # Mean Pooling: 在维度 1 (序列长度) 上取平均
                    return vec.mean(dim=1)
                
                elif hasattr(embed_obj, 'pooler_output'):
                    # 某些模型可能有直接的 pooler_output [B, Dim]
                    vec = embed_obj.pooler_output
                    print(f"[DEBUG] Extracted 'pooler_output', shape: {vec.shape}")
                    return vec
                
                elif isinstance(embed_obj, dict):
                    # 兼容旧版本或特殊情况
                    if 'last_hidden_state' in embed_obj:
                        return embed_obj['last_hidden_state'].mean(dim=1)
                    if 'pooler_output' in embed_obj:
                        return embed_obj['pooler_output']
                    raise KeyError(f"Dict keys found: {list(embed_obj.keys())}")
                
                else:
                    # 如果都不是，打印所有属性以便调试
                    attrs = [attr for attr in dir(embed_obj) if not attr.startswith('_')]
                    raise TypeError(f"Unknown embedding type: {type(embed_obj)}. Available attributes: {attrs}")

            vec_a = extract_vector(embed_a)
            vec_b = extract_vector(embed_b)

            # 4. 维度修正与检查
            if vec_a.dim() == 3:
                vec_a = vec_a.squeeze(1)
            if vec_b.dim() == 3:
                vec_b = vec_b.squeeze(1)

            print(f"[DEBUG] Final Vector A shape: {vec_a.shape}, Sample: {vec_a[0][:5]}")
            print(f"[DEBUG] Final Vector B shape: {vec_b.shape}, Sample: {vec_b[0][:5]}")

            if torch.all(vec_a == 0) or torch.all(vec_b == 0):
                raise ValueError("Vectors are all zeros! Check model or input.")

            # 5. 计算相似度
            with torch.no_grad():
                if vec_a.device != vec_b.device:
                    vec_b = vec_b.to(vec_a.device)
                
                sim_tensor = cosine_similarity(vec_a, vec_b, dim=1)
                score = float(sim_tensor[0].item())

            print(f"[DEBUG] !!! FINAL SCORE: {score} !!!\n")

            return (score,)

        except Exception as e:
            print(f"\n!!! FATAL ERROR !!!")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print("!!! END ERROR !!!\n")
            raise e

NODE_CLASS_MAPPINGS = {
    "ImageCLIPSimilarityPure": ImageCLIPSimilarityPure
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCLIPSimilarityPure": "CLIP Similarity (Fixed Output)"
}
