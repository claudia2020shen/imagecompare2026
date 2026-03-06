import torch
from torch.nn.functional import cosine_similarity

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

    # 【修改点】只返回 FLOAT，不再返回 STRING
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

            # 2. 编码 (关键步骤)
            # 我们直接调用 encode_image，并打印它返回的所有内容
            print("\n" + "="*30)
            print("[DEBUG] Encoding Image A...")
            embed_a = clip_vision_model.encode_image(image_a)
            
            print("[DEBUG] Encoding Image B...")
            embed_b = clip_vision_model.encode_image(image_b)

            # 【核心调试】打印返回对象的类型和键名
            # 很多时候结果为 0 是因为我们猜错了键名，导致取到了空值
            print(f"[DEBUG] Type of embed_a: {type(embed_a)}")
            
            keys_a = []
            if isinstance(embed_a, dict):
                keys_a = list(embed_a.keys())
                print(f"[DEBUG] Keys found in embed_a: {keys_a}")
            elif hasattr(embed_a, '__dict__'):
                print(f"[DEBUG] Attributes found: {embed_a.__dict__.keys()}")
            
            # 3. 智能提取向量 (尝试所有可能的键名)
            def extract_vector(embed):
                if isinstance(embed, dict):
                    # 尝试常见的键名顺序
                    possible_keys = ['pooler_output', 'last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                    for key in possible_keys:
                        if key in embed:
                            print(f"[DEBUG] Successfully extracted using key: '{key}'")
                            vec = embed[key]
                            # 如果是 last_hidden_state (通常是 [B, Seq, Dim])，需要 pooling (取平均)
                            if key == 'last_hidden_state' or key == 'penultimate_hidden_states':
                                return vec.mean(dim=1) 
                            return vec
                    
                    # 如果都没找到，抛出详细错误
                    raise KeyError(f"Could not find known vector keys. Available keys: {list(embed.keys())}")
                else:
                    raise TypeError(f"Unsupported embedding type: {type(embed)}. Expected dict.")

            vec_a = extract_vector(embed_a)
            vec_b = extract_vector(embed_b)

            # 4. 维度修正 (确保是 [Batch, Dim])
            if vec_a.dim() == 3:
                vec_a = vec_a.squeeze(1)
            if vec_b.dim() == 3:
                vec_b = vec_b.squeeze(1)

            # 【关键检查】打印向量前几个数，确认不是全 0
            print(f"[DEBUG] Vector A shape: {vec_a.shape}, Sample values: {vec_a[0][:5]}")
            print(f"[DEBUG] Vector B shape: {vec_b.shape}, Sample values: {vec_b[0][:5]}")

            if torch.all(vec_a == 0) or torch.all(vec_b == 0):
                raise ValueError("Vectors are all zeros! Check model loading or input images.")

            # 5. 计算余弦相似度
            with torch.no_grad():
                # 确保在同一设备
                if vec_a.device != vec_b.device:
                    vec_b = vec_b.to(vec_a.device)
                
                # dim=1 表示在特征维度上计算相似度
                sim_tensor = cosine_similarity(vec_a, vec_b, dim=1)
                score = float(sim_tensor[0].item())

            print(f"[DEBUG] !!! FINAL SCORE: {score} !!!")
            print("="*30 + "\n")

            # 【修改点】只返回数值
            return (score,)

        except Exception as e:
            # 发生任何错误，打印堆栈并中断，绝不静默返回 0
            print(f"\n!!! FATAL ERROR IN NODE !!!")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print("!!! END ERROR !!!\n")
            # 抛出异常让 ComfyUI 界面显示红色错误框，而不是输出 0
            raise e

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageCLIPSimilarityPure": ImageCLIPSimilarityPure
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCLIPSimilarityPure": "CLIP Similarity (Pure Float)"
}
