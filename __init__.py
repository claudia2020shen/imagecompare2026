import torch
from torch.nn.functional import cosine_similarity
import comfy.clip_vision

class ImageCLIPSimilarity:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "clip_vision_model": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity_score", "status_message")
    FUNCTION = "calculate_similarity"
    CATEGORY = "Analysis/Image"

    def calculate_similarity(self, image_a, image_b, clip_vision_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Debug] Using device: {device}")

        try:
            # 1. 预处理图片 (取第一张)
            if image_a.shape[0] > 1:
                image_a = image_a[0].unsqueeze(0)
            if image_b.shape[0] > 1:
                image_b = image_b[0].unsqueeze(0)
            
            print(f"[Debug] Input shapes: A={image_a.shape}, B={image_b.shape}")

            # 2. 编码图片
            # 注意：ComfyUI 的 encode_image 通常会自动处理 device 转移，但为了保险，我们传入前确保类型正确
            # 不需要手动 .to(device)，因为 comfy.clip_vision 内部会处理，手动转有时反而导致维度错乱
            
            print("[Debug] Encoding image A...")
            embed_a = clip_vision_model.encode_image(image_a)
            print("[Debug] Encoding image B...")
            embed_b = clip_vision_model.encode_image(image_b)

            # 3. 【关键修复】动态提取向量
            # 不同的 CLIP 模型返回的字典键名可能不同，我们需要智能判断
            
            def get_vector(embed_dict):
                # 情况 A: 直接是字典
                if isinstance(embed_dict, dict):
                    print(f"[Debug] Embed keys found: {embed_dict.keys()}")
                    
                    # 优先找 pooler_output (最常用)
                    if 'pooler_output' in embed_dict:
                        vec = embed_dict['pooler_output']
                        print("[Debug] Using 'pooler_output'")
                        return vec
                    
                    # 其次找 last_hidden_state (需要 pooling)
                    if 'last_hidden_state' in embed_dict:
                        vec = embed_dict['last_hidden_state']
                        print("[Debug] Using 'last_hidden_state' with mean pooling")
                        # 对序列维度取平均 (假设形状是 [B, Seq_Len, Dim])
                        return vec.mean(dim=1)
                    
                    # 兼容某些旧版本或特殊格式
                    if 'image_embeds' in embed_dict:
                         print("[Debug] Using 'image_embeds'")
                         return embed_dict['image_embeds']

                    # 如果都没找到，列出所有 key 并报错
                    raise KeyError(f"Unknown embedding format. Available keys: {list(embed_dict.keys())}")
                
                # 情况 B: 某些版本可能返回一个对象，其属性在 .output 里
                elif hasattr(embed_dict, 'output'):
                    return get_vector(embed_dict.output)
                
                else:
                    raise TypeError(f"Unsupported embedding type: {type(embed_dict)}")

            vec_a = get_vector(embed_a)
            vec_b = get_vector(embed_b)

            # 4. 形状修正与检查
            # 确保是 [Batch, Dim]
            if vec_a.dim() == 3:
                vec_a = vec_a.squeeze(1)
            if vec_b.dim() == 3:
                vec_b = vec_b.squeeze(1)
            
            print(f"[Debug] Vector shapes: A={vec_a.shape}, B={vec_b.shape}")
            print(f"[Debug] Vector A sample (first 5 values): {vec_a[0][:5]}")
            print(f"[Debug] Vector B sample (first 5 values): {vec_b[0][:5]}")

            # 检查是否全为 0 (这是导致相似度为 0 的直接原因)
            if torch.all(vec_a == 0) or torch.all(vec_b == 0):
                raise ValueError("Detected all-zero vectors! Model might be unloaded or input invalid.")

            # 5. 计算相似度
            with torch.no_grad():
                # 确保在同一设备
                if vec_a.device != vec_b.device:
                    vec_b = vec_b.to(vec_a.device)
                
                sim_tensor = cosine_similarity(vec_a, vec_b, dim=1)
                score = float(sim_tensor[0].item())

            print(f"[Debug] !!! FINAL SCORE: {score} !!!")

            status = f"Score: {score:.4f}"
            return (score, status)

        except Exception as e:
            # 【重要】打印完整错误堆栈，不再静默返回 0
            print(f"\n!!! CRITICAL ERROR IN SIMILARITY NODE !!!")
            print(f"Error Message: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"!!! END ERROR !!!\n")
            
            # 返回 0 和错误信息，让用户知道出错了
            return (0.0, f"ERROR: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "ImageCLIPSimilarity": ImageCLIPSimilarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCLIPSimilarity": "Image CLIP Similarity (Fixed)"
}
