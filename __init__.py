import torch
import numpy as np
from torch.nn.functional import cosine_similarity
import comfy.clip_vision
import folder_paths
from PIL import Image

class ImageCLIPSimilarity:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                # 改用模型名称选择，而非直接传入CLIP_VISION对象（避免封装兼容问题）
                "clip_vision_name": (folder_paths.get_filename_list("clip_vision"),),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity_score", "status_message")
    FUNCTION = "calculate_similarity"
    CATEGORY = "Analysis/Image"
    DESCRIPTION = "Calculates semantic similarity between two images using CLIP Vision embeddings."

    def calculate_similarity(self, image_a, image_b, clip_vision_name):
        # 1. 手动加载CLIP Vision模型（绕过ComfyUI的封装兼容问题）
        clip_vision_path = folder_paths.get_full_path("clip_vision", clip_vision_name)
        clip_vision = comfy.clip_vision.load_clip_vision(clip_vision_path)
        device = clip_vision.device  # 直接用模型的设备，100%对齐

        try:
            # 2. 通用图像预处理函数（适配ComfyUI IMAGE格式）
            def process_image(img_tensor):
                # ComfyUI IMAGE: [B, H, W, C] → 转PIL → 适配CLIP输入
                # 取批次第一张
                if img_tensor.shape[0] > 0:
                    img_tensor = img_tensor[0]
                # 从0-1转0-255，转numpy，再转PIL
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np).convert("RGB")  # 强制RGB
                # 用ComfyUI内置的CLIP图像预处理
                processed = clip_vision.preprocess(pil_img).unsqueeze(0).to(device)
                return processed

            # 3. 处理两张图片
            img_a_processed = process_image(image_a)
            img_b_processed = process_image(image_b)

            # 4. 提取CLIP Embedding（用ComfyUI官方推荐的方式）
            with torch.no_grad():
                # encode_image返回dict，包含image_embeds（核心embedding）
                embed_a = clip_vision.encode_image(img_a_processed)
                embed_b = clip_vision.encode_image(img_b_processed)
                
                # 关键：优先用image_embeds（部分版本叫pooler_output，兼容处理）
                vec_a = embed_a.get("image_embeds", embed_a.get("pooler_output"))
                vec_b = embed_b.get("image_embeds", embed_b.get("pooler_output"))

                # 确保维度正确
                vec_a = vec_a.squeeze()  # 去除多余维度
                vec_b = vec_b.squeeze()

            # 5. 计算余弦相似度（增加维度校验）
            if vec_a.dim() == 1 and vec_b.dim() == 1:
                similarity_score = cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1).item()
            else:
                similarity_score = cosine_similarity(vec_a, vec_b, dim=-1).mean().item()

            # 6. 生成状态信息
            status = f"CLIP Similarity: {similarity_score:.4f}"
            if similarity_score > 0.90:
                status += " (几乎相同 / Nearly Identical)"
            elif similarity_score > 0.80:
                status += " (高度相似 / Highly Similar)"
            elif similarity_score > 0.65:
                status += " (中度相似 / Moderately Similar)"
            elif similarity_score > 0.45:
                status += " (低度相似 / Low Similarity)"
            else:
                status += " (不相关 / Unrelated)"

            return (similarity_score, status)

        except Exception as e:
            error_msg = f"计算失败: {str(e)}"
            return (0.0, error_msg)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageCLIPSimilarity": ImageCLIPSimilarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCLIPSimilarity": "Image CLIP Similarity"
}
