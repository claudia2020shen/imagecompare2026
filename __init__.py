import torch
from torch.nn.functional import cosine_similarity
import comfy.clip_vision
import folder_paths
# 新增：导入CLIP的图像预处理
from transformers import CLIPImageProcessor

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
    DESCRIPTION = "Calculates semantic similarity between two images using CLIP Vision embeddings."

    def calculate_similarity(self, image_a, image_b, clip_vision_model):
        device = next(clip_vision_model.parameters()).device if hasattr(clip_vision_model, 'parameters') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 初始化CLIP图像处理器（原生方式，兼容所有CLIP Vision模型）
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

            def preprocess_image_comfy(image):
                # 1. 处理ComfyUI图像：[B, H, W, C] → [H, W, C]（单张）
                if image.shape[0] > 1:
                    image = image[0]  # 取第一张
                # 2. 转换为PIL图像（原生CLIP预处理需要）
                image_np = (image.cpu().numpy() * 255).astype("uint8")
                from PIL import Image
                pil_image = Image.fromarray(image_np)
                # 3. 用CLIP原生处理器预处理
                inputs = image_processor(images=pil_image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                return pixel_values

            # 预处理两张图片
            img_a = preprocess_image_comfy(image_a)
            img_b = preprocess_image_comfy(image_b)

            # 手动编码图像（绕过ComfyUI的封装）
            with torch.no_grad():
                outputs_a = clip_vision_model.model(pixel_values=img_a)
                outputs_b = clip_vision_model.model(pixel_values=img_b)
                # 提取embedding（用CLIP官方的方式：last_hidden_state的均值）
                vec_a = outputs_a.last_hidden_state.mean(dim=1)
                vec_b = outputs_b.last_hidden_state.mean(dim=1)

            # 计算余弦相似度
            sim_tensor = cosine_similarity(vec_a, vec_b, dim=1)
            similarity_score = sim_tensor.item()

            # 生成状态信息
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
            error_msg = f"Error: {str(e)}"
            return (0.0, error_msg)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageCLIPSimilarity": ImageCLIPSimilarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCLIPSimilarity": "Image CLIP Similarity"
}
