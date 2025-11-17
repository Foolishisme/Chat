"""
图片处理模块 - V2 多模态嵌入版本
使用CLIP模型直接向量化图片，无信息损失
"""
import os
import io
from typing import List, Dict
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

from app.config import settings


class MultimodalImageProcessor:
    """多模态图片处理器 - 使用CLIP直接向量化图片"""
    
    def __init__(self):
        """
        初始化多模态图片处理器
        使用CLIP模型实现文本和图片在统一向量空间
        """
        print("正在初始化CLIP多模态模型...")
        
        # 加载CLIP模型（文本和图片共享向量空间）
        # 使用ViT-B/32版本：速度快，效果好，512维向量
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # 设置为评估模式
        self.model.eval()
        
        # 如果有GPU，使用GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        print(f"✅ CLIP模型加载完成！")
        print(f"   - 模型: {self.model_name}")
        print(f"   - 设备: {self.device}")
        print(f"   - 向量维度: 512")
        
        # 图片保存目录
        self.image_save_dir = Path("data/images")
        self.image_save_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从PDF中提取所有图片
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            图片信息列表，每个元素包含：
            - page: 页码
            - image_index: 图片在该页的索引
            - image: PIL Image对象
            - image_path: 保存的图片路径
            - width, height: 图片尺寸
        """
        images_info = []
        
        try:
            # 打开PDF文件
            pdf_document = fitz.open(pdf_path)
            print(f"开始从PDF提取图片，共 {len(pdf_document)} 页")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                if not image_list:
                    continue
                    
                print(f"第 {page_num + 1} 页发现 {len(image_list)} 张图片")
                
                for img_index, img in enumerate(image_list):
                    try:
                        # 获取图片的xref
                        xref = img[0]
                        
                        # 提取图片数据
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # 转换为PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # 转换为RGB模式（CLIP需要RGB）
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        # 过滤太小的图片（可能是图标或装饰）
                        if pil_image.width < 100 or pil_image.height < 100:
                            print(f"  - 跳过小图片：{pil_image.width}x{pil_image.height}")
                            continue
                        
                        # 保存图片到本地
                        pdf_name = Path(pdf_path).stem
                        image_filename = f"{pdf_name}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                        image_path = self.image_save_dir / image_filename
                        
                        pil_image.save(image_path)
                        print(f"  - 提取图片：{image_filename} ({pil_image.width}x{pil_image.height})")
                        
                        images_info.append({
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "image": pil_image,
                            "image_path": str(image_path),
                            "width": pil_image.width,
                            "height": pil_image.height,
                        })
                        
                    except Exception as e:
                        print(f"  - 提取第 {page_num + 1} 页第 {img_index + 1} 张图片失败: {str(e)}")
                        continue
            
            pdf_document.close()
            print(f"✅ 图片提取完成，共提取 {len(images_info)} 张有效图片")
            return images_info
            
        except Exception as e:
            print(f"❌ 提取图片时发生错误: {str(e)}")
            return []
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        使用CLIP获取图片的向量表示
        
        Args:
            image: PIL Image对象
            
        Returns:
            512维向量 (CLIP ViT-B/32)
        """
        try:
            # 预处理图片
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成图片向量
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # L2归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            embedding = image_features.cpu().numpy().flatten()
            return embedding
            
        except Exception as e:
            print(f"❌ 图片向量化失败: {str(e)}")
            # 返回零向量作为fallback
            return np.zeros(512, dtype=np.float32)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        使用CLIP获取文本的向量表示（与图片在同一向量空间）
        用于查询向量化，实现跨模态检索
        
        Args:
            text: 文本内容
            
        Returns:
            512维向量
        """
        try:
            # 预处理文本
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成文本向量
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # L2归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            embedding = text_features.cpu().numpy().flatten()
            return embedding
            
        except Exception as e:
            print(f"❌ 文本向量化失败: {str(e)}")
            # 返回零向量作为fallback
            return np.zeros(512, dtype=np.float32)
    
    def process_pdf_images(self, pdf_path: str) -> List[Dict]:
        """
        处理PDF中的所有图片：提取 + CLIP向量化
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            处理后的图片信息列表，每个元素包含：
            - page: 页码
            - image_index: 图片索引
            - image_path: 图片路径
            - embedding: CLIP向量 (512维)
            - width, height: 图片尺寸
        """
        # 1. 提取图片
        images_info = self.extract_images_from_pdf(pdf_path)
        
        if not images_info:
            print("PDF中没有找到有效图片")
            return []
        
        # 2. 使用CLIP向量化每张图片
        print(f"\n开始使用CLIP向量化 {len(images_info)} 张图片...")
        for i, img_info in enumerate(images_info, 1):
            print(f"正在处理第 {i}/{len(images_info)} 张图片...")
            
            # 直接向量化，无需AI描述！
            embedding = self.get_image_embedding(img_info["image"])
            img_info["embedding"] = embedding
            
            print(f"  ✅ 向量维度: {embedding.shape}")
        
        print("\n✅ 所有图片向量化完成！")
        return images_info
    
    def get_image_documents_for_vectorization(self, images_info: List[Dict]) -> List[Dict]:
        """
        将图片信息转换为可向量化的文档格式
        注意：这里只保存元数据，向量由CLIP生成
        
        Args:
            images_info: 图片信息列表（包含embedding）
            
        Returns:
            文档列表，用于向量数据库存储
        """
        documents = []
        
        for img_info in images_info:
            # 只保存元数据，不需要文字描述
            # 向量已经由CLIP生成
            documents.append({
                "embedding": img_info["embedding"],  # CLIP向量
                "page_content": f"[图片 - 第{img_info['page']}页]",  # 简短标识
                "metadata": {
                    "page": img_info['page'],
                    "source": "image",
                    "image_path": img_info['image_path'],
                    "image_index": img_info['image_index'],
                    "type": "image",
                    "width": img_info['width'],
                    "height": img_info['height']
                }
            })
        
        return documents


# 创建全局实例
def create_image_processor():
    """创建图片处理器实例"""
    return MultimodalImageProcessor()

