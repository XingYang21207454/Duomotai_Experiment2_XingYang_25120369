import os
import shutil 
import glob
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb

class ImageManager:
    def __init__(self, db_path="./db", storage_path="./loaded_images", model_name="./clip_model"):
        """
        :param db_path: 向量数据库路径
        :param storage_path: 图片归档存放的目录
        """
        print(f"正在加载 CLIP 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # 确保目标存放目录存在
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="image_collection")

    def _get_image_embedding(self, image_path):
        """生成图片向量"""
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0].tolist()
        except Exception as e:
            print(f"处理图片失败 {image_path}: {e}")
            return None
        

    def _get_text_embedding(self, text):
        """生成文本的向量 (使用 CLIP 的 Text Encoder)"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()


    def _move_image(self, source_path):
        filename = os.path.basename(source_path)
        target_path = os.path.join(self.storage_path, filename)
        
        # 处理同名文件冲突：如果目标已存在，则重命名
        if os.path.exists(target_path) and source_path != target_path:
            name, ext = os.path.splitext(filename)
            target_path = os.path.join(self.storage_path, f"{name}_copy{ext}")

        try:
            shutil.move(source_path, target_path)
            return target_path
        except Exception as e:
            print(f"移动图片失败: {e}")
            return source_path


    def add_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"错误: 图片不存在 {image_path}")
            return

        filename = os.path.basename(image_path)
        print(f"正在处理图片: {filename} ...")
        
        # 1. 先生成向量 (必须在移动前，或者传入移动后的路径)
        embedding = self._get_image_embedding(image_path)
        
        if embedding:
            # 2. 物理移动文件
            new_path = self._move_image(image_path)
            
            # 3. 存入数据库 (存储新路径 new_path)
            self.collection.upsert(
                ids=[filename],
                metadatas={
                    "filename": filename, 
                    "path": os.path.abspath(new_path) # 存储绝对路径，防止出错
                },
                embeddings=[embedding]
            )
            print(f"图片已整理至: {new_path}")


    def add_folder(self, folder_path):
        """批量添加文件夹下的所有图片"""
        print(f"正在扫描图片文件夹: {folder_path}")
        # 支持常见图片格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        print(f"找到 {len(image_files)} 张图片，开始入库...")
        for img_path in image_files:
            self.add_image(img_path)


    def search_image(self, query, top_k=3):
        """以文搜图"""
        print(f"正在搜索匹配描述: '{query}' 的图片...")
        
        # 1. 将文本描述转换为向量
        text_embedding = self._get_text_embedding(query)
        
        # 2. 在数据库中检索
        results = self.collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k
        )
        
        output = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                score = results['distances'][0][i] # 距离越小越相似 (Chroma 默认 L2)
                # CLIP 使用 cosine similarity 效果更好，但 L2 距离也能用 (距离越小越好)
                output.append({
                    "filename": meta['filename'],
                    "path": meta['path'],
                    "score": score
                })
        return output