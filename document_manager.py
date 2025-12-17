import os
import shutil
import glob
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class DocumentManager:
    def __init__(self, db_path="./db", paper_storage_path="./papers", model_name="./MiniLM"):
        """
        :param db_path: 数据库路径
        :param paper_storage_path: 整理后的论文存放根目录
        """
        self.paper_storage_path = paper_storage_path
        
        # 确保根存放目录存在
        if not os.path.exists(self.paper_storage_path):
            os.makedirs(self.paper_storage_path)

        print("正在加载 Embedding 模型...")
        self.embed_model = SentenceTransformer(model_name)
        
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="paper_collection")

        # 初始化 LLM 
        self.llm_client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key="sk-5262dc4672ad4b68bf71a717f9df0a85",
        )

    def extract_text_from_pdf(self, pdf_path):
        try:
            reader = PdfReader(pdf_path)
            text = ""
            # 遍历每一页提取文本
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        except Exception as e:
            print(f"读取 PDF 失败 {pdf_path}: {e}")
            return None

    def classify_with_llm(self, text_content, topics):
        """调用 LLM 判断分类"""
        # 截取前 2000 字符给大模型判断
        snippet = text_content[:2000]
        prompt = f"""
        请将以下论文摘要归类到以下类别之一: [{topics}]。
        只返回类别名称(例如: CV)，不要加标点，不要解释。
        
        摘要:
        {snippet}
        """
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            category = response.choices[0].message.content.strip()
            return category.replace(".", "").replace("。", "")
        except Exception as e:
            print(f"LLM 分类出错: {e}")
            return "Uncategorized"

    def _move_file(self, source_path, category):
        """物理移动文件到分类文件夹"""
        target_dir = os.path.join(self.paper_storage_path, category)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        filename = os.path.basename(source_path)
        target_path = os.path.join(target_dir, filename)

        try:
            shutil.move(source_path, target_path)
            print(f"文件已移动: {source_path} -> {target_path}")
            return target_path
        except Exception as e:
            print(f"移动文件失败 (可能已存在): {e}")
            # 如果移动失败，通常是因为目标目录已有同名文件，
            # 此时返回原路径或做重命名处理，这里简单返回原路径
            return source_path 

    def add_document(self, file_path, topics_str):
        """单文件处理核心流程"""
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            return

        filename = os.path.basename(file_path)
        print(f"=== 正在处理: {filename} ===")

        # 1. 提取文本 (已替换为 pypdf)
        full_text = self.extract_text_from_pdf(file_path)
        if not full_text: 
            print("警告: 无法提取文本，跳过处理。")
            return

        # 2. LLM 分类
        category = self.classify_with_llm(full_text, topics_str)
        print(f"识别分类: [{category}]")

        # 3. 物理移动文件
        new_path = self._move_file(file_path, category)

        # 4. 存入数据库
        embedding = self.embed_model.encode(full_text[:1000]).tolist()
        
        self.collection.upsert(
            ids=[filename],
            documents=[full_text],
            metadatas=[{
                "filename": filename, 
                "category": category, 
                "filepath": new_path 
            }],
            embeddings=[embedding]
        )
        print("索引构建完成。\n")

    def batch_organize_folder(self, source_folder, topics_str):
        """批量扫描并整理"""
        print(f"正在扫描文件夹: {source_folder} ...")
        pdf_files = glob.glob(os.path.join(source_folder, "*.pdf"))
        
        if not pdf_files:
            print("未找到 PDF 文件。")
            return

        print(f"共发现 {len(pdf_files)} 个 PDF 文件，开始整理...")
        
        for pdf_path in pdf_files:
            try:
                self.add_document(pdf_path, topics_str)
            except Exception as e:
                print(f"处理 {pdf_path} 时发生未知错误: {e}")

    def search(self, query, top_k=3):
        """搜索文件"""
        query_embedding = self.embed_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        output = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                output.append({
                    "filename": meta['filename'],
                    "category": meta['category'],
                    "filepath": meta['filepath'],
                    "score": results['distances'][0][i]
                })
        return output
    