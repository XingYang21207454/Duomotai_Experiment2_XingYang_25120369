import argparse
import sys
import os


from document_manager import DocumentManager
from image_manager import ImageManager

DEFAULT_TOPICS = "CV, NLP, RL, IoT"

def main():
    parser = argparse.ArgumentParser(
        description="本地多模态 AI 智能助手 (Local Multimodal AI Agent)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ==========================
    # 命令 1: 添加/分类单个论文
    # ==========================
    add_paper_parser = subparsers.add_parser("add_paper", help="添加单篇论文并自动分类")
    add_paper_parser.add_argument("path", type=str, help="PDF 文件的路径")
    add_paper_parser.add_argument(
        "--topics", 
        type=str, 
        default=DEFAULT_TOPICS, 
        help=f"分类主题列表 (逗号分隔)。默认为: {DEFAULT_TOPICS}"
    )

    # ==========================
    # 命令 2: 批量整理论文文件夹
    # ==========================
    organize_parser = subparsers.add_parser("organize", help="一键整理文件夹下的所有论文")
    organize_parser.add_argument("folder", type=str, help="包含乱序 PDF 的文件夹路径")
    organize_parser.add_argument(
        "--topics", 
        type=str, 
        default=DEFAULT_TOPICS, 
        help=f"分类主题列表。默认为: {DEFAULT_TOPICS}" 
    )

    # ==========================
    # 命令 3: 语义搜索论文
    # ==========================
    search_paper_parser = subparsers.add_parser("search_paper", help="语义搜索论文")
    search_paper_parser.add_argument("query", type=str, help="搜索问题或关键词")

    # ==========================
    # 命令 4: 添加图片 (单张或文件夹)
    # ==========================
    add_image_parser = subparsers.add_parser("add_image", help="添加图片到索引")
    add_image_parser.add_argument("path", type=str, help="图片文件路径 或 包含图片的文件夹路径")

    # ==========================
    # 命令 5: 以文搜图
    # ==========================
    search_image_parser = subparsers.add_parser("search_image", help="使用自然语言搜索图片")
    search_image_parser.add_argument("query", type=str, help="图片描述 (例如: 'a cat sleeping on sofa')")

    # 解析参数
    args = parser.parse_args()

    # 如果没有输入命令，打印帮助
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # === 逻辑分发 ===
    
    # 涉及论文的操作，初始化 DocumentManager
    if args.command in ["add_paper", "organize", "search_paper"]:
        # 懒加载：只有在需要时才加载模型，节省启动时间
        dm = DocumentManager()
        
        if args.command == "add_paper":
            dm.add_document(args.path, args.topics)
            
        elif args.command == "organize":
            dm.batch_organize_folder(args.folder, args.topics)
            
        elif args.command == "search_paper":
            results = dm.search(args.query)
            print(f"\n=== '{args.query}' 的搜索结果 ===")
            if not results:
                print("未找到相关论文。")
            for res in results:
                print(f"文件: {res['filename']}")
                print(f"分类: {res['category']}")
                print(f"路径: {res['filepath']}")
                print(f"距离分数: {res['score']:.4f}")
                print("-" * 30)

    # 涉及图片的操作，初始化 ImageManager
    elif args.command in ["add_image", "search_image"]:
        im = ImageManager()
        
        if args.command == "add_image":
            if os.path.isdir(args.path):
                im.add_folder(args.path)
            else:
                im.add_image(args.path)
                
        elif args.command == "search_image":
            results = im.search_image(args.query)
            print(f"\n=== '{args.query}' 的图片搜索结果 ===")
            if not results:
                print("未找到匹配图片。")
            for res in results:
                print(f"文件: {res['filename']}")
                print(f"路径: {res['path']}")
                print(f"距离分数: {res['score']:.4f}")
                print("-" * 30)
                # 提示：在 Windows/Mac 上可以使用 os.startfile(res['path']) 自动打开图片

if __name__ == "__main__":
    main()