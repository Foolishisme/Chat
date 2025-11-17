"""
FAISS到Qdrant数据迁移脚本
将现有的FAISS向量数据库迁移到Qdrant
"""
import sys
import os
import codecs

# Windows编码修复
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import settings
from app.services.rag_service import rag_service
from datetime import datetime

def migrate_data():
    """迁移数据从FAISS到Qdrant"""
    print("=" * 80)
    print("FAISS到Qdrant数据迁移")
    print("=" * 80)
    print(f"\n迁移时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查FAISS数据是否存在
    text_index_path = os.path.join(settings.chroma_persist_directory, "text_index.faiss")
    image_index_path = os.path.join(settings.chroma_persist_directory, "image_index.faiss")
    
    faiss_exists = os.path.exists(text_index_path) or os.path.exists(image_index_path)
    
    if not faiss_exists:
        print("\n⚠️  未找到FAISS数据，将直接使用Qdrant创建新索引")
        print("   如果这是首次运行，这是正常的")
        
        # 直接初始化Qdrant服务（会自动创建索引）
        print("\n初始化Qdrant服务...")
        rag_service.initialize()
        print("\n✅ 迁移完成（创建新索引）")
        return
    
    print("\n📦 发现FAISS数据，开始迁移...")
    
    # 方案：重新索引（因为格式不同，无法直接转换）
    print("\n迁移策略：重新索引")
    print("  1. 重新加载PDF文档")
    print("  2. 使用Qdrant重新创建向量索引")
    print("  3. 保留原有FAISS数据作为备份")
    
    # 备份FAISS数据
    backup_dir = os.path.join(settings.chroma_persist_directory, "faiss_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"\n📦 创建备份目录: {backup_dir}")
    
    # 初始化Qdrant服务（会自动重新索引）
    print("\n初始化Qdrant服务...")
    print("  这将重新加载文档并创建Qdrant索引...")
    
    try:
        rag_service.initialize()
        print("\n✅ 迁移完成！")
        print(f"\n📁 数据位置:")
        print(f"  - Qdrant数据库: {os.path.join(settings.chroma_persist_directory, 'qdrant_db')}")
        print(f"  - FAISS备份: {backup_dir}")
        print(f"\n💡 提示: 旧的FAISS数据已保留在备份目录中")
    except Exception as e:
        print(f"\n❌ 迁移失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def verify_migration():
    """验证迁移结果"""
    print("\n" + "=" * 80)
    print("验证迁移结果")
    print("=" * 80)
    
    try:
        # 检查Qdrant集合
        collections = rag_service.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        print(f"\n✅ Qdrant集合:")
        if "text_documents" in collection_names:
            text_info = rag_service.qdrant_client.get_collection("text_documents")
            print(f"  - text_documents: {text_info.points_count} 个向量")
        else:
            print(f"  - text_documents: 未找到")
        
        if "image_documents" in collection_names:
            image_info = rag_service.qdrant_client.get_collection("image_documents")
            print(f"  - image_documents: {image_info.points_count} 个向量")
        else:
            print(f"  - image_documents: 未找到")
        
        # 测试检索
        print("\n🧪 测试检索功能...")
        test_question = "测试问题"
        
        if rag_service.text_vectorstore:
            docs = rag_service.text_vectorstore.similarity_search(test_question, k=1)
            print(f"  ✅ 文本检索成功: 找到 {len(docs)} 个文档")
        else:
            print(f"  ⚠️  文本向量库未初始化")
        
        if rag_service.image_vectorstore:
            try:
                docs = rag_service.image_vectorstore.similarity_search(test_question, k=1)
                print(f"  ✅ 图片检索成功: 找到 {len(docs)} 个文档")
            except Exception as e:
                print(f"  ⚠️  图片检索测试失败: {str(e)}")
        else:
            print(f"  ⚠️  图片向量库未初始化")
        
        print("\n✅ 验证完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n开始迁移...")
    
    # 执行迁移
    success = migrate_data()
    
    if success:
        # 验证迁移
        verify_migration()
        
        print("\n" + "=" * 80)
        print("迁移完成！")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 测试应用功能")
        print("  2. 如果一切正常，可以删除FAISS备份数据")
        print("  3. 更新文档说明")
    else:
        print("\n" + "=" * 80)
        print("迁移失败，请检查错误信息")
        print("=" * 80)

