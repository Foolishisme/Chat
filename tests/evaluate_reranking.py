"""
RAG检索优化评估脚本
对比优化前（k=3）和优化后（k=20→rerank→5）的效果
"""
import sys
import os
from datetime import datetime
import json
import codecs

# Windows编码修复
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.rag_service import rag_service
import time
import numpy as np

# 测试问题集
TEST_QUESTIONS = [
    "文档的主要内容是什么？",
    "文档中提到了哪些重要信息？",
    "请总结文档的核心观点",
    "文档中引用了哪些经典著作？",
    "作者对当前市场有什么看法？"
]

def calculate_similarity_scores(docs, question):
    """计算文档与问题的相似度"""
    query_embedding = rag_service.text_embeddings.embed_query(question)
    scores = []
    
    for doc in docs:
        doc_text = doc.page_content[:500]  # 取前500字符
        doc_embedding = rag_service.text_embeddings.embed_query(doc_text)
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        scores.append(similarity)
    
    return scores

def test_baseline(question):
    """测试优化前（k=3）"""
    # 禁用重排序
    original_reranking = rag_service.use_reranking
    rag_service.use_reranking = False
    
    start_time = time.time()
    
    # 检索
    retriever = rag_service.text_vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    
    # 获取答案
    result = rag_service.query(question)
    
    elapsed = time.time() - start_time
    
    # 计算相似度
    similarity_scores = calculate_similarity_scores(docs, question)
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
    
    # 恢复设置
    rag_service.use_reranking = original_reranking
    
    return {
        "docs": docs,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "latency": elapsed,
        "similarity_scores": similarity_scores,
        "avg_similarity": avg_similarity,
        "num_docs": len(docs)
    }

def test_reranked(question):
    """测试优化后（k=20→rerank→5）"""
    # 启用重排序
    original_reranking = rag_service.use_reranking
    rag_service.use_reranking = True
    
    start_time = time.time()
    
    # 检索Top-20
    retriever = rag_service.text_vectorstore.as_retriever(search_kwargs={"k": 20})
    candidate_docs = retriever.invoke(question)
    
    # 重排序到Top-5
    docs = rag_service._rerank_documents(question, candidate_docs, top_k=5)
    
    # 获取答案
    result = rag_service.query(question)
    
    elapsed = time.time() - start_time
    
    # 计算相似度
    similarity_scores = calculate_similarity_scores(docs, question)
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
    
    # 恢复设置
    rag_service.use_reranking = original_reranking
    
    return {
        "docs": docs,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "latency": elapsed,
        "similarity_scores": similarity_scores,
        "avg_similarity": avg_similarity,
        "num_docs": len(docs),
        "candidate_num": len(candidate_docs)
    }

def save_results(results, filename):
    """保存结果到文件"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'development')
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(results)
    
    print(f"\n✅ 结果已保存到: {filepath}")

def generate_report(baseline_results, reranked_results):
    """生成评估报告"""
    report = []
    report.append("=" * 80)
    report.append("RAG检索优化评估报告")
    report.append("=" * 80)
    report.append(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"测试问题数: {len(TEST_QUESTIONS)}")
    report.append("\n测试问题:")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        report.append(f"  {i}. {q}")
    
    report.append("\n" + "-" * 80)
    report.append("一、优化前（k=3）结果")
    report.append("-" * 80)
    
    baseline_avg_latency = np.mean([r["latency"] for r in baseline_results])
    baseline_avg_similarity = np.mean([r["avg_similarity"] for r in baseline_results])
    baseline_avg_answer_len = np.mean([len(r["answer"]) for r in baseline_results])
    
    report.append(f"\n平均指标:")
    report.append(f"  检索文档数: 3")
    report.append(f"  平均相似度: {baseline_avg_similarity:.4f}")
    report.append(f"  平均延迟: {baseline_avg_latency:.2f}秒")
    report.append(f"  平均答案长度: {baseline_avg_answer_len:.0f}字符")
    
    report.append("\n详细结果:")
    for i, (question, result) in enumerate(zip(TEST_QUESTIONS, baseline_results), 1):
        report.append(f"\n  问题 {i}: {question}")
        report.append(f"    检索文档数: {result['num_docs']}")
        report.append(f"    平均相似度: {result['avg_similarity']:.4f}")
        report.append(f"    延迟: {result['latency']:.2f}秒")
        report.append(f"    答案长度: {len(result['answer'])}字符")
        report.append(f"    来源页数: {[s.get('page', '?') for s in result['sources']]}")
    
    report.append("\n" + "-" * 80)
    report.append("二、优化后（k=20→rerank→5）结果")
    report.append("-" * 80)
    
    reranked_avg_latency = np.mean([r["latency"] for r in reranked_results])
    reranked_avg_similarity = np.mean([r["avg_similarity"] for r in reranked_results])
    reranked_avg_answer_len = np.mean([len(r["answer"]) for r in reranked_results])
    
    report.append(f"\n平均指标:")
    report.append(f"  候选文档数: {reranked_results[0].get('candidate_num', 20)}")
    report.append(f"  最终文档数: 5")
    report.append(f"  平均相似度: {reranked_avg_similarity:.4f}")
    report.append(f"  平均延迟: {reranked_avg_latency:.2f}秒")
    report.append(f"  平均答案长度: {reranked_avg_answer_len:.0f}字符")
    
    report.append("\n详细结果:")
    for i, (question, result) in enumerate(zip(TEST_QUESTIONS, reranked_results), 1):
        report.append(f"\n  问题 {i}: {question}")
        report.append(f"    候选文档数: {result.get('candidate_num', 20)}")
        report.append(f"    最终文档数: {result['num_docs']}")
        report.append(f"    平均相似度: {result['avg_similarity']:.4f}")
        report.append(f"    延迟: {result['latency']:.2f}秒")
        report.append(f"    答案长度: {len(result['answer'])}字符")
        report.append(f"    来源页数: {[s.get('page', '?') for s in result['sources']]}")
    
    report.append("\n" + "-" * 80)
    report.append("三、对比分析")
    report.append("-" * 80)
    
    similarity_improvement = ((reranked_avg_similarity - baseline_avg_similarity) / baseline_avg_similarity * 100) if baseline_avg_similarity > 0 else 0
    latency_increase = ((reranked_avg_latency - baseline_avg_latency) / baseline_avg_latency * 100) if baseline_avg_latency > 0 else 0
    answer_len_change = ((reranked_avg_answer_len - baseline_avg_answer_len) / baseline_avg_answer_len * 100) if baseline_avg_answer_len > 0 else 0
    
    report.append(f"\n指标对比:")
    report.append(f"  平均相似度: {baseline_avg_similarity:.4f} → {reranked_avg_similarity:.4f} ({similarity_improvement:+.2f}%)")
    report.append(f"  平均延迟: {baseline_avg_latency:.2f}秒 → {reranked_avg_latency:.2f}秒 ({latency_increase:+.2f}%)")
    report.append(f"  平均答案长度: {baseline_avg_answer_len:.0f} → {reranked_avg_answer_len:.0f} ({answer_len_change:+.2f}%)")
    
    report.append("\n" + "-" * 80)
    report.append("四、结论")
    report.append("-" * 80)
    
    if similarity_improvement > 0:
        report.append(f"\n✅ 优化效果:")
        report.append(f"  - 检索质量提升: 相似度提升 {similarity_improvement:.2f}%")
        report.append(f"  - 检索范围扩大: 从Top-3扩展到Top-20候选，最终Top-5")
        if latency_increase < 50:
            report.append(f"  - 性能影响: 延迟增加 {latency_increase:.2f}%（可接受）")
        else:
            report.append(f"  - 性能影响: 延迟增加 {latency_increase:.2f}%（需要优化）")
    else:
        report.append(f"\n⚠️  优化效果不明显，可能需要调整参数")
    
    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """主函数"""
    print("=" * 80)
    print("RAG检索优化评估")
    print("=" * 80)
    
    # 初始化RAG服务
    print("\n初始化RAG服务...")
    rag_service.initialize()
    
    baseline_results = []
    reranked_results = []
    
    print("\n开始测试...")
    print("-" * 80)
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n测试问题 {i}/{len(TEST_QUESTIONS)}: {question}")
        
        # 测试优化前
        print("  测试优化前（k=3）...")
        baseline = test_baseline(question)
        baseline_results.append(baseline)
        print(f"    检索文档数: {baseline['num_docs']}")
        print(f"    平均相似度: {baseline['avg_similarity']:.4f}")
        print(f"    延迟: {baseline['latency']:.2f}秒")
        
        # 测试优化后
        print("  测试优化后（k=20→rerank→5）...")
        reranked = test_reranked(question)
        reranked_results.append(reranked)
        print(f"    候选文档数: {reranked.get('candidate_num', 20)}")
        print(f"    最终文档数: {reranked['num_docs']}")
        print(f"    平均相似度: {reranked['avg_similarity']:.4f}")
        print(f"    延迟: {reranked['latency']:.2f}秒")
    
    # 生成报告
    print("\n" + "=" * 80)
    print("生成评估报告...")
    report = generate_report(baseline_results, reranked_results)
    
    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"RAG检索优化评估报告_{timestamp}.txt"
    save_results(report, filename)
    
    # 打印报告
    print("\n" + report)

if __name__ == "__main__":
    main()

