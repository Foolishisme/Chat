"""
RAGAS评估脚本
使用RAGAS进行更全面的RAG质量评估
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

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevance
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS未安装，请运行: pip install ragas datasets")

from app.services.rag_service import rag_service
import time

# 测试问题集（需要ground_truth）
TEST_QUESTIONS = [
    {
        "question": "文档的主要内容是什么？",
        "ground_truth": ""  # 可以手动填写或留空
    },
    {
        "question": "文档中提到了哪些重要信息？",
        "ground_truth": ""
    },
    {
        "question": "请总结文档的核心观点",
        "ground_truth": ""
    }
]

def test_with_reranking(question, use_reranking=True):
    """测试检索（支持切换重排序）"""
    original_reranking = rag_service.use_reranking
    rag_service.use_reranking = use_reranking
    
    # 检索
    if use_reranking:
        retriever = rag_service.text_vectorstore.as_retriever(search_kwargs={"k": 20})
        candidate_docs = retriever.invoke(question)
        docs = rag_service._rerank_documents(question, candidate_docs, top_k=5)
    else:
        retriever = rag_service.text_vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
    
    # 获取答案
    result = rag_service.query(question)
    
    # 恢复设置
    rag_service.use_reranking = original_reranking
    
    return {
        "question": question,
        "answer": result["answer"],
        "contexts": [doc.page_content for doc in docs],
        "ground_truth": ""
    }

def run_ragas_evaluation(results, name):
    """使用RAGAS评估结果"""
    if not RAGAS_AVAILABLE:
        print(f"\n⚠️  跳过RAGAS评估: {name} (RAGAS未安装)")
        return None
    
    print(f"\n{'=' * 80}")
    print(f"RAGAS评估: {name}")
    print("=" * 80)
    
    # 准备数据
    dataset_dict = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r.get("ground_truth", "") for r in results]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # 运行评估
    try:
        print("运行RAGAS评估...")
        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevance
            ]
        )
        
        print("\n评估结果:")
        print(result)
        
        # 提取指标
        metrics = {
            "context_precision": float(result["context_precision"]) if "context_precision" in result else 0,
            "context_recall": float(result["context_recall"]) if "context_recall" in result else 0,
            "faithfulness": float(result["faithfulness"]) if "faithfulness" in result else 0,
            "answer_relevance": float(result["answer_relevance"]) if "answer_relevance" in result else 0
        }
        
        return metrics
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results(results, filename):
    """保存结果到文件"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'development')
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(results)
    
    print(f"\n✅ 结果已保存到: {filepath}")

def generate_ragas_report(baseline_metrics, reranked_metrics):
    """生成RAGAS评估报告"""
    report = []
    report.append("=" * 80)
    report.append("RAGAS评估报告")
    report.append("=" * 80)
    report.append(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"测试问题数: {len(TEST_QUESTIONS)}")
    
    if not baseline_metrics or not reranked_metrics:
        report.append("\n⚠️  评估数据不完整，无法生成完整报告")
        return "\n".join(report)
    
    report.append("\n" + "-" * 80)
    report.append("一、RAGAS指标对比")
    report.append("-" * 80)
    
    report.append(f"\n{'指标':<25} {'优化前':<15} {'优化后':<15} {'提升':<15}")
    report.append("-" * 70)
    
    for metric_name in ["context_precision", "context_recall", "faithfulness", "answer_relevance"]:
        baseline_score = baseline_metrics.get(metric_name, 0)
        reranked_score = reranked_metrics.get(metric_name, 0)
        improvement = reranked_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
        
        metric_display = {
            "context_precision": "上下文精确度",
            "context_recall": "上下文召回率",
            "faithfulness": "忠实度",
            "answer_relevance": "答案相关性"
        }.get(metric_name, metric_name)
        
        report.append(f"{metric_display:<25} {baseline_score:<15.4f} {reranked_score:<15.4f} {improvement_pct:>+10.2f}%")
    
    report.append("\n" + "-" * 80)
    report.append("二、指标说明")
    report.append("-" * 80)
    report.append("\n- Context Precision (上下文精确度): 检索到的文档是否相关")
    report.append("- Context Recall (上下文召回率): 是否检索到所有相关文档")
    report.append("- Faithfulness (忠实度): 答案是否基于检索到的文档")
    report.append("- Answer Relevance (答案相关性): 答案是否回答了问题")
    
    report.append("\n" + "-" * 80)
    report.append("三、结论")
    report.append("-" * 80)
    
    avg_improvement = sum([
        ((reranked_metrics.get(m, 0) - baseline_metrics.get(m, 0)) / baseline_metrics.get(m, 1) * 100) 
        if baseline_metrics.get(m, 0) > 0 else 0
        for m in ["context_precision", "context_recall", "faithfulness", "answer_relevance"]
    ]) / 4
    
    if avg_improvement > 0:
        report.append(f"\n✅ 优化效果:")
        report.append(f"  - 平均指标提升: {avg_improvement:.2f}%")
        report.append(f"  - 检索质量显著提升")
    else:
        report.append(f"\n⚠️  优化效果不明显，可能需要调整参数")
    
    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """主函数"""
    print("=" * 80)
    print("RAGAS评估")
    print("=" * 80)
    
    if not RAGAS_AVAILABLE:
        print("\n❌ RAGAS未安装")
        print("请运行: pip install ragas datasets")
        return
    
    # 初始化RAG服务
    print("\n初始化RAG服务...")
    rag_service.initialize()
    
    # 评估优化前
    print("\n评估优化前（k=3）...")
    baseline_results = []
    for test in TEST_QUESTIONS:
        result = test_with_reranking(test["question"], use_reranking=False)
        result["ground_truth"] = test.get("ground_truth", "")
        baseline_results.append(result)
    
    baseline_metrics = run_ragas_evaluation(baseline_results, "优化前（k=3）")
    
    # 评估优化后
    print("\n评估优化后（k=20→rerank→5）...")
    reranked_results = []
    for test in TEST_QUESTIONS:
        result = test_with_reranking(test["question"], use_reranking=True)
        result["ground_truth"] = test.get("ground_truth", "")
        reranked_results.append(result)
    
    reranked_metrics = run_ragas_evaluation(reranked_results, "优化后（k=20→rerank→5）")
    
    # 生成报告
    if baseline_metrics and reranked_metrics:
        print("\n" + "=" * 80)
        print("生成RAGAS评估报告...")
        report = generate_ragas_report(baseline_metrics, reranked_metrics)
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"RAGAS评估报告_{timestamp}.txt"
        save_results(report, filename)
        
        # 打印报告
        print("\n" + report)
    else:
        print("\n⚠️  无法生成完整报告，请检查评估结果")

if __name__ == "__main__":
    main()

