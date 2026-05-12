#!/usr/bin/env python3
"""
QA Quality Analysis for Multi-Chunk Combinations
Analyzes the quality of QA pairs that require information from multiple chunks
"""

import json
import re
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

def load_qa_data(filepath):
    """Load QA data from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def analyze_chunk_dependency(qa_pair, chunks):
    """
    Analyze if a QA pair truly requires multiple chunks
    Returns: dependency_score, chunk_usage_analysis
    """
    question = qa_pair['question'].lower()
    answer = qa_pair['answer'].lower()
    chunks_used = qa_pair.get('chunks_used', [])
    
    # Extract key terms from each chunk
    chunk_key_terms = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk['chunk_text'].lower()
        # Extract key technical terms, numbers, and specific concepts
        terms = re.findall(r'\b(?:[a-z]+(?:\s+[a-z]+)*\s+\d+(?:\.\d+)?(?:mg|l-1|%|°c)?|[a-z]+(?:\s+[a-z]+)*)\b', chunk_text)
        chunk_key_terms.append(set(terms))
    
    # Check if answer contains terms from multiple chunks
    multi_chunk_terms = 0
    chunk_usage = defaultdict(int)
    
    for i, terms in enumerate(chunk_key_terms):
        for term in terms:
            if term in answer and len(term) > 3:  # Avoid very short terms
                chunk_usage[i] += 1
                if i in chunks_used:
                    multi_chunk_terms += 1
    
    # Calculate dependency score
    if len(chunks_used) > 1:
        dependency_score = min(1.0, multi_chunk_terms / 10)  # Normalize
    else:
        dependency_score = 0.0
    
    return dependency_score, chunk_usage

def analyze_synthesis_quality(qa_pair, chunks):
    """
    Analyze the synthesis quality of the answer
    Returns: synthesis_score, analysis
    """
    answer = qa_pair['answer']
    chunks_used = qa_pair.get('chunks_used', [])
    
    # Check for synthesis indicators
    synthesis_indicators = [
        'while', 'however', 'although', 'but', 'on the other hand',
        'in addition', 'furthermore', 'moreover', 'also', 'additionally',
        'this means', 'therefore', 'thus', 'as a result', 'consequently',
        'combining', 'together', 'both', 'and', 'or', 'if', 'when'
    ]
    
    synthesis_count = sum(1 for indicator in synthesis_indicators if indicator in answer.lower())
    
    # Check for specific information from chunks
    chunk_specific_info = 0
    for chunk in chunks:
        # Look for specific numbers, measurements, technical terms
        specific_terms = re.findall(r'\d+(?:\.\d+)?(?:mg|l-1|%|°c)?', chunk['chunk_text'])
        for term in specific_terms:
            if term in answer:
                chunk_specific_info += 1
    
    # Calculate synthesis score
    synthesis_score = min(1.0, (synthesis_count + chunk_specific_info) / 20)
    
    analysis = {
        'synthesis_indicators': synthesis_count,
        'chunk_specific_info': chunk_specific_info,
        'answer_length': len(answer.split())
    }
    
    return synthesis_score, analysis

def analyze_farmer_appropriateness(qa_pair):
    """
    Analyze if the QA pair is appropriate for the farmer persona
    """
    question = qa_pair['question'].lower()
    answer = qa_pair['answer'].lower()
    
    # Check for farmer-appropriate language
    farmer_indicators = [
        'my', 'i', 'me', 'my field', 'my crops', 'my plants',
        'how to', 'what should', 'can i', 'will it', 'does it',
        'simple', 'easy', 'practical', 'useful', 'help'
    ]
    
    farmer_score = sum(1 for indicator in farmer_indicators if indicator in question)
    
    # Check for technical jargon (should be minimal)
    technical_jargon = [
        'agrobacterium', 'transformation', 'vector', 'plasmid', 'genome',
        'pcr', 'electrophoresis', 'centrifugation', 'cocultivation'
    ]
    
    jargon_count = sum(1 for jargon in technical_jargon if jargon in answer)
    jargon_penalty = max(0, jargon_count * 0.2)
    
    # Check question length (should be 10-25 words)
    question_words = len(question.split())
    length_score = 1.0 if 10 <= question_words <= 25 else 0.5
    
    # Check answer length (should be 50-150 words)
    answer_words = len(answer.split())
    answer_length_score = 1.0 if 50 <= answer_words <= 150 else 0.7
    
    appropriateness_score = (farmer_score / 5 + length_score + answer_length_score - jargon_penalty) / 3
    
    return appropriateness_score, {
        'farmer_indicators': farmer_score,
        'jargon_count': jargon_count,
        'question_length': question_words,
        'answer_length': answer_words
    }

def analyze_agricultural_relevance(qa_pair):
    """
    Analyze agricultural relevance of the QA pair
    """
    question = qa_pair['question'].lower()
    answer = qa_pair['answer'].lower()
    
    agricultural_terms = [
        'crop', 'plant', 'seed', 'soil', 'fertilizer', 'pest', 'disease',
        'harvest', 'yield', 'growth', 'watering', 'temperature', 'light',
        'potato', 'farm', 'field', 'agriculture', 'farming', 'farmer'
    ]
    
    relevance_score = sum(1 for term in agricultural_terms if term in question or term in answer)
    relevance_score = min(1.0, relevance_score / 5)
    
    return relevance_score

def comprehensive_qa_analysis(data):
    """
    Comprehensive analysis of QA pairs
    """
    results = {
        'total_combinations': len(data['results']),
        'total_qa_pairs': 0,
        'multi_chunk_qa_pairs': 0,
        'high_quality_pairs': 0,
        'analysis_details': [],
        'summary_stats': {}
    }
    
    all_scores = {
        'dependency_scores': [],
        'synthesis_scores': [],
        'appropriateness_scores': [],
        'relevance_scores': []
    }
    
    for i, combination in enumerate(data['results']):
        chunks = combination['chunks']
        qa_pairs = combination.get('qa_pairs', [])
        
        results['total_qa_pairs'] += len(qa_pairs)
        
        for j, qa_pair in enumerate(qa_pairs):
            # Check if it uses multiple chunks
            chunks_used = qa_pair.get('chunks_used', [])
            is_multi_chunk = len(chunks_used) > 1
            
            if is_multi_chunk:
                results['multi_chunk_qa_pairs'] += 1
                
                # Analyze quality metrics
                dependency_score, chunk_usage = analyze_chunk_dependency(qa_pair, chunks)
                synthesis_score, synthesis_analysis = analyze_synthesis_quality(qa_pair, chunks)
                appropriateness_score, appropriateness_analysis = analyze_farmer_appropriateness(qa_pair)
                relevance_score = analyze_agricultural_relevance(qa_pair)
                
                # Calculate overall quality score
                overall_score = (dependency_score + synthesis_score + appropriateness_score + relevance_score) / 4
                
                # Store scores
                all_scores['dependency_scores'].append(dependency_score)
                all_scores['synthesis_scores'].append(synthesis_score)
                all_scores['appropriateness_scores'].append(appropriateness_score)
                all_scores['relevance_scores'].append(relevance_score)
                
                # High quality threshold
                if overall_score >= 0.7:
                    results['high_quality_pairs'] += 1
                
                # Store detailed analysis
                analysis_detail = {
                    'combination_id': i,
                    'qa_pair_id': j,
                    'question': qa_pair['question'],
                    'answer': qa_pair['answer'],
                    'chunks_used': chunks_used,
                    'scores': {
                        'dependency': dependency_score,
                        'synthesis': synthesis_score,
                        'appropriateness': appropriateness_score,
                        'relevance': relevance_score,
                        'overall': overall_score
                    },
                    'analysis': {
                        'chunk_usage': dict(chunk_usage),
                        'synthesis': synthesis_analysis,
                        'appropriateness': appropriateness_analysis
                    }
                }
                results['analysis_details'].append(analysis_detail)
    
    # Calculate summary statistics
    if all_scores['dependency_scores']:
        results['summary_stats'] = {
            'avg_dependency_score': sum(all_scores['dependency_scores']) / len(all_scores['dependency_scores']),
            'avg_synthesis_score': sum(all_scores['synthesis_scores']) / len(all_scores['synthesis_scores']),
            'avg_appropriateness_score': sum(all_scores['appropriateness_scores']) / len(all_scores['appropriateness_scores']),
            'avg_relevance_score': sum(all_scores['relevance_scores']) / len(all_scores['relevance_scores']),
            'high_quality_percentage': (results['high_quality_pairs'] / results['multi_chunk_qa_pairs']) * 100 if results['multi_chunk_qa_pairs'] > 0 else 0
        }
    
    return results

def generate_quality_report(analysis_results):
    """
    Generate a comprehensive quality report
    """
    report = f"""
QA QUALITY ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
========
Total Combinations: {analysis_results['total_combinations']}
Total QA Pairs: {analysis_results['total_qa_pairs']}
Multi-Chunk QA Pairs: {analysis_results['multi_chunk_qa_pairs']}
High Quality Pairs (Score ≥ 0.7): {analysis_results['high_quality_pairs']}

QUALITY METRICS
===============
"""
    
    if analysis_results['summary_stats']:
        stats = analysis_results['summary_stats']
        report += f"""
Average Dependency Score: {stats['avg_dependency_score']:.3f}
Average Synthesis Score: {stats['avg_synthesis_score']:.3f}
Average Appropriateness Score: {stats['avg_appropriateness_score']:.3f}
Average Relevance Score: {stats['avg_relevance_score']:.3f}
High Quality Percentage: {stats['high_quality_percentage']:.1f}%

DETAILED ANALYSIS
=================
"""
    
    # Show top performing QA pairs
    high_quality_pairs = [detail for detail in analysis_results['analysis_details'] 
                         if detail['scores']['overall'] >= 0.8]
    
    if high_quality_pairs:
        report += f"\nTOP PERFORMING QA PAIRS (Score ≥ 0.8):\n"
        for i, pair in enumerate(high_quality_pairs[:5]):  # Show top 5
            report += f"""
{i+1}. Question: {pair['question']}
   Answer: {pair['answer'][:200]}...
   Scores: Dependency={pair['scores']['dependency']:.3f}, 
           Synthesis={pair['scores']['synthesis']:.3f}, 
           Appropriateness={pair['scores']['appropriateness']:.3f}, 
           Relevance={pair['scores']['relevance']:.3f}
   Overall: {pair['scores']['overall']:.3f}
"""
    
    # Show areas for improvement
    low_quality_pairs = [detail for detail in analysis_results['analysis_details'] 
                        if detail['scores']['overall'] < 0.5]
    
    if low_quality_pairs:
        report += f"\nAREAS FOR IMPROVEMENT (Score < 0.5):\n"
        for i, pair in enumerate(low_quality_pairs[:3]):  # Show top 3
            report += f"""
{i+1}. Question: {pair['question']}
   Answer: {pair['answer'][:200]}...
   Scores: Dependency={pair['scores']['dependency']:.3f}, 
           Synthesis={pair['scores']['synthesis']:.3f}, 
           Appropriateness={pair['scores']['appropriateness']:.3f}, 
           Relevance={pair['scores']['relevance']:.3f}
   Overall: {pair['scores']['overall']:.3f}
"""
    
    return report

def main():
    """Main analysis function"""
    # Load the QA data
    data = load_qa_data('search_results/Combination_2_results_with_qa.json')
    
    if not data:
        print("Failed to load QA data")
        return
    
    print("Starting QA Quality Analysis...")
    
    # Perform comprehensive analysis
    analysis_results = comprehensive_qa_analysis(data)
    
    # Generate report
    report = generate_quality_report(analysis_results)
    
    # Save detailed results
    with open('qa_quality_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    # Save report
    with open('qa_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Print summary
    print(report)
    
    # Create DataFrame for further analysis
    df_data = []
    for detail in analysis_results['analysis_details']:
        df_data.append({
            'combination_id': detail['combination_id'],
            'qa_pair_id': detail['qa_pair_id'],
            'dependency_score': detail['scores']['dependency'],
            'synthesis_score': detail['scores']['synthesis'],
            'appropriateness_score': detail['scores']['appropriateness'],
            'relevance_score': detail['scores']['relevance'],
            'overall_score': detail['scores']['overall'],
            'question_length': detail['analysis']['appropriateness']['question_length'],
            'answer_length': detail['analysis']['appropriateness']['answer_length']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv('qa_quality_scores.csv', index=False)
    
    print(f"\nAnalysis complete! Results saved to:")
    print("- qa_quality_analysis_results.json (detailed results)")
    print("- qa_quality_report.txt (human-readable report)")
    print("- qa_quality_scores.csv (scores for further analysis)")

if __name__ == "__main__":
    main() 