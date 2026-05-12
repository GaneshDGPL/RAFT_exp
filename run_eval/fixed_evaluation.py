import pandas as pd
import json
import numpy as np
import re
import openai
from typing import List, Dict, Any, Tuple
import ast
import os
from collections import defaultdict


class EnhancedFactEvaluator:
    def __init__(self, openai_api_key: str = None):
        # Modified to use environment variable or passed key
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.token_tracker = TokenTracker()
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        else:
            print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")

    def check_valid_json(self, predicted_facts: Any) -> Dict[str, Any]:
        """Check if the response is valid JSON or structured data"""
        try:
            # Handle different input formats
            if isinstance(predicted_facts, str):
                # Try to parse as JSON string
                try:
                    parsed = json.loads(predicted_facts)
                except json.JSONDecodeError:
                    # Try to parse as Python literal
                    try:
                        parsed = ast.literal_eval(predicted_facts)
                    except (ValueError, SyntaxError):
                        return {
                            'is_valid': False,
                            'parsed_data': {"facts": []},
                            'error': f'Failed to parse string: {str(predicted_facts)[:100]}...'
                        }
            elif isinstance(predicted_facts, list):
                # Already a list - use directly
                parsed = predicted_facts
            elif isinstance(predicted_facts, dict):
                # Already a dict - extract facts if available
                parsed = predicted_facts.get('facts', predicted_facts)
            else:
                return {
                    'is_valid': False,
                    'parsed_data': {"facts": []},
                    'error': f'Unsupported data type: {type(predicted_facts)}'
                }

            # Ensure we have a list of facts
            if isinstance(parsed, list):
                facts_list = parsed
            elif isinstance(parsed, dict) and 'facts' in parsed:
                facts_list = parsed['facts']
            else:
                facts_list = [parsed] if parsed else []

            return {
                'is_valid': True,
                'parsed_data': {"facts": facts_list},
                'error': None
            }
        except Exception as e:
            return {
                'is_valid': False,
                'parsed_data': {"facts": []},
                'error': str(e)
            }

    def extract_facts_by_category(self, facts_data: Any) -> Dict[str, List[Dict]]:
        """Extract facts grouped by category"""
        category_facts = defaultdict(list)
        
        if isinstance(facts_data, str):
            try:
                facts_data = json.loads(facts_data)
            except:
                try:
                    facts_data = ast.literal_eval(facts_data)
                except:
                    return category_facts
        
        if isinstance(facts_data, list):
            facts_list = facts_data
        elif isinstance(facts_data, dict) and 'facts' in facts_data:
            facts_list = facts_data.get('facts', [])
        else:
            return category_facts
        
        for fact in facts_list:
            if isinstance(fact, dict):
                category = fact.get('category', 'unknown')
                category_facts[category].append(fact)
            elif isinstance(fact, str):
                # If it's just a string, assign to 'unknown' category
                category_facts['unknown'].append({'fact': fact, 'category': 'unknown'})
        
        return dict(category_facts)

    def iterative_fact_matching(self, gold_facts: List[str], pred_facts: List[str],
                                        category: str, debug: bool = False) -> List[Dict]:
        """
        Optimized iterative matching: Always iterate through the smaller list
        Returns list of match pairs with reasoning
        """
        if not gold_facts or not pred_facts:
            return []
        
        # Determine which list is smaller and set up iteration strategy
        gold_count = len(gold_facts)
        pred_count = len(pred_facts)
        
        if gold_count <= pred_count:
            # Iterate through gold facts (smaller or equal)
            smaller_list = gold_facts
            larger_list = pred_facts.copy()
            iterate_gold = True
            num_iterations = gold_count
        else:
            # Iterate through predicted facts (smaller)
            smaller_list = pred_facts
            larger_list = gold_facts.copy()
            iterate_gold = False
            num_iterations = pred_count
        
        if debug:
            print(f"\nOptimized matching strategy:")
            print(f"Gold facts: {gold_count}, Pred facts: {pred_count}")
            print(f"Iterating through: {'gold facts' if iterate_gold else 'predicted facts'}")
            print(f"Number of iterations: {num_iterations}")
        
        matched_pairs = []
        
        for i, fact_to_match in enumerate(smaller_list):
            if not larger_list:
                break
                
            if debug:
                fact_type = "gold" if iterate_gold else "predicted"
                print(f"\nMatching {fact_type} fact {i+1}: {fact_to_match[:100]}...")
                print(f"Against {len(larger_list)} remaining facts")
            
            # Find best match using LLM
            match_result = self.find_best_semantic_match(
                fact_to_match,
                larger_list,
                category,
                debug=debug
            )
            
            if match_result and match_result['best_match']:
                if iterate_gold:
                    # We iterated through gold, matched against pred
                    matched_pairs.append({
                        'gold': fact_to_match,
                        'pred': match_result['best_match'],
                        'reason': match_result['reason'],
                        'confidence': match_result.get('confidence', 0.0),
                        'contradictions': match_result.get('contradictions', [])
                    })
                else:
                    # We iterated through pred, matched against gold
                    matched_pairs.append({
                        'pred': fact_to_match,
                        'gold': match_result['best_match'],
                        'reason': match_result['reason'],
                        'confidence': match_result.get('confidence', 0.0),
                        'contradictions': match_result.get('contradictions', [])
                    })
                
                larger_list.remove(match_result['best_match'])
                if debug:
                    print(f"Found match: {match_result['best_match'][:100]}...")
                    print(f"Reason: {match_result['reason']}")
            elif debug:
                print("No match found")
                if match_result:
                    print(f"Reason: {match_result['reason']}")
        
        return matched_pairs

    # Updated category_wise_fact_matching method
    def category_wise_fact_matching(self, predicted_facts: Any, golden_facts: Any, debug: bool = False) -> Dict[str, Any]:
        """
        Enhanced category-wise fact matching with optimized iteration strategy and contradiction tracking
        """
        
        # Extract facts by category
        pred_by_category = self.extract_facts_by_category(predicted_facts)
        gold_by_category = self.extract_facts_by_category(golden_facts)
        
        if debug:
            print(f"Predicted categories: {list(pred_by_category.keys())}")
            print(f"Golden categories: {list(gold_by_category.keys())}")
        
        total_matches = 0
        total_gold_facts = 0
        total_contradictions = 0
        category_results = {}
        all_contradictions = []  # Store all contradictions separately
        
        # Only consider categories where ground truth has facts
        for category in gold_by_category.keys():
            gold_facts_in_category = gold_by_category[category]
            pred_facts_in_category = pred_by_category.get(category, [])
            
            total_gold_facts += len(gold_facts_in_category)
            
            if debug:
                print(f"\nCategory: {category}")
                print(f"Gold facts: {len(gold_facts_in_category)}")
                print(f"Pred facts: {len(pred_facts_in_category)}")
            
            if not pred_facts_in_category:
                # No predicted facts in this category
                category_results[category] = {
                    'matches': 0,
                    'gold_count': len(gold_facts_in_category),
                    'pred_count': 0,
                    'matched_facts': [],
                    'contradictions': []
                }
                continue
            
            # Extract fact texts for matching
            gold_fact_texts = [self._extract_fact_text(fact) for fact in gold_facts_in_category]
            pred_fact_texts = [self._extract_fact_text(fact) for fact in pred_facts_in_category]
            
            # Use optimized iterative matching
            matched_facts = self.iterative_fact_matching(
                gold_fact_texts,
                pred_fact_texts,
                category,
                debug=debug
            )
            
            matches_in_category = len(matched_facts)
            total_matches += matches_in_category
            
            # Collect contradictions from matched facts
            category_contradictions = []
            for match in matched_facts:
                if match.get('contradictions'):
                    for contradiction in match['contradictions']:
                        contradiction_entry = {
                            'category': category,
                            'reference_fact': match.get('gold', match.get('pred')),
                            'contradicting_fact': contradiction['contradicting_fact'],
                            'contradiction_reason': contradiction['contradiction_reason']
                        }
                        category_contradictions.append(contradiction_entry)
                        all_contradictions.append(contradiction_entry)
            
            total_contradictions += len(category_contradictions)
            
            category_results[category] = {
                'matches': matches_in_category,
                'gold_count': len(gold_facts_in_category),
                'pred_count': len(pred_facts_in_category),
                'matched_facts': matched_facts,
                'contradictions': category_contradictions,
                'iteration_strategy': 'gold' if len(gold_fact_texts) <= len(pred_fact_texts) else 'pred'
            }
            
            if debug:
                print(f"Matches in {category}: {matches_in_category}")
                print(f"Contradictions in {category}: {len(category_contradictions)}")
                print(f"Strategy used: iterate through {'gold' if len(gold_fact_texts) <= len(pred_fact_texts) else 'predicted'} facts")
        
        # Calculate overall IoGT score
        iogt_score = total_matches / total_gold_facts if total_gold_facts > 0 else 0
        
        if debug:
            print(f"\nTotal matches: {total_matches}")
            print(f"Total gold facts: {total_gold_facts}")
            print(f"Total contradictions: {total_contradictions}")
            print(f"IoGT Score: {iogt_score}")
        
        return {
            'iogt_score': iogt_score,
            'total_matches': total_matches,
            'total_gold_facts': total_gold_facts,
            'total_contradictions': total_contradictions,
            'contradictions': all_contradictions,  # Separate column for all contradictions
            'category_results': category_results,
            'semantic_similarity': iogt_score
        }

    def find_best_semantic_match(self, gold_fact: str, pred_facts: List[str],
                                category: str, debug: bool = False) -> Dict:
        """Find the best semantic match for a gold fact from predicted facts using LLM with single unified prompt"""
        
        if not pred_facts:
            return {'best_match': None, 'reason': 'No predicted facts available', 'contradictions': []}
        
        # Single unified prompt for both matching and contradiction detection
        prompt = f"""You are an agricultural fact comparison expert. Compare the reference fact with the candidate facts and find the best semantic match AND identify any contradictions.

REFERENCE FACT (Category: {category}):
{gold_fact}

CANDIDATE FACTS:
{json.dumps(pred_facts, indent=2)}

INSTRUCTIONS:
1. Find the candidate fact that conveys the most similar agricultural meaning to the reference fact
2. Consider facts as matching if they convey similar agricultural concepts, advice, or information, even with different wording
3. Focus on semantic similarity rather than exact word matching
4. Consider agricultural context, techniques, timing, and outcomes
5. If no candidate fact is semantically similar enough, return null for best_match
6. **Find Contradictions**: Identify facts that directly conflict with the reference fact
  - "Use method A" vs "Use method B"
  - "Speed: 5 km/h" vs "Speed: 2 km/h"
  - "Apply morning" vs "Apply evening"
  - Different dosages, timings, methods, or conflicting advice

RESPOND WITH ONLY JSON:
{{
   "best_match": "exact text of best matching candidate fact or null if no good match",
   "reason": "detailed explanation of why this is the best match or why no match was found, including specific agricultural concepts that align or differ",
   "confidence": 0.0-1.0,
   "contradictions": [
       {{
           "contradicting_fact": "exact text of contradicting candidate fact",
           "contradiction_reason": "detailed explanation of how this fact contradicts the reference fact"
       }}
   ]
}}

Examples of what constitutes a match:
- "Apply NPK fertilizer" matches "Use balanced fertilizer with nitrogen, phosphorus, and potassium"
- "Sow wheat in November" matches "Plant wheat during late autumn"
- "Control pests with neem oil" matches "Use organic neem-based pesticide for pest management"

Examples of contradictions:
- "Sow wheat in November" contradicts "Sow wheat in February"
- "Apply 100kg fertilizer" contradicts "Apply 200kg fertilizer"
- "Water in morning" contradicts "Water in evening"
"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert agricultural fact comparison specialist. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            best_match = data.get('best_match')
            reason = data.get('reason', 'No reason provided')
            confidence = data.get('confidence', 0.0)
            contradictions = data.get('contradictions', [])
            
            # Verify the match is in our candidate list
            if best_match and best_match in pred_facts:
                return {'best_match': best_match, 'reason': reason, 'confidence': confidence, 'contradictions': contradictions}
            else:
                return {'best_match': None, 'reason': reason, 'confidence': 0.0, 'contradictions': contradictions}
            
        except Exception as e:
            if debug:
                print(f"Error in finding best match: {e}")
            return {'best_match': None, 'reason': f'Error during matching: {str(e)}', 'confidence': 0.0, 'contradictions': []}
        
        return {'best_match': None, 'reason': 'Unknown error occurred', 'confidence': 0.0, 'contradictions': []}

    def _extract_fact_text(self, fact_item: Any) -> str:
        """Extract fact text from various formats"""
        if isinstance(fact_item, str):
            return fact_item
        elif isinstance(fact_item, dict):
            return fact_item.get('fact', str(fact_item))
        else:
            return str(fact_item)

    def check_required_fields(self, predicted_facts: Any, required_fields: List[str]) -> Dict[str, Any]:
        """Check schema compliance for structured facts"""
        if not isinstance(predicted_facts, dict):
            return {
                'compliance_score': 0.0,
                'missing_fields': required_fields,
                'field_coverage': {}
            }

        facts_list = predicted_facts.get('facts', [])
        if not isinstance(facts_list, list):
            return {
                'compliance_score': 0.0,
                'missing_fields': required_fields,
                'field_coverage': {}
            }

        field_coverage = {}
        total_facts = len(facts_list)

        if total_facts == 0:
            return {
                'compliance_score': 0.0,
                'missing_fields': required_fields,
                'field_coverage': {}
            }

        for field in required_fields:
            facts_with_field = sum(1 for fact in facts_list if isinstance(fact, dict) and field in fact)
            field_coverage[field] = facts_with_field / total_facts

        compliance_score = np.mean(list(field_coverage.values()))
        missing_fields = [field for field, coverage in field_coverage.items() if coverage < 1.0]

        return {
            'compliance_score': compliance_score,
            'missing_fields': missing_fields,
            'field_coverage': field_coverage
        }

    def evaluate_category_classification(self, predicted_facts: Any, golden_facts: Any) -> Dict[str, Any]:
        """Evaluate category classification accuracy"""
        pred_categories = self._extract_categories(predicted_facts)
        gold_categories = self._extract_categories(golden_facts)

        if not pred_categories or not gold_categories:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'category_distribution': {}
            }

        # Calculate category-wise metrics
        all_categories = set(pred_categories + gold_categories)
        category_metrics = {}

        for category in all_categories:
            pred_count = pred_categories.count(category)
            gold_count = gold_categories.count(category)

            if gold_count > 0:
                precision = pred_count / len(pred_categories) if pred_categories else 0
                recall = min(pred_count, gold_count) / gold_count
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0

            category_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

        # Overall metrics
        overall_accuracy = len(set(pred_categories) & set(gold_categories)) / len(set(gold_categories)) if gold_categories else 0
        avg_precision = np.mean([m['precision'] for m in category_metrics.values()])
        avg_recall = np.mean([m['recall'] for m in category_metrics.values()])
        avg_f1 = np.mean([m['f1_score'] for m in category_metrics.values()])

        return {
            'accuracy': overall_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'category_distribution': category_metrics
        }

    def evaluate_location_dependency(self, predicted_facts: Any, golden_facts: Any) -> Dict[str, Any]:
        """Evaluate location dependency classification"""
        pred_locations = self._extract_location_dependencies(predicted_facts)
        gold_locations = self._extract_location_dependencies(golden_facts)

        if not pred_locations or not gold_locations:
            return {
                'accuracy': 0.0,
                'correct_predictions': 0,
                'total_predictions': len(pred_locations)
            }

        correct = sum(1 for p, g in zip(pred_locations, gold_locations) if p == g)
        accuracy = correct / len(gold_locations) if gold_locations else 0

        return {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': len(pred_locations)
        }

    def evaluate_confidence_accuracy(self, predicted_facts: Any, golden_facts: Any) -> Dict[str, Any]:
        """Evaluate confidence calibration"""
        pred_confidences = self._extract_confidences(predicted_facts)

        if not pred_confidences:
            return {
                'avg_confidence': 0.0,
                'confidence_distribution': {},
                'calibration_score': 0.0
            }

        avg_confidence = np.mean(pred_confidences)
        confidence_bins = {
            'low (0-0.3)': sum(1 for c in pred_confidences if 0 <= c <= 0.3),
            'medium (0.3-0.7)': sum(1 for c in pred_confidences if 0.3 < c <= 0.7),
            'high (0.7-1.0)': sum(1 for c in pred_confidences if 0.7 < c <= 1.0)
        }

        # Simple calibration score (can be enhanced with actual accuracy correlation)
        calibration_score = 1.0 - np.std(pred_confidences)  # Lower std = better calibration

        return {
            'avg_confidence': avg_confidence,
            'confidence_distribution': confidence_bins,
            'calibration_score': max(0, calibration_score)
        }

    def evaluate_fact_sft_model(self, predicted_facts: Any, golden_facts: Any) -> Dict[str, Any]:
        """Comprehensive evaluation for fact-based SFT models with enhanced category-wise matching"""

        metrics = {}

        # 1. JSON Format Validity
        json_result = self.check_valid_json(predicted_facts)
        metrics['json_validity'] = json_result['is_valid']

        # Use parsed data if valid JSON
        pred_data = json_result['parsed_data'] if json_result['is_valid'] else predicted_facts

        # 2. Schema Compliance
        required_fields = ['fact', 'category', 'location_dependency', 'confidence']
        metrics['schema_compliance'] = self.check_required_fields(pred_data, required_fields)

        # 3. Category Accuracy
        metrics['category_accuracy'] = self.evaluate_category_classification(pred_data, golden_facts)

        # 4. Location Dependency Accuracy
        metrics['location_classification'] = self.evaluate_location_dependency(pred_data, golden_facts)

        # 5. Enhanced Category-wise Semantic Accuracy
        metrics['semantic_accuracy'] = self.category_wise_fact_matching(pred_data, golden_facts)

        # 6. Confidence Calibration
        metrics['confidence_calibration'] = self.evaluate_confidence_accuracy(pred_data, golden_facts)

        return metrics

    # Helper methods for extracting different components
    def _extract_categories(self, facts_data: Any) -> List[str]:
        """Extract categories from structured facts"""
        if not isinstance(facts_data, dict) or 'facts' not in facts_data:
            return []

        categories = []
        for fact in facts_data.get('facts', []):
            if isinstance(fact, dict) and 'category' in fact:
                categories.append(fact['category'])
        return categories

    def _extract_location_dependencies(self, facts_data: Any) -> List[bool]:
        """Extract location dependencies from structured facts"""
        if not isinstance(facts_data, dict) or 'facts' not in facts_data:
            return []

        dependencies = []
        for fact in facts_data.get('facts', []):
            if isinstance(fact, dict) and 'location_dependency' in fact:
                dependencies.append(fact['location_dependency'])
        return dependencies

    def _extract_confidences(self, facts_data: Any) -> List[float]:
        """Extract confidence scores from structured facts"""
        if not isinstance(facts_data, dict) or 'facts' not in facts_data:
            return []

        confidences = []
        for fact in facts_data.get('facts', []):
            if isinstance(fact, dict) and 'confidence' in fact:
                try:
                    confidences.append(float(fact['confidence']))
                except (ValueError, TypeError):
                    continue
        return confidences


# Token tracker class (simplified version)
class TokenTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.calls = []

    def add_call(self, input_tokens: int, output_tokens: int, model: str, operation: str):
        cost = self.calculate_api_cost(input_tokens, output_tokens, model)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.calls.append({
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "model": model
        })

    def calculate_api_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        if model == "gpt-4o":
            input_cost_per_1k = 0.005
            output_cost_per_1k = 0.015
        else:
            input_cost_per_1k = 0.001
            output_cost_per_1k = 0.002

        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        return input_cost + output_cost


# Modified integration function to work with your CSV format
def integrate_enhanced_evaluation(df: pd.DataFrame, openai_api_key: str = None) -> pd.DataFrame:
    """
    Integrate enhanced category-wise evaluation with your CSV data format
    """
    evaluator = EnhancedFactEvaluator(openai_api_key)
    results = []

    print(f"Starting enhanced category-wise evaluation for {len(df)} samples...")

    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing row {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")

        try:
            # Get predictions and ground truth from your CSV columns
            predicted_facts = row['FT_facts']  # Your model output column
            golden_facts = row['GT_facts']     # Ground truth column
            query = row['query']               # Question column

            # Run enhanced evaluation
            metrics = evaluator.evaluate_fact_sft_model(predicted_facts, golden_facts)

            # Create result record
            result = {
                'input': query,
                'predicted_facts': predicted_facts,
                'golden_facts': golden_facts,

                # Core metrics
                'json_validity': metrics['json_validity'],
                'schema_compliance_score': metrics['schema_compliance']['compliance_score'],
                'missing_fields': metrics['schema_compliance']['missing_fields'],

                # Category metrics
                'category_accuracy': metrics['category_accuracy']['accuracy'],
                'category_precision': metrics['category_accuracy']['precision'],
                'category_recall': metrics['category_accuracy']['recall'],
                'category_f1': metrics['category_accuracy']['f1_score'],

                # Location classification
                'location_accuracy': metrics['location_classification']['accuracy'],

                # Enhanced semantic accuracy (Category-wise IoGT)
                'iogt_score': metrics['semantic_accuracy']['iogt_score'],
                'total_matches': metrics['semantic_accuracy']['total_matches'],
                'total_gold_facts': metrics['semantic_accuracy']['total_gold_facts'],
                'total_contradictions': metrics['semantic_accuracy']['total_contradictions'],
                'contradictions': metrics['semantic_accuracy']['contradictions'],  # Separate column for contradictions
                'category_results': metrics['semantic_accuracy']['category_results'],

                # Confidence calibration
                'avg_confidence': metrics['confidence_calibration']['avg_confidence'],
                'confidence_calibration': metrics['confidence_calibration']['calibration_score'],

                # Full metrics for detailed analysis
                'full_metrics': metrics
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Add error record
            error_result = {
                'input': row.get('query', ''),
                'predicted_facts': row.get('FT_facts', ''),
                'golden_facts': row.get('GT_facts', ''),
                'error': str(e)
            }
            results.append(error_result)

    return pd.DataFrame(results)


def generate_enhanced_report(results_df: pd.DataFrame) -> Dict:
    """Generate enhanced evaluation report with category-wise analysis - FIXED VERSION"""

    # FIXED: Correct filtering logic for valid results
    if 'error' in results_df.columns:
        # Filter out rows with errors (only keep rows where error is NaN)
        valid_results = results_df[results_df['error'].isna()].copy()
    else:
        # No error column, use all results
        valid_results = results_df.copy()

    if len(valid_results) == 0:
        return {"error": "No valid results found"}

    # Analyze category-wise performance
    category_analysis = {}
    for idx, row in valid_results.iterrows():
        if 'category_results' in row and row['category_results']:
            # Handle both string and dict types for category_results
            if isinstance(row['category_results'], str):
                try:
                    import ast
                    category_results = ast.literal_eval(row['category_results'])
                except:
                    continue
            else:
                category_results = row['category_results']
            
            for category, cat_metrics in category_results.items():
                if category not in category_analysis:
                    category_analysis[category] = {
                        'total_samples': 0,
                        'total_matches': 0,
                        'total_gold_facts': 0,
                        'total_pred_facts': 0
                    }
                
                category_analysis[category]['total_samples'] += 1
                category_analysis[category]['total_matches'] += cat_metrics['matches']
                category_analysis[category]['total_gold_facts'] += cat_metrics['gold_count']
                category_analysis[category]['total_pred_facts'] += cat_metrics['pred_count']

    # Calculate category-wise IoGT scores
    for category in category_analysis:
        cat_data = category_analysis[category]
        cat_data['iogt_score'] = cat_data['total_matches'] / cat_data['total_gold_facts'] if cat_data['total_gold_facts'] > 0 else 0

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    report = {
        'summary': {
            'total_samples': len(results_df),
            'valid_samples': len(valid_results),
            'error_samples': len(results_df) - len(valid_results)
        },

        'json_validity': {
            'valid_json_rate': convert_numpy_types(valid_results['json_validity'].mean()) if 'json_validity' in valid_results.columns else 0,
            'invalid_json_count': convert_numpy_types((~valid_results['json_validity']).sum()) if 'json_validity' in valid_results.columns else 0
        },

        'schema_compliance': {
            'avg_compliance_score': convert_numpy_types(valid_results['schema_compliance_score'].mean()) if 'schema_compliance_score' in valid_results.columns else 0,
            'perfect_compliance_rate': convert_numpy_types((valid_results['schema_compliance_score'] == 1.0).mean()) if 'schema_compliance_score' in valid_results.columns else 0
        },

        'location_classification': {
            'accuracy': convert_numpy_types(valid_results['location_accuracy'].mean()) if 'location_accuracy' in valid_results.columns else 0
        },

        'enhanced_semantic_accuracy': {
            'avg_iogt_score': convert_numpy_types(valid_results['iogt_score'].mean()) if 'iogt_score' in valid_results.columns else 0,
            'total_matches': convert_numpy_types(valid_results['total_matches'].sum()) if 'total_matches' in valid_results.columns else 0,
            'total_gold_facts': convert_numpy_types(valid_results['total_gold_facts'].sum()) if 'total_gold_facts' in valid_results.columns else 0,
            'total_contradictions': convert_numpy_types(valid_results['total_contradictions'].sum()) if 'total_contradictions' in valid_results.columns else 0
        },

        'category_wise_performance': category_analysis,

        'confidence_analysis': {
            'avg_confidence': convert_numpy_types(valid_results['avg_confidence'].mean()) if 'avg_confidence' in valid_results.columns else 0,
            'calibration_score': convert_numpy_types(valid_results['confidence_calibration'].mean()) if 'confidence_calibration' in valid_results.columns else 0
        }
    }

    return report


def print_enhanced_report(report: Dict):
    """Print formatted enhanced evaluation report with category-wise analysis"""

    if 'error' in report:
        print(f"❌ Error: {report['error']}")
        return

    print("\n" + "="*80)
    print("           ENHANCED CATEGORY-WISE FACT EVALUATION REPORT")
    print("="*80)

    print(f"\n📊 SUMMARY:")
    print(f"   Total Samples: {report['summary']['total_samples']}")
    print(f"   Valid Samples: {report['summary']['valid_samples']}")
    print(f"   Error Samples: {report['summary']['error_samples']}")

    print(f"\n🔧 JSON VALIDITY:")
    print(f"   Valid JSON Rate: {report['json_validity']['valid_json_rate']:.3f}")
    print(f"   Invalid JSON Count: {report['json_validity']['invalid_json_count']}")

    print(f"\n📋 SCHEMA COMPLIANCE:")
    print(f"   Avg Compliance Score: {report['schema_compliance']['avg_compliance_score']:.3f}")
    print(f"   Perfect Compliance Rate: {report['schema_compliance']['perfect_compliance_rate']:.3f}")

    print(f"\n📍 LOCATION CLASSIFICATION:")
    print(f"   Accuracy: {report['location_classification']['accuracy']:.3f}")

    print(f"\n🎯 ENHANCED SEMANTIC ACCURACY (Category-wise IoGT):")
    print(f"   Overall IoGT Score: {report['enhanced_semantic_accuracy']['avg_iogt_score']:.3f}")
    print(f"   Total Matches: {report['enhanced_semantic_accuracy']['total_matches']}")
    print(f"   Total Gold Facts: {report['enhanced_semantic_accuracy']['total_gold_facts']}")

    print(f"\n📂 CATEGORY-WISE PERFORMANCE:")
    for category, metrics in report['category_wise_performance'].items():
        print(f"   {category}:")
        print(f"     IoGT Score: {metrics['iogt_score']:.3f}")
        print(f"     Matches: {metrics['total_matches']}/{metrics['total_gold_facts']}")
        print(f"     Samples: {metrics['total_samples']}")

    print(f"\n🎲 CONFIDENCE ANALYSIS:")
    print(f"   Avg Confidence: {report['confidence_analysis']['avg_confidence']:.3f}")
    print(f"   Calibration Score: {report['confidence_analysis']['calibration_score']:.3f}")


# Modified main function to work with your CSV
def enhanced_main(csv_file_path: str = "sample_data.csv", openai_api_key: str = None, sample_size: int = None):
    """Enhanced main function with category-wise evaluation for your CSV data"""
    
    print("Loading data...")
    df = pd.read_csv(csv_file_path)
    
    if sample_size:
        df = df.head(sample_size)  # Sample for testing
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns in CSV: {list(df.columns)}")

    print("\nStarting enhanced category-wise evaluation...")
    results_df = integrate_enhanced_evaluation(df, openai_api_key)

    report = generate_enhanced_report(results_df)
    print_enhanced_report(report)

    # Save results
    results_df.to_csv('enhanced_category_wise_evaluation_results.csv', index=False)

    with open('enhanced_category_wise_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)  # Added default=str to handle any remaining serialization issues

    print(f"\n✅ Results saved to 'enhanced_category_wise_evaluation_results.csv'")
    print(f"✅ Report saved to 'enhanced_category_wise_evaluation_report.json'")

    return results_df, report


if __name__ == "__main__":
    # Set your OpenAI API key here or as environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")  # set OPENAI_API_KEY in your environment / .env
    
    # Run main evaluation on your CSV
    print("\n" + "="*50)
    print("Running evaluation on your CSV data...")
    results, report = enhanced_main(
        csv_file_path="sample_data.csv", 
        openai_api_key=openai_api_key,
        sample_size=None  # Process all data
    )
