import json
import argparse
from tqdm import tqdm
from agent import Agent

def load_questions(question_path):
    """Load questions from a JSONL file."""
    questions = []
    options_list = []
    correct_answers = []
    
    with open(question_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                questions.append(data['question'])
                options_list.append([data['options'][k] for k in sorted(data['options'].keys())])
                correct_answers.append(data['answer_idx'])
            if len(questions) > 10:
                break
    
    return questions, options_list, correct_answers

def evaluate_agent(agent, questions, options_list, correct_answers):
    """Evaluate agent's performance on the questions."""
    predictions = agent.get_answers(questions, options_list)
    breakpoint()
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predictions, correct_answers) if pred == true)
    accuracy = correct / len(predictions)
    
    return {
        'total_questions': len(predictions),
        'correct_answers': correct,
        'accuracy': accuracy,
        'predictions': predictions
    }

def save_results(results, output_path):
    """Save evaluation results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance on medical questions')
    parser.add_argument('--thinking_mode', type=int, default=0, choices=[0, 1, 2],
                       help='0: no thinking, 1: thinking, 2: guided thinking')
    parser.add_argument('--main_model', type=str, default="Qwen/Qwen3-4B",
                       help='Name of the main model to use')
    parser.add_argument('--small_model', type=str, default="Qwen/Qwen3-0.6B",
                       help='Name of the small model to use (only for thinking_mode=2)')
    parser.add_argument('--question_path', type=str, default='data_clean/questions/US/test.jsonl',
                       help='Path to the questions JSONL file')
    parser.add_argument('--output_path', type=str, default='results.json',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Load questions
    questions, options_list, correct_answers = load_questions(args.question_path)
    
    # Initialize agent
    agent = Agent(
        thinking_mode=args.thinking_mode,
        main_model_name=args.main_model,
        small_model_name=args.small_model
    )
    
    # Evaluate
    print(f"Evaluating with thinking_mode={args.thinking_mode}")
    results = evaluate_agent(agent, questions, options_list, correct_answers)
    
    # Add configuration to results
    results['config'] = {
        'thinking_mode': args.thinking_mode,
        'main_model': args.main_model,
        'small_model': args.small_model,
        'question_path': args.question_path
    }
    
    # Save results
    save_results(results, args.output_path)
    print(f"\nResults saved to {args.output_path}")
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct_answers']}/{results['total_questions']})")

if __name__ == "__main__":
    main()