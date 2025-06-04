import json
import argparse
import pandas as pd
from tqdm import tqdm
from agent import Agent
import os

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
                yield questions, options_list, correct_answers

                questions = []
                options_list = []
                correct_answers = []

def evaluate_agent(agent, questions, options_list, correct_answers, epoch):
    """Evaluate agent's performance on the questions."""
    predictions, char_counts, times = agent.get_answers(questions, options_list)
    
    # Create DataFrame with all information
    results_df = pd.DataFrame({
        'epoch': [epoch] * len(questions),
        'question': questions,
        'options': options_list,
        'true_answer': correct_answers,
        'predicted_answer': predictions,
        'correct': [pred == true for pred, true in zip(predictions, correct_answers)],
        'char_count': char_counts,
        'time_seconds': times,
        'thinking_mode': [agent.thinking_mode] * len(questions),
        'prompt_mode': [agent.prompt_mode] * len(questions),
        'main_model': [agent.main_model_name] * len(questions),
        'small_model': [agent.small_model_name] * len(questions)
    })
    
    # Calculate summary statistics
    summary = {
        'total_questions': len(predictions),
        'correct_answers': results_df['correct'].sum(),
        'accuracy': results_df['correct'].mean(),
        'avg_chars_per_question': results_df['char_count'].mean(),
        'total_chars': results_df['char_count'].sum(),
        'avg_time_per_question': results_df['time_seconds'].mean(),
        'total_time': results_df['time_seconds'].sum()
    }
    
    return summary, results_df

def save_results(results, results_df, output_path):
    """Save evaluation results to JSON and CSV files."""
    
    # Save detailed results to CSV
    csv_path = output_path.replace('.json', '') + '_detailed.csv'
    # Check if file exists and append without header if it does
    if os.path.exists(csv_path):
        results_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_path, index=False)

def get_last_epoch(csv_path, config):
    """Get the last epoch for the current configuration."""
    if not os.path.exists(csv_path):
        return -1
    
    try:
        df = pd.read_csv(csv_path)
        # Filter for current configuration
        mask = (
            (df['thinking_mode'] == config['thinking_mode']) &
            (df['main_model'] == config['main_model']) &
            (df['small_model'] == config['small_model']) &
            (df['prompt_mode'] == config['prompt_mode'])
        )
        if mask.any():
            return df[mask]['epoch'].max()
        return -1
    except (pd.errors.EmptyDataError, KeyError):
        return -1

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance on medical questions')
    parser.add_argument('--thinking_mode', type=int, default=0, choices=[0, 1, 2],
                       help='0: no thinking, 1: thinking, 2: guided thinking')
    parser.add_argument('--prompt_mode', type=str, default="seconds", choices=["seconds", "relative_size", 'tokens'],)
    parser.add_argument('--main_model', type=str, default="Qwen/Qwen3-4B",
                       help='Name of the main model to use')
    parser.add_argument('--small_model', type=str, default="Qwen/Qwen3-0.6B",
                       help='Name of the small model to use (only for thinking_mode=2)')
    parser.add_argument('--question_path', type=str, default='data_clean/questions/US/test.jsonl',
                       help='Path to the questions JSONL file')
    parser.add_argument('--output_path', type=str, default='results.json',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    

    # Initialize agent
    agent = Agent(
        thinking_mode=args.thinking_mode,
        main_model_name=args.main_model,
        small_model_name=args.small_model,
        prompt_mode=args.prompt_mode
    )

    # Create config dict
    config = {
        'thinking_mode': args.thinking_mode,
        'main_model': args.main_model,
        'small_model': args.small_model,
        'prompt_mode': args.prompt_mode
    }

    # Get the last epoch for this configuration
    csv_path = args.output_path.replace('.json', '') + '_detailed.csv'
    last_epoch = get_last_epoch(csv_path, config)
    
    # Load questions
    dataloader = load_questions(args.question_path)
    
    # Skip to the last completed epoch + 1
    for _ in range(last_epoch + 1):
        try:
            next(dataloader)
        except StopIteration:
            print(f"All epochs completed for this configuration")
            return

    # Continue evaluation from the next epoch
    epoch = last_epoch + 1
    for questions, options_list, correct_answers in dataloader:
        print(f"Evaluating epoch {epoch} with thinking_mode={args.thinking_mode}")
        results, results_df = evaluate_agent(agent, questions, options_list, correct_answers, epoch)
        
        # Add configuration to results
        results['config'] = config
        results['epoch'] = epoch
        
        # Save results
        save_results(results, results_df, args.output_path)
        
        epoch += 1


if __name__ == "__main__":
    main()