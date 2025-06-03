from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Tuple
from tqdm import tqdm
import time

class Agent:
    def __init__(self, thinking_mode: int = 0, main_model_name: str = "Qwen/Qwen3-8B", small_model_name: str = "Qwen/Qwen3-0.6B", prompt_mode: str = "seconds"):
        """
        Initialize the Agent with specific thinking mode and model.
        
        Args:
            thinking_mode: 0 for no thinking, 1 for thinking, 2 for guided thinking
            model_name: Name of the main model to use (default: Qwen3-8B)
            prompt_mode: How to prompt for thinking duration ('relative_size', 'tokens', or 'seconds')
        """
        self.thinking_mode = thinking_mode
        self.prompt_mode = prompt_mode
        self.main_model_name = main_model_name
        self.small_model_name = small_model_name

        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)
        self.large_model = AutoModelForCausalLM.from_pretrained(
            main_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.large_tokenizer = AutoTokenizer.from_pretrained(main_model_name)
        
        # Configure sampling parameters
        self.small_sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=30
        )
        self.large_sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=5000
        )

    def _format_prompt(self, question: str, options: List[str], for_duration: bool = False) -> str:
        """
        Format the prompt based on whether it's for duration estimation or answer.
        """
        options_text = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        
        if for_duration:
            if self.prompt_mode == "relative_size":
                return f"""Given this multiple choice question, assess how complicated it is.

Question: {question}

Rate the complexity of this question on a scale from 'very easy' to 'very hard'. This will determine how long the model should think about it.
Respond with exactly one of these options: very easy, easy, moderate, hard, very hard."""
            elif self.prompt_mode == "tokens":
                return f"""Given this multiple choice question, estimate how many tokens the larger model should spend thinking about it.

Question: {question}

Given the above multiple choice question, estimate how many tokens the larger model should spend thinking about it. Respond with just a number between 1 and 500. Only respond with a number, no other text."""
            else:  # seconds mode
                return f"""Given this multiple choice question, estimate how many seconds the larger model should spend thinking about it.

Question: {question}

Given the above multiple choice question, estimate how many seconds the larger model should spend thinking about it. Respond with just a number between 1 and 60. Only respond with a number, no other text."""
        else:
            return f"""Question: {question}

Options:
{options_text}

Please select the most appropriate answer from the options above. Respond with just the letter (A, B, C, etc.) corresponding to your choice."""

    def _get_completion(self, prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, sampling_params: SamplingParams, thinking: bool = False, think_duration: int = None) -> str:
        """Helper method to get completion from the model."""
        messages = [{"role": "user", "content": prompt}]
        
        # If thinking mode is enabled and duration specified, add it to the prompt
        if thinking and think_duration:
            if self.prompt_mode == "relative_size":
                # Map seconds back to complexity levels
                if think_duration <= 1:
                    complexity = "very little"
                elif think_duration <= 5:
                    complexity = "a little"
                elif think_duration <= 15:
                    complexity = "moderately"
                elif think_duration <= 30:
                    complexity = "hard"
                else:
                    complexity = "very hard"
                messages[0]["content"] += f"\n\nPlease think about this {complexity} before answering."
            elif self.prompt_mode == "tokens":
                messages[0]["content"] += f"\n\nPlease think for {think_duration} tokens before answering."
            else:  # seconds mode
                messages[0]["content"] += f"\n\nPlease think about this for {think_duration} seconds before answering."
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        
        # Prepare model inputs
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate output
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=sampling_params.max_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content if enabled
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
        except ValueError:
            index = 0
            
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content.strip()

    def _parse_answer_with_small_model(self, answer_text: str) -> str:
        """
        Use the small model to parse the answer text and extract a single letter answer.
        """
        prompt = f"""Given this answer text from a multiple choice question, what single letter (A, B, C, D, or E) is the answer?
If you cannot determine the letter, output 'X'.

Answer text: {answer_text}

Output just a single letter:"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.small_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = self.small_tokenizer([text], return_tensors="pt").to(self.small_model.device)
        generated_ids = self.small_model.generate(
            **model_inputs,
            max_new_tokens=10,
            temperature=0.1,
            top_p=0.95
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        parsed = self.small_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        # Extract first letter and validate
        parsed = parsed.upper()
        if parsed and parsed[0] in 'ABCDEX':
            return parsed[0]
        return 'X'

    def get_answers(self, questions: List[str], options_list: List[List[str]]) -> Tuple[List[str], List[int], List[float]]:
        """
        Perform batched inference to get answers for multiple questions.
        Returns a tuple of (answers, character_counts, time_taken)
        """
        if len(questions) != len(options_list):
            raise ValueError("Number of questions must match number of option lists")

        answers = []
        char_counts = []
        time_taken = []

        for question, options in tqdm(zip(questions, options_list), total=len(questions)):
            start_time = time.time()
            
            if self.thinking_mode == 2:
                # First, use small model to determine thinking duration
                duration_prompt = self._format_prompt(question, options, for_duration=True)
                duration_str = self._get_completion(
                    duration_prompt,
                    self.small_model,
                    self.small_tokenizer,
                    self.small_sampling_params,
                    thinking=False,
                )
                
                # Parse the duration based on prompt_mode
                if self.prompt_mode == "relative_size":
                    complexity_map = {
                        "very easy": 1,
                        "easy": 5,
                        "moderate": 15,
                        "hard": 30,
                        "very hard": 60
                    }
                    think_duration = complexity_map.get(duration_str.lower().strip(), 10)
                elif self.prompt_mode == "tokens":
                    try:
                        think_duration = min(max(int(duration_str), 1), 500)
                    except ValueError:
                        think_duration = 10
                        print(f"Error parsing token count: {duration_str}")
                else:  # seconds mode
                    try:
                        think_duration = min(max(int(duration_str), 1), 60)
                    except ValueError:
                        think_duration = 10
                        print(f"Error parsing duration: {duration_str}")
                
                # Then use large model with the suggested thinking duration
                raw_answer = self._get_completion(
                    self._format_prompt(question, options),
                    self.large_model,
                    self.large_tokenizer,
                    self.large_sampling_params,
                    thinking=True,
                    think_duration=think_duration
                )
                answer = self._parse_answer_with_small_model(raw_answer)
                char_counts.append(len(duration_str) + len(raw_answer))
            else:
                # Mode 0 or 1: direct completion with appropriate thinking mode
                raw_answer = self._get_completion(
                    self._format_prompt(question, options),
                    self.large_model,
                    self.large_tokenizer,
                    self.large_sampling_params,
                    thinking=(self.thinking_mode == 1)
                )
                answer = self._parse_answer_with_small_model(raw_answer)
                char_counts.append(len(raw_answer))
            
            answers.append(answer)
            time_taken.append(time.time() - start_time)
            
        return answers, char_counts, time_taken
