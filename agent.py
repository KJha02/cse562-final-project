from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List

class Agent:
    def __init__(self, thinking_mode: int = 0, main_model_name: str = "Qwen/Qwen3-8B", small_model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initialize the Agent with specific thinking mode and model.
        
        Args:
            thinking_mode: 0 for no thinking, 1 for thinking, 2 for guided thinking
            model_name: Name of the main model to use (default: Qwen3-8B)
        """
        self.thinking_mode = thinking_mode
        
        if thinking_mode == 2:
            # Initialize both small and large models
            self.small_model = LLM(model=small_model_name)
            self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)
            self.large_model = LLM(model=main_model_name)
            self.large_tokenizer = AutoTokenizer.from_pretrained(main_model_name)
        else:
            self.model = LLM(model=main_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(main_model_name)
        
        # Configure sampling parameters
        self.small_sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=30
        )
        self.large_sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=1024
        )

    def _format_prompt(self, question: str, options: List[str], for_duration: bool = False) -> str:
        """
        Format the prompt based on whether it's for duration estimation or answer.
        """
        options_text = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        
        if for_duration:
            return f"""Given this multiple choice question, estimate how many seconds the larger model should spend thinking about it. Respond with just a number between 1 and 30.

Question: {question}

Options:
{options_text}"""
        else:
            return f"""Question: {question}

Options:
{options_text}

Please select the most appropriate answer from the options above. Respond with just the letter (A, B, C, etc.) corresponding to your choice."""

    def _get_completion(self, prompt: str, model: LLM, tokenizer: AutoTokenizer, sampling_params: SamplingParams, thinking: bool = False, think_duration: int = None) -> str:
        """Helper method to get completion from the model."""
        messages = [{"role": "user", "content": prompt}]
        
        # If thinking mode is enabled and duration specified, add it to the prompt
        if thinking and think_duration:
            messages[0]["content"] += f"\n\nPlease think about this for {think_duration} seconds before answering."
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        
        # Generate output
        outputs = model.generate([text], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def get_answers(self, questions: List[str], options_list: List[List[str]]) -> List[str]:
        """
        Perform batched inference to get answers for multiple questions.
        """
        if len(questions) != len(options_list):
            raise ValueError("Number of questions must match number of option lists")

        answers = []
        
        for question, options in zip(questions, options_list):
            if self.thinking_mode == 2:
                # First, use small model to determine thinking duration
                duration_prompt = self._format_prompt(question, options, for_duration=True)
                duration_str = self._get_completion(
                    duration_prompt,
                    self.small_model,
                    self.small_tokenizer,
                    self.small_sampling_params,
                    thinking=False
                )
                breakpoint()
                try:
                    think_duration = min(max(int(duration_str), 1), 30)
                except ValueError:
                    think_duration = 5  # default if parsing fails
                
                # Then use large model with the suggested thinking duration
                answer = self._get_completion(
                    self._format_prompt(question, options),
                    self.large_model,
                    self.large_tokenizer,
                    self.large_sampling_params,
                    thinking=True,
                    think_duration=think_duration
                )
            else:
                # Mode 0 or 1: direct completion with appropriate thinking mode
                answer = self._get_completion(
                    self._format_prompt(question, options),
                    self.model,
                    self.tokenizer,
                    self.large_sampling_params,
                    thinking=(self.thinking_mode == 1)
                )
            
            # Clean the answer to just the letter
            answer = answer.upper()
            answer = answer[0] if answer and answer[0].isalpha() else "INVALID"
            answers.append(answer)
            
        return answers
