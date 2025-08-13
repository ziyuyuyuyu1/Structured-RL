#!/usr/bin/env python3
"""
LLM-based sentence processing with one-by-one processing and gradual prompt modification.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging

from openai import OpenAI
import torch

# Predefined meanings for mathematical classification
PREDEFINED_MATHEMATICAL_MEANINGS = [
    "Nonsense or meaningless statements (no valid logical or mathematical content)",
    "Statements about factoring a polynomial or equation, including dividing by a factor, identifying a specific linear factor (e.g., $x - 1$), and giving the full factorization into linear factors (listing all roots)",
    "Verifying that a given value is a root (by substitution or by applying the rational root test)",
    "Giving the roots of a cubic equation using the cubic formula",
    "Stating a general property of cubic polynomials (e.g., \"a cubic has three roots\")",
    "Identifying the coefficients of a given polynomial",
    "Transforming the equation into a new form, including reciprocal transformation (e.g. dividing through by $x^3$) and change of variables (e.g. $x = y + \\frac{2}{3}$)",
    "Other meaningful statements not covered above"
]


@dataclass
class ProcessingResult:
    """Result of processing a single sentence."""
    sentence: str
    prompt_id: int
    original_prompt: str
    modified_prompt: str
    llm_response: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptModifier:
    """Base class for prompt modification strategies."""
    
    def modify_prompt(self, base_prompt: str, sentence: str, prompt_id: int, 
                     previous_results: List[ProcessingResult], **kwargs) -> str:
        """Modify the base prompt based on context."""
        return base_prompt


@dataclass
class AdaptivePromptModifier(PromptModifier):
    """Adaptive prompt modification based on previous results."""
    
    def modify_prompt(self, base_prompt: str, sentence: str, prompt_id: int,
                     previous_results: List[ProcessingResult], **kwargs) -> str:
        """Modify prompt based on previous processing results."""
        if not previous_results:
            return base_prompt
        
        # Analyze previous results to adapt the prompt
        recent_results = previous_results[-5:]  # Last 5 results
        
        # Count response patterns
        response_patterns = {}
        for result in recent_results:
            response = result.llm_response.strip()
            response_patterns[response] = response_patterns.get(response, 0) + 1
        
        # Add context about recent patterns
        context_lines = []
        if len(response_patterns) > 1:
            context_lines.append("Recent response patterns:")
            for response, count in sorted(response_patterns.items(), key=lambda x: x[1], reverse=True):
                context_lines.append(f"  - '{response}': {count} times")
        
        # Add sentence-specific context
        context_lines.append(f"\nProcessing sentence {prompt_id + 1}: {sentence}")
        
        modified_prompt = base_prompt + "\n\n" + "\n".join(context_lines)
        return modified_prompt


@dataclass
class FeedbackBasedModifier(PromptModifier):
    """Modify prompt based on explicit feedback."""
    
    feedback_rules: Dict[str, str] = field(default_factory=dict)
    
    def modify_prompt(self, base_prompt: str, sentence: str, prompt_id: int,
                     previous_results: List[ProcessingResult], **kwargs) -> str:
        """Modify prompt based on feedback rules."""
        modified_prompt = base_prompt
        
        # Apply feedback rules
        for pattern, modification in self.feedback_rules.items():
            if pattern in sentence.lower():
                modified_prompt += f"\n\nNote: {modification}"
        
        return modified_prompt


class LLMSentenceProcessor:
    """Process sentences one by one using an LLM API with gradual prompt modification."""
    
    def __init__(
        self,
        api_url: str = "http://10.220.5.151:30500/v1",
        api_key: str = "",
        model_name: str = "gpt-oss-120b",
        max_length: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 512,
        timeout: int = 30,
        use_fixed_meanings: bool = False,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
            timeout=timeout
        )
        
        # Processing state
        self.results: List[ProcessingResult] = []
        self.prompt_modifier: Optional[PromptModifier] = None
        
        # Meanings management
        self.meanings: List[str] = []
        self.next_meaning_id: int = 0
        self.use_fixed_meanings: bool = use_fixed_meanings
        
        # Prompt tracking
        self.prompt_to_sentences: Dict[int, List[int]] = {}  # prompt_id -> sentence_indices
        self.sentence_to_prompt: Dict[int, int] = {}  # sentence_index -> prompt_id
        self.prompt_meaning_distributions: Dict[int, Dict[int, int]] = {}  # prompt_id -> {meaning_id -> count}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized OpenAI client with base URL: {api_url}")
        if api_key:
            self.logger.info("API key provided")
        else:
            self.logger.warning("No API key provided - requests may fail")
    
    def set_prompt_modifier(self, modifier: PromptModifier):
        """Set the prompt modification strategy."""
        self.prompt_modifier = modifier
    
    def test_api_connection(self) -> bool:
        """Test the API connection with a simple request."""
        test_prompt = "Hello, this is a test message. Please respond with 'API test successful'."
        
        try:
            response = self._generate_response(test_prompt)
            self.logger.info(f"API test successful. Response: {response}")
            return True
        except Exception as e:
            self.logger.error(f"API test failed: {e}")
            return False
    
    def initialize_meanings(self, initial_meanings: List[str]):
        """Initialize the meanings list with starting meanings."""
        self.meanings = initial_meanings.copy()
        self.next_meaning_id = len(initial_meanings)
        self.logger.info(f"Initialized with {len(initial_meanings)} meanings: {initial_meanings}")
    
    def add_new_meaning(self, meaning_description: str) -> int:
        """Add a new meaning and return its ID."""
        meaning_id = self.next_meaning_id
        self.meanings.append(meaning_description)
        self.next_meaning_id += 1
        self.logger.info(f"Added new meaning {meaning_id}: {meaning_description}")
        return meaning_id
    
    def create_dynamic_prompt(self) -> str:
        """Create a prompt with current meanings."""
        if self.use_fixed_meanings:
            return create_fixed_classification_prompt(self.meanings)
        else:
            return create_classification_prompt(self.meanings)
    
    def parse_response(self, response: str) -> Tuple[Optional[int], Optional[str]]:
        """Parse the LLM response to extract meaning ID and description."""
        response = response.strip()
        
        # Check for "Create new meaning" (only in dynamic mode)
        if not self.use_fixed_meanings and response.lower().startswith("create new meaning"):
            # Extract the description after "Create new meaning"
            description = response[len("Create new meaning"):].strip()
            if description.startswith(":"):
                description = description[1:].strip()
            return None, description
        
        # Check for "Meaning X" format
        if response.lower().startswith("meaning"):
            parts = response.split(":", 1)
            if len(parts) >= 2:
                try:
                    meaning_id = int(parts[0].split()[1])
                    description = parts[1].strip()
                    return meaning_id, description
                except (ValueError, IndexError):
                    pass
        
        # Check for "X: description" format (new format)
        import re
        number_colon_match = re.match(r'^(\d+)\s*:\s*(.+)$', response.strip())
        if number_colon_match:
            try:
                meaning_id = int(number_colon_match.group(1))
                description = number_colon_match.group(2).strip()
                return meaning_id, description
            except (ValueError, IndexError):
                pass
        
        # Check for "X - description" format
        number_dash_match = re.match(r'^(\d+)\s*-\s*(.+)$', response.strip())
        if number_dash_match:
            try:
                meaning_id = int(number_dash_match.group(1))
                description = number_dash_match.group(2).strip()
                return meaning_id, description
            except (ValueError, IndexError):
                pass
        
        # Find the first number in the response and use it as meaning ID
        first_number_match = re.search(r'(\d+)', response)
        if first_number_match:
            try:
                meaning_id = int(first_number_match.group(1))
                # Extract everything after the first number as description
                # Remove common separators like :, -, *, etc.
                description = re.sub(r'^\d+\s*[:\-\*\.\s]*', '', response).strip()
                return meaning_id, description
            except (ValueError, IndexError):
                pass
        
        # Try to extract meaning ID from anywhere in the response
        meaning_match = re.search(r'meaning\s+(\d+)', response.lower())
        if meaning_match:
            try:
                meaning_id = int(meaning_match.group(1))
                # Extract description after the meaning ID
                description_match = re.search(r'meaning\s+\d+\s*:?\s*(.+)', response, re.IGNORECASE)
                description = description_match.group(1).strip() if description_match else ""
                return meaning_id, description
            except (ValueError, IndexError):
                pass
        
        # If we can't parse it, handle based on mode
        if self.use_fixed_meanings:
            # In fixed mode, default to Meaning 0 if we can't parse
            return 0, "Unable to parse response, defaulting to Meaning 0"
        else:
            # In dynamic mode, assume it's a new meaning
            return None, response
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using OpenAI client."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=False,
                n=1,
                stop=None,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            
            # Extract response content
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("No response content received")
                
        except Exception as e:
            self.logger.error(f"OpenAI API request failed: {e}")
            raise
    
    def process_sentence(
        self, 
        sentence: str, 
        prompt_id: int, 
        callback: Optional[Callable[[ProcessingResult], None]] = None,
        **kwargs
    ) -> ProcessingResult:
        """Process a single sentence with dynamic meaning expansion."""
        start_time = time.time()
        
        # Create dynamic prompt with current meanings
        base_prompt = self.create_dynamic_prompt()
        
        # Create modified prompt if modifier is set
        if self.prompt_modifier:
            modified_prompt = self.prompt_modifier.modify_prompt(
                base_prompt, sentence, prompt_id, self.results, **kwargs
            )
        else:
            modified_prompt = base_prompt
        
        # Replace placeholder with actual sentence
        final_prompt = modified_prompt.replace("[Insert new sentence here]", sentence)
        
        # Generate response
        try:
            llm_response = self._generate_response(final_prompt)
        except Exception as e:
            self.logger.error(f"Error generating response for sentence {prompt_id}: {e}")
            llm_response = f"ERROR: {str(e)}"
        
        processing_time = time.time() - start_time
        
        # Parse response and handle new meanings
        meaning_id, meaning_description = self.parse_response(llm_response)
        
        # Add new meaning if created (only in dynamic mode)
        if not self.use_fixed_meanings and meaning_id is None and meaning_description:
            meaning_id = self.add_new_meaning(meaning_description)
        
        # Create result with additional metadata
        result = ProcessingResult(
            sentence=sentence,
            prompt_id=prompt_id,
            original_prompt=base_prompt,
            modified_prompt=modified_prompt,
            llm_response=llm_response,
            processing_time=processing_time,
            metadata={
                **kwargs,
                "meaning_id": meaning_id,
                "meaning_description": meaning_description,
                "total_meanings": len(self.meanings)
            }
        )
        
        # Store result
        self.results.append(result)
        
        # Call callback if provided
        if callback:
            callback(result)
        
        self.logger.info(f"Processed sentence {prompt_id} in {processing_time:.2f}s: {llm_response[:100]}...")
        if meaning_id is not None:
            self.logger.info(f"Assigned to Meaning {meaning_id}")
        else:
            self.logger.info(f"Created new meaning: {meaning_description}")
        
        return result
    
    def process_sentences_from_jsonl(
        self,
        jsonl_path: str,
        output_path: Optional[str] = None,
        start_from: int = 0,
        max_sentences: Optional[int] = None,
        callback: Optional[Callable[[ProcessingResult], None]] = None,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process sentences from a JSONL file one by one with dynamic meaning expansion."""
        sentences = []
        sentences_prompt = []
        prompt_id = 0
        
        # Read sentences from JSONL and build prompt mapping
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                code = obj.get("code")
                if isinstance(code, list):
                    valid_sentences = [s for s in code if isinstance(s, str) and s.strip()]
                    sentences.extend(valid_sentences)
                    
                    # Track which sentences belong to which prompt
                    start_idx = len(sentences) - len(valid_sentences)
                    for i, sentence in enumerate(valid_sentences):
                        sentence_idx = start_idx + i
                        self.sentence_to_prompt[sentence_idx] = prompt_id
                        if prompt_id not in self.prompt_to_sentences:
                            self.prompt_to_sentences[prompt_id] = []
                        self.prompt_to_sentences[prompt_id].append(sentence_idx)
                    
                    sentences_prompt.append([f"Prompt {prompt_id}: {s}" for s in valid_sentences])
                    prompt_id += 1
            
        self.logger.info(f"Found {len(sentences)} sentences from {len(self.prompt_to_sentences)} prompts in {jsonl_path}")
        
        # Initialize meaning distributions for each prompt
        for prompt_id in self.prompt_to_sentences.keys():
            self.prompt_meaning_distributions[prompt_id] = {}
        
        # Process sentences
        results = []
        end_idx = len(sentences) if max_sentences is None else min(start_from + max_sentences, len(sentences))
        
        for i in range(start_from, end_idx):
            sentence = sentences[i]
            result = self.process_sentence(
                sentence=sentence,
                prompt_id=i,
                callback=callback,
                **kwargs
            )
            results.append(result)
            
            # Update meaning distribution for this prompt
            meaning_id = result.metadata.get("meaning_id")
            if meaning_id is not None:
                prompt_id = self.sentence_to_prompt.get(i)
                if prompt_id is not None:
                    if meaning_id not in self.prompt_meaning_distributions[prompt_id]:
                        self.prompt_meaning_distributions[prompt_id][meaning_id] = 0
                    self.prompt_meaning_distributions[prompt_id][meaning_id] += 1
            
            # Optional: add delay between requests
            time.sleep(0.1)
        
        # Save results if output path provided
        if output_path:
            self.save_results(output_path)
        
        return results
    
    def get_meaning_distributions(self) -> Dict[int, Dict[int, int]]:
        """Get the meaning distribution for each prompt."""
        return self.prompt_meaning_distributions.copy()
    
    def get_meaning_summary(self) -> Dict[str, Any]:
        """Get a summary of all meanings and their usage."""
        summary = {
            "total_meanings": len(self.meanings),
            "meanings": self.meanings.copy(),
            "prompt_distributions": self.prompt_meaning_distributions.copy(),
            "total_sentences_processed": len(self.results)
        }
        
        # Calculate overall meaning frequencies
        overall_frequencies = {}
        for prompt_dist in self.prompt_meaning_distributions.values():
            for meaning_id, count in prompt_dist.items():
                if meaning_id not in overall_frequencies:
                    overall_frequencies[meaning_id] = 0
                overall_frequencies[meaning_id] += count
        
        summary["overall_frequencies"] = overall_frequencies
        return summary
    
    def save_results(self, output_path: str):
        """Save processing results to file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'sentence': result.sentence,
                'prompt_id': result.prompt_id,
                'original_prompt': result.original_prompt,
                'modified_prompt': result.modified_prompt,
                'llm_response': result.llm_response,
                'processing_time': result.processing_time,
                'metadata': result.metadata
            })
        
        # Create comprehensive output
        output_data = {
            'results': serializable_results,
            'meaning_summary': self.get_meaning_summary(),
            'prompt_mapping': {
                'sentence_to_prompt': self.sentence_to_prompt,
                'prompt_to_sentences': self.prompt_to_sentences
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(self.results)} results to {output_path}")
        
        # Also save meaning distributions separately
        distributions_path = output_path.replace('.json', '_distributions.json')
        with open(distributions_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_meaning_summary(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved meaning distributions to {distributions_path}")
    
    def load_results(self, input_path: str):
        """Load processing results from file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.results = []
        for item in data:
            result = ProcessingResult(
                sentence=item['sentence'],
                prompt_id=item['prompt_id'],
                original_prompt=item['original_prompt'],
                modified_prompt=item['modified_prompt'],
                llm_response=item['llm_response'],
                processing_time=item['processing_time'],
                metadata=item['metadata']
            )
            self.results.append(result)
        
        self.logger.info(f"Loaded {len(self.results)} results from {input_path}")


def create_classification_prompt(meanings: List[str]) -> str:
    """Create a classification prompt template."""
    prompt = "You are a semantic classifier.\n\n"
    prompt += "Currently, I have these logical meanings with example sentences or short definitions:\n"
    
    for i, meaning in enumerate(meanings):
        prompt += f"  • Meaning {i}: {meaning}\n"
    
    prompt += "\nNow, I will give you a new sentence:\n\n"
    prompt += '"[Insert new sentence here]"\n\n'
    prompt += "Your task:\n"
    prompt += "  • Assign this sentence to one of the existing logical meanings by number (e.g., 'Meaning 2'), based on its general logical content.\n"
    prompt += "  • If it does not fit any existing meaning, respond with 'Create new meaning'.\n"
    prompt += "  • If the sentence is nonsense or meaningless, respond with 'Meaning 0'.\n"
    prompt += "  • Small typos or minor errors should not prevent you from understanding the general logical meaning. Focus on the overall logical intent.\n\n"
    prompt += "Please answer only with:\n"
    prompt += "  • The meaning number and a brief restatement or label of that meaning, or\n"
    prompt += "  • 'Create new meaning' with a brief restatement or label of that meaning."
    
    return prompt


def create_fixed_classification_prompt(meanings: List[str]) -> str:
    """Create a classification prompt template with fixed predefined meanings."""
    prompt = "You are a semantic classifier.\n\n"
    prompt += "Here is a list of predefined logical meanings:\n"
    
    for i, meaning in enumerate(meanings):
        prompt += f"  • Meaning {i}: {meaning}\n"
    
    prompt += (
        "\nNow, I will give you a new sentence:\n\n"
        '"[Insert new sentence here]"\n\n'
        "Your task:\n"
        "  • Assign this sentence to exactly one of the meanings above, based on its general logical content.\n"
        "  • If the sentence is nonsense or meaningless, choose Meaning 0.\n"
        "  • Small typos or minor errors should not prevent you from understanding the intended meaning.\n"
        "  • You must choose from the meanings listed above — do not invent new categories.\n\n"
        "Please answer only with:\n"
        "  • The meaning number, followed by a brief restatement or label of that meaning.\n"
        # "  • Also write a short explanation of why you chose this meaning."
    )
    
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process sentences one by one using OpenAI client with gradual prompt modification")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", help="Path to save results JSON file")
    parser.add_argument("--api-url", default="http://10.220.5.151:30500/v1", help="OpenAI API base URL")
    parser.add_argument("--api-key", default="", help="API key for authentication")
    parser.add_argument("--model", default="sqz-gpt-oss-120b", help="Model name for API")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--timeout", type=int, default=30, help="API request timeout in seconds")
    parser.add_argument("--start-from", type=int, default=0, help="Start processing from this sentence index")
    parser.add_argument("--max-sentences", type=int, help="Maximum number of sentences to process")
    parser.add_argument("--meanings", nargs="*", 
                       help="List of meaning definitions (defaults to predefined mathematical meanings if --fixed-meanings is used)")
    parser.add_argument("--modifier", choices=["none", "adaptive", "feedback"], default="none",
                       help="Prompt modification strategy")
    parser.add_argument("--fixed-meanings", action="store_true", 
                       help="Use fixed predefined meanings instead of dynamic meaning generation")
    parser.add_argument("--test-api", action="store_true", help="Test API connection and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create processor
    processor = LLMSentenceProcessor(
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        use_fixed_meanings=args.fixed_meanings,
    )
    
    # Test API connection if requested
    if args.test_api:
        print("Testing API connection...")
        if processor.test_api_connection():
            print("✅ API connection successful!")
            return
        else:
            print("❌ API connection failed!")
            return
    
    # Set prompt modifier if requested
    if args.modifier == "adaptive":
        processor.set_prompt_modifier(AdaptivePromptModifier())
    elif args.modifier == "feedback":
        feedback_rules = {
            "error": "This sentence contains an error - classify based on intended meaning",
            "math": "This is a mathematical statement - pay attention to mathematical logic",
        }
        processor.set_prompt_modifier(FeedbackBasedModifier(feedback_rules=feedback_rules))
    
    # Initialize meanings
    if args.fixed_meanings and not args.meanings:
        # Use predefined mathematical meanings if fixed mode is enabled and no meanings provided
        meanings_to_use = PREDEFINED_MATHEMATICAL_MEANINGS
        print(f"Using predefined mathematical meanings: {len(meanings_to_use)} categories")
    else:
        meanings_to_use = args.meanings or ["Nonsense or meaningless statements"]
    
    processor.initialize_meanings(meanings_to_use)
    
    # Log the mode being used
    mode = "fixed predefined meanings" if args.fixed_meanings else "dynamic meaning generation"
    print(f"Running in {mode} mode")
    
    # Define callback for real-time feedback
    def print_result(result: ProcessingResult):
        print(f"\n--- Sentence {result.prompt_id} ---")
        print(f"Input: {result.sentence}")
        print(f"Response: {result.llm_response}")
        print(f"Meaning ID: {result.metadata.get('meaning_id', 'N/A')}")
        print(f"Total Meanings: {result.metadata.get('total_meanings', 'N/A')}")
        print(f"Time: {result.processing_time:.2f}s")
        print("-" * 50)
    
    # Process sentences
    results = processor.process_sentences_from_jsonl(
        jsonl_path=args.input,
        output_path=args.output,
        start_from=args.start_from,
        max_sentences=args.max_sentences,
        callback=print_result,
    )
    
    print(f"\nProcessed {len(results)} sentences successfully!")
    
    # Print final meaning distribution summary
    print("\n" + "="*60)
    print("FINAL MEANING DISTRIBUTION SUMMARY")
    print("="*60)
    
    summary = processor.get_meaning_summary()
    print(f"Total meanings discovered: {summary['total_meanings']}")
    print(f"Total sentences processed: {summary['total_sentences_processed']}")
    
    print("\nMeanings:")
    for i, meaning in enumerate(summary['meanings']):
        count = summary['overall_frequencies'].get(i, 0)
        print(f"  Meaning {i}: {meaning} (used {count} times)")
    
    print("\nPrompt-wise distributions:")
    for prompt_id, distribution in summary['prompt_distributions'].items():
        print(f"\nPrompt {prompt_id}:")
        total_in_prompt = sum(distribution.values())
        for meaning_id, count in sorted(distribution.items()):
            meaning_name = summary['meanings'][meaning_id] if meaning_id < len(summary['meanings']) else f"Unknown Meaning {meaning_id}"
            percentage = (count / total_in_prompt * 100) if total_in_prompt > 0 else 0
            print(f"  - Meaning {meaning_id} ({meaning_name}): {count} sentences ({percentage:.1f}%)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()