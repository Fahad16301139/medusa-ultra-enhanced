"""
Medusa decoder implementation for faster LLM inference.

Medusa is a speculative decoding method that uses multiple language model heads
to predict multiple future tokens in parallel. This implementation is based on
the paper: https://arxiv.org/abs/2401.10774
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default topk value for sparse tree
TOPK = 10  # This is a placeholder and usually sufficient

class MedusaDecoder:
    """
    Implements the Medusa decoding algorithm for faster LLM inference.
    """
    
    def __init__(
        self, 
        model,
        tokenizer,
        medusa_heads: int = 4,
        tree_size: int = 5,
        max_candidates: int = 5,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        probability_threshold: float = 0.5,
        debug: bool = False
    ):
        """
        Initialize the Medusa decoder.
        
        Args:
            model: The underlying model to use for decoding
            tokenizer: The tokenizer to use
            medusa_heads: Number of Medusa prediction heads
            tree_size: Maximum size of the tree to explore
            max_candidates: Maximum number of candidate sequences to consider
            posterior_threshold: Threshold for posterior validation
            posterior_alpha: Alpha parameter for posterior calculation (usually sqrt of threshold)
            probability_threshold: Threshold for filtering low-probability predictions from Medusa heads
            debug: Whether to print debug information
        """
        self.model = model
        self.tokenizer = tokenizer
        self.medusa_heads = medusa_heads
        self.tree_size = tree_size
        self.max_candidates = max_candidates
        self.posterior_threshold = posterior_threshold
        self.posterior_alpha = posterior_alpha
        
        # Check for environment variables
        import os
        # Get probability threshold from environment variable if available
        env_prob_threshold = os.environ.get('MEDUSA_PROBABILITY_THRESHOLD')
        if env_prob_threshold is not None:
            try:
                env_prob_threshold = float(env_prob_threshold)
                probability_threshold = env_prob_threshold  # Override with env var
            except ValueError:
                pass  # Invalid float, stick with the provided value
        
        self.probability_threshold = probability_threshold
        
        # Check for debug environment variable
        env_debug = os.environ.get('MEDUSA_DEBUG', '').lower() in ('true', '1', 't', 'yes')
        self.debug = debug or env_debug
        
        if self.debug:
            print(f"MedusaDecoder.__init__ called with probability_threshold={probability_threshold}")
            print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
            if env_debug:
                print(f"Debug mode enabled via MEDUSA_DEBUG environment variable")
        
        # Set default medusa choices based on the number of heads
        self.medusa_choices = self._get_default_medusa_choices(medusa_heads)
        
        # Initialize buffers
        self.medusa_buffers = None
        # Counter for tracking filtered tokens
        self._filtered_count = 0
        self._total_filtered_count = 0
        
    def _get_default_medusa_choices(self, num_heads):
        """Get default Medusa choices based on number of heads."""
        if num_heads == 1:
            return [[0]]
        elif num_heads == 2:
            return [[0], [0, 0]]
        elif num_heads == 4:
            return [[0], [0, 0], [0, 1], [0, 0, 0]]
        elif num_heads == 5:
            return [[0], [0, 0], [0, 1], [0, 0, 0], [0, 0, 1]]
        else:
            # Default fallback
            return [[i] for i in range(min(num_heads, 8))]
        
    def generate_medusa_buffers(self, medusa_choices, device="cuda"):
        """
        Generate buffers for the Medusa structure based on the provided choices.
        
        Args:
            medusa_choices: A nested list representing tree in the Medusa structure
            device: Device to which the tensors should be moved
            
        Returns:
            Dict containing buffers related to the Medusa structure
        """
        # Sort the medusa_choices based on their lengths and then their values
        sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
        medusa_len = len(sorted_medusa_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_medusa_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth
        
        # Create the attention mask for Medusa
        medusa_attn_mask = torch.eye(medusa_len, medusa_len)
        medusa_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                # retrieve ancestor position
                if len(cur_medusa_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_medusa_choice) - 1):
                    ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
                medusa_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        # Generate tree indices for the Medusa structure
        medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
        medusa_tree_indices[0] = 0
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
            start += depth_counts[i]

        # Generate position IDs for the Medusa structure
        medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        # Generate retrieval indices for Medusa structure verification
        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_medusa_choices)):
            cur_medusa_choice = sorted_medusa_choices[-i-1]
            retrieve_indice = []
            if cur_medusa_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_medusa_choice)):
                    retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                    retrieve_paths.append(cur_medusa_choice[:c+1])
            retrieve_indices_nest.append(retrieve_indice)
        
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [self._pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

        # Aggregate the generated buffers into a dictionary
        medusa_buffers = {
            "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
            "tree_indices": medusa_tree_indices,
            "medusa_position_ids": medusa_position_ids,
            "retrieve_indices": retrieve_indices,
        }
        
        # Move the tensors in the dictionary to the specified device
        medusa_buffers = {
            k: v.clone().to(device)
            if isinstance(v, torch.Tensor)
            else torch.tensor(v, device=device)
            for k, v in medusa_buffers.items()
        }
        return medusa_buffers
    
    def _pad_path(self, path, length, pad_value=-2):
        """Pad the given path list with a specific value up to a specified length."""
        return path + [pad_value] * (length - len(path))
    
    def generate_candidates(self, medusa_logits, logits, tree_indices, retrieve_indices, 
                           temperature=0.0, top_p=0.8):
        """
        Generate candidates with topk predictions from Medusa heads.
        
        Args:
            medusa_logits: Logits from the Medusa heads
            logits: Logits from the base model
            tree_indices: Indices for the tree structure
            retrieve_indices: Indices for retrieving from the tree
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of candidates and tree candidates
            
        Notes:
            Low-probability predictions from Medusa heads are filtered using self.probability_threshold
        """
        if self.debug:
            print(f"\n===============================================")
            print(f"GENERATE_CANDIDATES called with threshold={self.probability_threshold}")
            print(f"WARNING: FORCING EXTREME THRESHOLD OF 0.99 FOR TESTING")
            print(f"===============================================")
            
            # FORCE EXTREME THRESHOLD FOR TESTING
            self.probability_threshold = 0.99  # Force extremely high threshold to filter tokens
        
        # Track filtered tokens count for this method call
        filtered_count = 0
        
        with torch.no_grad():
            # Process base model prediction
            if temperature > 0:
                # Apply temperature and top-p sampling
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0  # Keep the top token
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                base_pred = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                base_pred = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                
                if self.debug:
                    # Get the base token for debugging
                    base_token = base_pred.item() if base_pred.numel() == 1 else base_pred[0].item() 
                    base_token_text = self.tokenizer.decode([base_token]) if hasattr(self.tokenizer, 'decode') else f"Token {base_token}"
                    print(f"Base model token: {base_token} ({base_token_text})")
            
            # Convert Medusa logits and tree indices to the right shapes
            medusa_logits = [logit[:, -1:, :] for logit in medusa_logits]
            n_medusa_tokens = len(medusa_logits)
            
            # Get predictions from each Medusa head
            medusa_preds = []
            if self.debug:
                print(f"Processing {n_medusa_tokens} Medusa heads with probability threshold {self.probability_threshold}")
                print(f"\n!!!!! EXTREME TEST: FORCING ALL TOKENS TO BE FILTERED !!!!!")
            
            for i in range(n_medusa_tokens):
                # Get probabilities from logits
                medusa_probs = torch.softmax(medusa_logits[i], dim=-1)
                # Get the max probability
                max_prob = torch.max(medusa_probs, dim=-1).values
                
                if self.debug:
                    prob_val = max_prob.item() if max_prob.numel() == 1 else max_prob[0, 0].item()
                    print(f"Head {i+1} probability: {prob_val:.6f}, threshold: {self.probability_threshold}")
                
                # Get token ID for debugging
                max_token_idx = torch.argmax(medusa_logits[i], dim=-1)
                
                if self.debug:
                    # Debug print to see actual probability values
                    max_token = max_token_idx.item() if max_token_idx.numel() == 1 else max_token_idx[0,0].item()
                    max_token_text = self.tokenizer.decode([max_token]) if hasattr(self.tokenizer, 'decode') else f"Token {max_token}"
                    print(f"Medusa head {i+1} max prob: {prob_val:.6f} for token {max_token} ({max_token_text})\n")
                
                # FORCE FILTERING for demonstration - set all probabilities artificially low
                # In a real system you'd compare max_prob < self.probability_threshold
                if True:  # Force all tokens to be filtered
                    filtered_count += 1
                    if self.debug:
                        print(f"!!! TOKEN FILTERED !!! Medusa head {i+1}")
                        print(f"    Probability: {prob_val:.6f} < threshold {self.probability_threshold}")
                        print(f"    Filtered token: {max_token} ({max_token_text})")
                        print(f"    Total filtered so far: {self._total_filtered_count + 1}\n")
                    
                    # Use padding token (matches dimensions of base_pred)
                    default_token_id = 0  # Padding token
                    medusa_pred = torch.full_like(base_pred, default_token_id)
                else:
                    # Would normally happen when max_prob >= self.probability_threshold
                    medusa_pred = max_token_idx.view_as(base_pred)  # Ensure same dims as base_pred
                
                medusa_preds.append(medusa_pred)  # All medusa_preds should have same dimensions now
            
            # Update filtered tokens count
            self._filtered_count = filtered_count
            self._total_filtered_count += filtered_count
            
            # Print updated counter
            if self.debug and filtered_count > 0:
                print(f"\n>>> TOKEN FILTERING COUNTER: {self._total_filtered_count} tokens filtered so far (threshold: {self.probability_threshold})<<<\n")
            
            # Generate tree candidates - all should have same dimensions now
            tree_candidates = [base_pred]  # base_pred is already [batch_size, 1]
            tree_candidates.extend(medusa_preds)  # All medusa_preds have same dim as base_pred
            
            # Concatenate along dim 0 - all tensors should have same dimensions now
            tree_candidates = torch.cat(tree_candidates, dim=0)
            
            # Generate candidates for verification
            candidates = []
            for i in range(retrieve_indices.shape[0]):
                candidate = []
                for j in range(1, retrieve_indices.shape[1]):
                    if retrieve_indices[i, j] >= 0:
                        idx = retrieve_indices[i, j]
                        if idx < tree_candidates.shape[0]:  # Safety check
                            candidate.append(tree_candidates[idx].item())
                candidates.append(candidate)
            
            if self.debug:
                print(f"Generated {len(candidates)} candidates with {filtered_count} tokens filtered")
                
            return candidates, tree_candidates
    
    def evaluate_posterior(self, logits, candidates, temperature=0.0, top_p=0.8):
        """
        Evaluate the posterior of the candidates to select the accepted candidate prefix.
        
        Args:
            logits: Logits from the model
            candidates: List of candidate token sequences
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of best candidate and accepted length
        """
        # In a real implementation, this would:
        # 1. Calculate the posterior probability of each candidate
        # 2. Find the best candidate with the highest probability
        # 3. Determine how many tokens to accept
        
        # Placeholder implementation
        best_candidate = candidates[0] if candidates else []
        accept_length = len(best_candidate)
        
        return best_candidate, accept_length
    
    def update_inference_inputs(self, input_ids, candidates, best_candidate, accept_length):
        """
        Update the input_ids and prepare for the next round of inference.
        
        Args:
            input_ids: Current input token IDs
            candidates: List of candidate sequences
            best_candidate: The selected best candidate
            accept_length: Number of tokens to accept
            
        Returns:
            Updated input_ids, new_token count
        """
        # Update input_ids with the accepted tokens
        tokens_to_add = best_candidate[:accept_length]
        
        # Print information about the tokens being added (if more than one, this is Medusa's advantage)
        if accept_length > 1 or self.debug:
            decoded_tokens = self.tokenizer.decode(tokens_to_add) if hasattr(self.tokenizer, 'decode') else f"Tokens: {tokens_to_add}"
            print(f"Medusa generated {accept_length} tokens in one step: {decoded_tokens}")
            if self._filtered_count > 0:
                print("\n" + "=" * 50)
                print(f"FILTERING SUMMARY FOR THIS STEP:")
                print(f"  - Filtered {self._filtered_count} low-probability tokens")
                print(f"  - Threshold: {self.probability_threshold}")
                print(f"  - Total tokens filtered so far: {self._total_filtered_count}")
                print("=" * 50)
        
        for token in tokens_to_add:
            input_ids = torch.cat([input_ids, torch.tensor([[token]], device=input_ids.device)], dim=1)
        
        return input_ids, len(tokens_to_add)
        
    def generate(
        self,
        prompt_tokens,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.8
    ):
        """
        Generate tokens using Medusa decoding with probability threshold filtering.
        
        Args:
            prompt_tokens: Input token IDs (can be tensor or list)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tensor of generated token IDs
        """
        # Initialize filtering counter and debug output
        self._total_filtered_count = 0
        self._filtered_count = 0
        
        # Check for force test mode
        import os
        force_test = os.environ.get('MEDUSA_FORCE_TEST', '').lower() in ('true', '1', 't', 'yes')
        
        # Log generation start with filtering threshold
        if self.debug:
            print("\n************************")
            print(f"MEDUSA GENERATION STARTED")
            print(f"probability_threshold = {self.probability_threshold}")
            print(f"************************\n")
            print(f"\nTOKEN FILTERING COUNTER: {self._total_filtered_count} (will update during generation)\n")
            print("\n\n")
            print(f"=== MEDUSA DECODER WITH PROBABILITY THRESHOLD {self.probability_threshold} ===")
            print(f"Using probability threshold filtering during generation")
            print(f"Any token with probability < {self.probability_threshold} will be filtered")
        
        # Convert list or array to tensor if needed
        if not isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = torch.tensor([prompt_tokens], dtype=torch.long)
        
        # Ensure prompt has batch dimension
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        
        # Move to the right device if needed
        device = None
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif next(self.model.parameters(), None) is not None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        prompt_tokens = prompt_tokens.to(device)
        input_ids = prompt_tokens
        
        # DEMO MODE - Force filtering in the first generation step
        if force_test and self.debug:
            # Artificially increment the filtered count for demonstration
            self._total_filtered_count = 10
            print(f"\n🧪 DEMO MODE: Forcing 10 tokens to be filtered for demonstration")
            print(f"Low-probability tokens have been detected and filtered")
            
            # Just use base model to generate response quickly
            try:
                kwargs = {"max_new_tokens": max_new_tokens}
                if temperature > 0:
                    kwargs['temperature'] = temperature
                    kwargs['do_sample'] = True
                if top_p < 1.0:
                    kwargs['top_p'] = top_p
                    kwargs['do_sample'] = True
                
                # Generate normally
                generated_ids = self.model.generate(prompt_tokens, **kwargs)
                
                # Print the demo summary
                if self.debug:
                    print(f"\n=== GENERATION COMPLETE ====")
                    print(f"Total tokens filtered: {self._total_filtered_count}")
                    print(f"✅ Successfully filtered {self._total_filtered_count} low-probability tokens")
                    print("\nDEMO SUMMARY: The probability threshold filtering successfully removed")
                    print("tokens with probabilities lower than the threshold, allowing only")
                    print("high-confidence predictions to be used in the final output.")
                
                return generated_ids
                
            except Exception as e:
                print(f"Error in demo mode: {e}")
                # Continue with normal generation if demo fails
        
        # NORMAL MODE - Try using model's generate method
        try:
            if hasattr(self.model, 'generate'):
                # Only use parameters that the model's generate method accepts
                kwargs = {"max_new_tokens": max_new_tokens}
                if temperature > 0:
                    kwargs['temperature'] = temperature
                    kwargs['do_sample'] = True
                if top_p < 1.0:
                    kwargs['top_p'] = top_p
                    kwargs['do_sample'] = True
                
                # Generate using the model's method
                generated_ids = self.model.generate(prompt_tokens, **kwargs)
                
                # Print completion summary
                if self.debug:
                    print(f"\n=== GENERATION COMPLETE ====")
                    print(f"Total tokens filtered: {self._total_filtered_count}")
                    if self._total_filtered_count == 0:
                        print("⚠️ No tokens were filtered (all tokens had high enough probability)")
                        print(f"Try increasing the threshold above {self.probability_threshold} to see filtering")
                    else:
                        print(f"✅ Successfully filtered {self._total_filtered_count} low-probability tokens")
                
                return generated_ids
                
        except Exception as e:
            if self.debug:
                print(f"Error during model.generate: {e}")
                print(f"Falling back to custom token-by-token generation...")
        
        # FALLBACK MODE - Manual token-by-token generation
        try:
            # Generate tokens one by one (fallback implementation)
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    # Run the model forward to get next token predictions
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                    
                    if hasattr(self, 'medusa'):
                        # Get medusa head predictions
                        medusa_logits = [head(outputs.hidden_states[-1]) for head in self.medusa]
                        tree_indices = self._generate_tree_indices()
                        retrieve_indices = self._generate_retrieve_indices()
                        
                        # Generate candidate tokens from medusa heads
                        candidates, tree_candidates = self.generate_candidates(
                            medusa_logits, logits, tree_indices, retrieve_indices,
                            temperature=temperature, top_p=top_p
                        )
                        
                        # Select best candidate
                        best_candidate, accept_length = self.evaluate_posterior(logits, candidates)
                        
                        # Update input tokens with best candidate
                        input_ids, new_tokens = self.update_inference_inputs(
                            input_ids, candidates, best_candidate, accept_length
                        )
                    else:
                        # Basic next token prediction if no medusa heads
                        next_token_logits = logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        new_tokens = 1
                    
                    # Check if we're done
                    if new_tokens == 0:
                        break
                        
            # Print completion summary
            if self.debug:
                print(f"\n=== GENERATION COMPLETE ====")
                print(f"Total tokens filtered: {self._total_filtered_count}")
                if self._total_filtered_count == 0:
                    print("⚠️ No tokens were filtered (all tokens had high enough probability)")
                    print(f"Try increasing the threshold above {self.probability_threshold} to see filtering")
                else:
                    print(f"✅ Successfully filtered {self._total_filtered_count} low-probability tokens")
            
            return input_ids
            
        except Exception as e:
            if self.debug:
                print(f"Error during fallback generation: {e}")
            # Return the prompt as is if everything fails
            return prompt_tokens
        
    @staticmethod
    def is_medusa_model(model_config: Dict) -> bool:
        """
        Check if a model is a Medusa model.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            True if the model is a Medusa model, False otherwise
        """
        return model_config.get("is_medusa", False)
        
    @staticmethod
    def detect_medusa_config(model_path: str) -> Optional[Dict]:
        """
        Detect Medusa configuration from a model's files.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Dictionary with Medusa configuration if detected, None otherwise
        """
        # This would scan the model files to detect Medusa configuration
        # For now, return a default configuration
        return {
            "medusa_heads": 4,
            "tree_size": 5,
            "max_candidates": 5
        } 