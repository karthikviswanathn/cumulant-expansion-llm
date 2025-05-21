import torch
import numpy as np
from tuned_lens.nn.lenses import TunedLens
from typing import List, Dict, Union, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import tempfile

def to_numpy(tensor):
    """
    From neel_plotly repository (https://github.com/neelnanda-io/neel-plotly)
    Convert tensor or list to numpy array.
    
    Args:
        tensor: Input tensor, list, or scalar value.
        
    Returns:
        NumPy array representation.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        try:
            array = np.array(tensor)
            if array.dtype != np.dtype("O"):
                return array
            else:
                # This would need a to_numpy_ragged_2d function implementation
                # Return as-is for now
                return array
        except:
            # Convert each element to numpy
            return [ModelAnalyzer.to_numpy(t) for t in tensor]
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.detach().cpu().numpy()
    elif type(tensor) in [int, float, bool, str]:
        return np.array(tensor)
    else:
        try:
            # Try to handle pandas Series or similar
            return tensor.values
        except:
            raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def shuffle_tokens(ids):
    assert ids.shape[0] == 1 and len(ids.shape) == 2, f"Expected shape (1, N), but got {ids.shape}"
    permutation = np.random.permutation(ids.shape[1])
    return ids[:, permutation]

def simulate_sums(
    centered_delta_logits: torch.Tensor,  # [T, V]
    probs: torch.Tensor,                  # [T, V]
    n_sims: int       = 10_000,           # number of *initial* per-token samples
    bins: int         = 50,
    density: bool     = True,
):
    T, V = probs.shape
    device = probs.device

    # 1) accumulator on device
    sums = torch.zeros(n_sims, device=device)

    # 2) stream over tokens
    for t in range(T):
        # draw n_sims independent samples for token t
        idx_t = torch.multinomial(probs[t], n_sims, replacement=True)  # → [n_sims]
        # gather that token’s deltas
        row = centered_delta_logits[t].gather(0, idx_t)                # → [n_sims]
        # accumulate into our running sums
        sums += row

    # 3) move to CPU & histogram
    sums_np = sums.cpu().numpy()
    hist, bin_edges = np.histogram(sums_np, bins=bins, density=density)
    return sums_np, hist, bin_edges

def calculate_cumulants(
        all_logits: List[torch.Tensor],
        all_probs: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate layer-wise cumulants (2nd through 8th) and related statistics.

        Args:
            all_logits: List of tensors of shape (B, V) for each layer's logits.
            all_probs: List of tensors of shape (B, V) for each layer's probabilities.

        Returns:
            Dictionary with calculated statistics including cumulants, normalized cumulants, 
            entropies, and KL divergences.
        """
        # Disable gradients for cumulant calculation
        with torch.no_grad():
            # Compute combined probabilities and centered logits per layer
            probs_com_list = []
            logits_com_list = []
            for logits, probs in zip(all_logits, all_probs):
                probs = probs + 1e-12  # avoid zeros
                probs_com = probs.mean(dim=-2).squeeze()
                logits_com = torch.log(probs_com)
                logits_com -= (probs_com * logits_com).sum()
                probs_com_list.append(probs_com)
                logits_com_list.append(logits_com)

            all_probs_com = torch.stack(probs_com_list, dim=0)
            all_logits_com = torch.stack(logits_com_list, dim=0)

            # Cumulant formulas based on raw moments up to 8th order
            cumulant_formulas = {
                2: lambda m: m[2],
                3: lambda m: m[3],
                4: lambda m: m[4] - 3 * m[2] ** 2,
                5: lambda m: m[5] - 10 * m[2] * m[3],
                6: lambda m: m[6] - 15 * m[2] * m[4] - 10 * m[3] ** 2 + 30 * m[2] ** 3,
                7: lambda m: m[7] - 21 * m[2] * m[5] - 35 * m[3] * m[4] + 210 * m[2] ** 2 * m[3],
                8: lambda m: (
                    m[8]
                    - 28 * m[2] * m[6]
                    - 56 * m[3] * m[5]
                    + 420 * m[2] ** 2 * m[4]
                    - 35 * m[4] ** 2
                    + 560 * m[2] * m[3] ** 2
                    - 630 * m[2] ** 4
                ),
            }
            
            # Calculate statistics for each layer
            all_cumulants, all_normalized_cumulants, all_entropies, all_kld = [], [], [], []
            for idx, (logits_layer, probs_layer) in enumerate(zip(all_logits, all_probs)):
                # difference from combined logits
                delta = all_logits_com[idx] - logits_layer
                delta_centered = delta - (probs_layer * delta).sum(dim=-1, keepdim=True)

                # raw moments m1..m8
                moments = {k: (probs_layer * delta_centered.pow(k)).sum(dim=-1) for k in range(1, 9)}

                # compute cumulants k2..k8 as mean over batch
                cumulants = torch.stack([
                    cumulant_formulas[k](moments) for k in range(2, 9)
                ])
                all_cumulants.append(cumulants)
                
                # Calculate normalized cumulants
                normalization_factors = torch.tensor(
                    [[2, 6, 24, 120, 720, 5040, 40320]], 
                    device=cumulants.device
                ).T
                all_normalized_cumulants.append(cumulants / normalization_factors)
                
                # Calculate entropy
                all_entropies.append(-(probs_layer * torch.log(probs_layer + 1e-12)).sum(axis=-1))
                
                # Calculate KL divergence
                kld = (probs_layer * (torch.log(probs_layer + 1e-12) - 
                                    torch.log(all_probs_com[idx] + 1e-12))).sum(dim=-1)
                all_kld.append(kld)
                
            # Calculate combined entropy
            entropy_com = -torch.sum(all_probs_com * torch.log(all_probs_com + 1e-12), dim=-1)
            
            return {
                'entropy': torch.stack(all_entropies),
                'entropy_com': entropy_com,
                'cumulants': torch.stack(all_cumulants),
                'normalized_cumulants': torch.stack(all_normalized_cumulants),
                'kld_center': torch.stack(all_kld, dim=0),
                'avg_entropy': torch.stack(all_entropies).mean(axis = -1),
                'avg_normalized_cumulants': torch.stack(all_normalized_cumulants).mean(axis = -1),
                'avg_cumulants': torch.stack(all_cumulants).mean(axis = -1),
            }
    


class CumulantAnalyzer:
    """
    A class for analyzing language models using TunedLens.
    
    This class provides functionality to:
    - Load a pretrained causal language model and its tuned lens
    - Generate logits and probabilities for each layer
    - Calculate layer-wise cumulants and other statistics
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        max_length: int = 256,
        load_lens: bool = True,
        revision: str = None
    ):
        """
        Initialize the ModelAnalyzer with a specified model.
        
        Args:
            model_name: Hugging Face model identifier.
            device: Device string for PyTorch ('cuda' or 'cpu').
            max_length: Maximum sequence length for tokenization.
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        
        # Load model components
        self.model, self.tokenizer = self._load_model(revision)

        # Load lens
        if load_lens:
            self.tuned_lens = self._load_lens()
        
    def _load_model(self, revision: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a causal LM and its tokenizer.
        
        Args:
            revision: Optional git revision/branch/tag to use when loading from HuggingFace Hub.
        
        Returns:
            Tuple containing the model and tokenizer.
        """
        # Load model and tokenizer
        self._temp_dir = tempfile.TemporaryDirectory()
        
        # Include revision parameter if provided
        model_kwargs = {
            'device_map': 'auto',
            'cache_dir': self._temp_dir.name,
        }
        tokenizer_kwargs = {
            'cache_dir': self._temp_dir.name,
        }
        
        # Add revision to kwargs if specified
        if revision is not None:
            model_kwargs['revision'] = revision
            tokenizer_kwargs['revision'] = revision
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        
        # Note: your return statement included tuned_lens, but that's not defined in this method
        # If you want to return tuned_lens, it should be loaded or defined within this method
        return model, tokenizer
        
    def _load_lens(self) -> TunedLens:
        """
        Load tuned lens.
        
        Returns:
            Tuned lens.
        """
        
        # Load TunedLens and move to device
        tuned_lens = TunedLens.from_model_and_pretrained(self.model, cache_dir=self._temp_dir.name)
        tuned_lens = tuned_lens.to(self.device)

        return tuned_lens
    
    def get_logits(
        self,
        test_sequence: str,
        shuffled: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate logits and probabilities for each layer using the tuned lens.

        Args:
            test_sequence: Input text string.
            shuffled: If True, shuffle tokens; otherwise, keep order.

        Returns:
            all_logits: List of layer-wise logit tensors.
            all_probs: List of layer-wise probability tensors.
        """
        # Encode tokens
        input_ids = self.tokenizer(
            test_sequence.strip(),
            return_tensors='pt',
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )['input_ids']

        # Optionally shuffle
        if shuffled:
            input_ids = shuffle_tokens(input_ids)
        input_ids = input_ids.to(self.device)

        all_logits: List[torch.Tensor] = []
        all_probs: List[torch.Tensor] = []

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Lens predictions for each hidden state
            for idx, h in enumerate(hidden_states[:-1]):
                logits = self.tuned_lens(h, idx).squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                all_logits.append(logits)
                all_probs.append(probs)

            # Final model logits
            final_logits = outputs.logits.squeeze(0)
            all_logits.append(final_logits)
            all_probs.append(torch.softmax(final_logits, dim=-1))

        return all_logits, all_probs

    def compute_stats(
        self,
        sequence: str,
        shuffled: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Compute all statistics for a given text sequence.
        
        Args:
            sequence: Text sequence to analyze.
            shuffled: Whether to shuffle the tokens.
            
        Returns:
            Dictionary of NumPy arrays with computed statistics.
        """
        # Get logits and probabilities
        all_logits, all_probs = self.get_logits(sequence.strip(), shuffled=shuffled)
        
        # Calculate statistics
        stats = calculate_cumulants(all_logits, all_probs)
        
        # Convert all tensors to numpy arrays
        return {key: to_numpy(tensor) for key, tensor in stats.items()}

    def do_monte_carlo(self, test_sequence, shuffled = False, LNUM = 20, n_sims = 10000000, n_bins = 1000):
        all_logits, all_probs = self.get_logits(test_sequence, shuffled = shuffled)
        logits, probs = all_logits[LNUM], all_probs[LNUM]
        probs_com = torch.mean(probs, axis = 0)
        logits_com_mean = torch.sum(probs_com * torch.log(probs_com))
        logits_com_centered = (torch.log(probs_com) - logits_com_mean).unsqueeze(0) 
        delta_logits = logits_com_centered - logits # -\delta X
        centered_delta_logits = delta_logits - (probs * delta_logits).sum(axis = 1, keepdims = True) # Physicists like to call this "gauge fixing".
        sums, hist, edges = simulate_sums(centered_delta_logits,
                                          probs,
                                          n_sims=n_sims,
                                          bins=n_bins,
                                          density=True)
        # p_dist = np.histogram()
        mean = sums.mean()
        m2_emp = ((sums-mean)**2).mean()
        m3_emp = ((sums-mean)**3).mean()
        m4_emp = ((sums-mean)**4).mean()

        max_length = probs.shape[0]
        # 2) form cumulants
        k1_emp = mean
        k2_emp = m2_emp/max_length/2
        k3_emp = m3_emp/max_length/6
        k4_emp = (m4_emp - 3 * m2_emp**2)/max_length/24
    
        
        centers = 0.5*(edges[:-1] + edges[1:])  # (B,)
        # plt.figure()
        is_shuffled = "shuffled" if shuffled else "structured"    
        model_chi = stats = self.compute_stats(test_sequence, shuffled = shuffled)['avg_normalized_cumulants'][LNUM]
        # Make DataFrame with comparison
        df = pd.DataFrame([{
                # "test_indx": test_indx,
                # "shuffle_indx": shuffle_indx,
                "type": is_shuffled,
                r"$\kappa_2$ (monte carlo)": k2_emp,
                r"$\kappa_2$ (avg of tokens)": model_chi[0],
                r"$\kappa_3$ (monte carlo)": k3_emp,
                r"$\kappa_3$ (avg of tokens)": model_chi[1],
                r"$\kappa_4$ (monte carlo)": k4_emp,
                r"$\kappa_4$ (avg of tokens)": model_chi[2]
            }])
        return {'df': df, 'centers': centers, 'hist': hist}
