import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any
from scipy.spatial.distance import jensenshannon
from enum import Enum, auto
from dataclasses import dataclass, field

# ===================================================================
# ENUMS AND DATA STRUCTURES
# ===================================================================

class VizMode(Enum):
    """Enum for visualization modes to avoid using brittle 'magic strings'."""
    TOKEN_HIGHLIGHT = auto()
    TOKEN_WEIGHTS = auto()
    LAYER_MATRIX = auto()
    LAYER_DISTRIBUTION = auto()
    SIMILARITY_PER_LAYER = auto()
    SIMILARITY_OVERALL = auto()

@dataclass
class RoutingData:
    """A structured container for raw data from a single forward pass."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    top_k_indices: List[torch.Tensor]
    top_k_logits: List[torch.Tensor]

@dataclass
class AnalysisResult:
    """A structured container for the results of an analysis run."""
    name: str
    per_layer_dist: Dict[int, np.ndarray]
    overall_dist: np.ndarray
    raw_data: List[RoutingData] = field(repr=False)

# ===================================================================
# MAIN ANALYZER CLASS
# ===================================================================

class RoutingAnalyzer:
    """
    Analyzes and visualizes routing decisions from Mixture-of-Experts models.
    Follows a log -> analyze -> visualize workflow.
    """
    _COLORS = {
        0: '\033[95m', 1: '\033[94m', 2: '\033[96m', 3: '\033[92m', 4: '\033[93m',
        5: '\033[91m', 6: '\033[35m', 7: '\033[34m', 8: '\033[36m', 9: '\033[32m',
        'ENDC': '\033[0m', 'BOLD': '\033[1m'
    }

    def __init__(self):
        self._raw_logs: List[RoutingData] = []
        self._current_pass_indices: List[torch.Tensor] = [] 
        self._current_pass_logits: List[torch.Tensor] = []
        print("✅ RoutingAnalyzer initialized.")

    # ===================================================================
    # STEP 1: LOGGING
    # ===================================================================

    def log_forward_pass(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self._current_pass_indices:
            last_input_ids, last_attn_mask = self._last_inputs
            self._raw_logs.append(RoutingData(
                input_ids=last_input_ids,
                attention_mask=last_attn_mask,
                top_k_indices=self._current_pass_indices,
                top_k_logits=self._current_pass_logits
            ))
        self._current_pass_indices = []
        self._current_pass_logits = []
        self._last_inputs = (input_ids.detach().cpu(), attention_mask.detach().cpu())

    def log_routing_decision(self, top_k_logits: torch.Tensor, top_k_indices: torch.Tensor):
        self._current_pass_indices.append(top_k_indices.detach().cpu())
        self._current_pass_logits.append(top_k_logits.detach().cpu())

    def clear(self):
        self._raw_logs = []
        self._current_pass_indices = []
        self._current_pass_logits = []
        print("🧹 Analyzer logs cleared.")

    # ===================================================================
    # STEP 2: ANALYSIS
    # ===================================================================

    def analyze(self, name: str, temperature: float = 1.0) -> AnalysisResult:
        """
        Processes logged data to compute expert distributions using the correct
        top-k reconstruction logic.

        Args:
            name: A unique name for this analysis run.
            temperature: The router temperature to use for the softmax calculation.
        """
        if self._current_pass_indices:
            self.log_forward_pass(torch.empty(0), torch.empty(0))
        if not self._raw_logs:
            raise ValueError("Cannot analyze without logged data.")

        # Dynamically determine num_experts by finding the highest index seen
        max_index = 0
        for log in self._raw_logs:
            for indices in log.top_k_indices:
                max_index = max(max_index, indices.max().item())
        num_experts = max_index + 1
        
        num_layers = len(self._raw_logs[0].top_k_logits)
        per_layer_prob_sum = {layer: torch.zeros(num_experts) for layer in range(num_layers)}

        for layer in range(num_layers):
            for log in self._raw_logs:
                if layer < len(log.top_k_logits):
                    top_k_logits = log.top_k_logits[layer]      # Shape is (t, k)
                    top_k_indices = log.top_k_indices[layer]    # Shape is (t, k)
                    mask = log.attention_mask.bool()            # Shape is (B, S)
                    batch_size, seq_len = mask.shape


                    full_score = torch.full((top_k_logits.shape[0], num_experts), float("-inf"))
                    full_score.scatter_(1, top_k_indices, top_k_logits)
                    probabilities_flat = torch.softmax(full_score / temperature, dim=-1)
                    
                    probabilities = probabilities_flat.view(batch_size, seq_len, num_experts)

                    masked_probabilities = probabilities * mask.unsqueeze(-1)
                    pass_prob_sum = torch.sum(masked_probabilities, dim=(0, 1))
                    per_layer_prob_sum[layer] += pass_prob_sum

        per_layer_dist = {
            layer: (probs / (probs.sum() + 1e-9)).numpy()
            for layer, probs in per_layer_prob_sum.items()
        }
        overall_prob_sum = sum(per_layer_prob_sum.values())
        overall_dist = (overall_prob_sum / (overall_prob_sum.sum() + 1e-9)).numpy()

        return AnalysisResult(name=name, per_layer_dist=per_layer_dist, overall_dist=overall_dist, raw_data=list(self._raw_logs))
    
    # ===================================================================
    # STEP 3: VISUALIZATION (STATIC METHODS)
    # ===================================================================

    @staticmethod
    def visualize(mode: VizMode, *, results: Dict[str, AnalysisResult], save_path: str = "my_analysis", **kwargs):
        print(f"\n--- Visualizing Mode: {mode.name} ---")
        os.makedirs(save_path, exist_ok=True)
        
        if not results:
            print("ERROR: No results provided for visualization.")
            return

        if mode in (VizMode.SIMILARITY_PER_LAYER, VizMode.SIMILARITY_OVERALL, VizMode.LAYER_DISTRIBUTION):
            dispatch_map = {
                VizMode.SIMILARITY_PER_LAYER: RoutingAnalyzer._visualize_similarity,
                VizMode.SIMILARITY_OVERALL: RoutingAnalyzer._visualize_similarity,
                VizMode.LAYER_DISTRIBUTION: RoutingAnalyzer._visualize_layer_distribution,
            }
            method = dispatch_map.get(mode)
            if method:
                method(results=results, save_path=save_path, mode=mode, **kwargs)
        else:
            first_result = next(iter(results.values()))
            dispatch_map = {
                VizMode.TOKEN_HIGHLIGHT: RoutingAnalyzer._visualize_token_highlight,
                VizMode.TOKEN_WEIGHTS: RoutingAnalyzer._visualize_token_weights,
                VizMode.LAYER_MATRIX: RoutingAnalyzer._visualize_layer_matrix,
                VizMode.LAYER_DISTRIBUTION: RoutingAnalyzer._visualize_layer_distribution,
            }
            method = dispatch_map.get(mode)
            if method:
                method(first_result, save_path=save_path, **kwargs)
            else:
                print(f"ERROR: Visualization mode '{mode.name}' not implemented or invalid.")

    # --- PLOTTING HELPERS ---

    @staticmethod
    def _visualize_similarity(mode: VizMode, results: Dict[str, AnalysisResult], save_path: str, layer_idx: int = 0):
        run_names = sorted(results.keys())
        matrix = np.zeros((len(run_names), len(run_names)))
        for i, n1 in enumerate(run_names):
            for j, n2 in enumerate(run_names):
                dist1 = results[n1].per_layer_dist.get(layer_idx) if mode == VizMode.SIMILARITY_PER_LAYER else results[n1].overall_dist
                dist2 = results[n2].per_layer_dist.get(layer_idx) if mode == VizMode.SIMILARITY_PER_LAYER else results[n2].overall_dist
                if dist1 is None or dist2 is None: matrix[i, j] = np.nan
                else: matrix[i, j] = 1 - jensenshannon(dist1, dist2, base=2)
        
        title = f"Similarity for Layer {layer_idx}" if mode == VizMode.SIMILARITY_PER_LAYER else "Overall Similarity"
        fname = f"{save_path}/similarity_layer_{layer_idx}.png" if mode == VizMode.SIMILARITY_PER_LAYER else f"{save_path}/similarity_overall.png"
        RoutingAnalyzer._plot_heatmap(matrix, run_names, title, fname)

    @staticmethod
    def _visualize_layer_distribution(results: Dict[str, AnalysisResult], save_path: str, **kwargs):
        """
        Generates a grid of plots showing the expert distribution for each layer.
        Each plot compares the distributions from all provided analysis runs.
        """
        import pandas as pd
        plot_data = []
        for run_name, result in results.items():
            for layer, dist in result.per_layer_dist.items():
                for expert_id, prob in enumerate(dist):
                    plot_data.append({
                        "run_name": run_name,
                        "layer": layer,
                        "expert_id": expert_id,
                        "probability": prob,
                    })
        
        if not plot_data:
            print("No distribution data found to plot.")
            return

        df = pd.DataFrame(plot_data)
        num_layers = df['layer'].nunique()


        FIG_SIZE = (32, 28)

        TITLE_FONTSIZE = 30
        LABEL_FONTSIZE = 24
        TICK_FONTSIZE = 21
        LEGEND_FONTSIZE = 24
        
        GRID_ROWS, GRID_COLS = 8, 8

        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=FIG_SIZE, sharex=True, sharey=True)
        axes = axes.flatten()

        for layer in range(num_layers):
            ax = axes[layer]
            layer_df = df[df['layer'] == layer]
            
            sns.barplot(data=layer_df, x='expert_id', y='probability', hue='run_name', ax=ax)
            
            ax.set_title(f'Layer {layer}', fontsize=TITLE_FONTSIZE)
            ax.set_ylabel('Probability Mass', fontsize=LABEL_FONTSIZE)
            ax.set_xlabel('')
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            ax.legend().set_visible(False)

        for i in range(num_layers, len(axes)):
            axes[i].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        
        RIGHT_MARGIN = 0.92  
        LEGEND_X_ANCHOR = 0.93 

        H_SPACE = 0.3 
        W_SPACE = 0.05  

        fig.legend(handles, labels, title='Run Name', bbox_to_anchor=(LEGEND_X_ANCHOR, 0.9), loc='upper left', fontsize=LEGEND_FONTSIZE)
        plt.subplots_adjust(left=0.03, right=RIGHT_MARGIN, hspace=H_SPACE, wspace=W_SPACE)
        
        first_run_name = next(iter(results.keys()))
        filename = f"{save_path}/{first_run_name}_combined_layer_distribution.png"
        
        plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.04)
        plt.close()
        print(f"✅ Combined layer distribution plot saved to {filename}")

    @staticmethod
    def _visualize_token_weights(result: AnalysisResult, save_path: str, batch_idx: int = 0, sample_idx: int = 0, layer_idx: int = 0, **kwargs):
        log = result.raw_data[batch_idx]
        mask = log.attention_mask[sample_idx]
        true_len = int(mask.sum().item())
        
        logits = torch.stack(log.top_k_logits)[layer_idx, sample_idx, :true_len, :]
        indices = torch.stack(log.top_k_indices)[layer_idx, sample_idx, :true_len, :]
        weights = torch.softmax(logits, dim=-1)
        
        plt.figure(figsize=(15, 6))
        for k_idx in range(weights.shape[1]):
            expert_ids = indices[:, k_idx]
            # Plot each expert's weight with a consistent color
            for expert_id in torch.unique(expert_ids):
                mask = expert_ids == expert_id
                plt.scatter(np.arange(true_len)[mask], weights[mask, k_idx], label=f'Expert {expert_id}' if k_idx==0 else None, color=plt.cm.tab10(expert_id))

        plt.title(f'"{result.name}" - L{layer_idx}, B{batch_idx}, S{sample_idx}: Top-{weights.shape[1]} Expert Weights')
        plt.xlabel('Token Position')
        plt.ylabel('Softmax Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = f"{save_path}/{result.name}_token_weights_L{layer_idx}_B{batch_idx}_S{sample_idx}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Token weights plot saved to {filename}")

    @staticmethod
    def _visualize_layer_matrix(result: AnalysisResult, save_path: str, batch_idx: int = 0, sample_idx: int = 0, **kwargs):
        log = result.raw_data[batch_idx]
        mask = log.attention_mask[sample_idx]
        true_len = int(mask.sum().item())
        
        indices = torch.stack(log.top_k_indices)[:, sample_idx, :true_len, 0] # (layers, seq_len)
        num_layers = indices.shape[0]

        plt.figure(figsize=(15, 8))
        for layer in range(num_layers):
            plt.scatter(x=np.arange(true_len), y=np.full(true_len, layer), c=indices[layer], cmap='tab10', marker='|', s=100, vmin=0, vmax=9)

        plt.colorbar(label='Top Expert ID').set_ticks(np.arange(10))
        plt.title(f'"{result.name}" - B{batch_idx}, S{sample_idx}: Top Expert Choice vs. Layer and Token')
        plt.xlabel('Token Position')
        plt.ylabel('Layer Index')
        plt.yticks(np.arange(num_layers))
        plt.grid(True, axis='x', alpha=0.2, linestyle=':')
        filename = f"{save_path}/{result.name}_layer_matrix_B{batch_idx}_S{sample_idx}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Layer matrix plot saved to {filename}")

    @staticmethod
    def _visualize_token_highlight(result: AnalysisResult, tokenizer: Any, batch_idx: int = 0, sample_idx: int = 0, **kwargs):
        if tokenizer is None: raise ValueError("Tokenizer must be provided.")
        log = result.raw_data[batch_idx]
        mask = log.attention_mask[sample_idx]
        true_len = int(mask.sum().item())
        
        indices = torch.stack(log.top_k_indices)[:, sample_idx, :true_len, 0]
        input_ids = log.input_ids[sample_idx, :true_len]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        dominant_expert, _ = torch.mode(indices, dim=0)

        print(f"\nDominant expert per token for '{result.name}':")
        for token, expert_id in zip(tokens, dominant_expert):
            color = RoutingAnalyzer._COLORS.get(expert_id.item(), '')
            print(f"{RoutingAnalyzer._COLORS['BOLD']}{color}{token}{RoutingAnalyzer._COLORS['ENDC']}", end=" ")
        print("\n")

    @staticmethod
    def _plot_heatmap(matrix: np.ndarray, labels: List[str], title: str, filename: str):
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel("Run Name")
        plt.ylabel("Run Name")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Heatmap saved to {filename}")
        
    # ===================================================================
    # METHODS for Saving and Loading State
    # ===================================================================
    
    @staticmethod
    def save_state(results: Dict[str, AnalysisResult], file_path: str):
        """Saves the analysis results dictionary to a file using pickle."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"💾 Analysis state saved to {file_path}")
        except Exception as e:
            print(f"Error saving state: {e}")

    @staticmethod
    def load_state(file_path: str) -> Dict[str, AnalysisResult]:
        """Loads the analysis results dictionary from a file if it exists."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
                print(f"✅ Analysis state loaded from {file_path}. Found {len(results)} completed runs.")
                return results
            except Exception as e:
                print(f"Could not load state file. Starting fresh. Error: {e}")
        return {}
