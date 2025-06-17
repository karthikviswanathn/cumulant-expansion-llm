import torch
import sys
sys.path.append("../")

sys.path.append("../../")
import cumulant_analyzer
from cumulant_analyzer import CumulantAnalyzer, to_numpy, calculate_cumulants

import pandas as pd

from IPython.display import display, HTML

def display_tokens(test_sequence, tokenizer, start_idx, end_idx):
    """Quick display of token sequence slice."""
    input_ids = tokenizer.encode(test_sequence.strip(), add_special_tokens=False, return_tensors="pt")
    token_slice = input_ids[0, start_idx:end_idx]
    decoded = tokenizer.decode(token_slice)
    # print(f"Tokens [{start_idx}:{end_idx}]: '{decoded}'")
    return decoded

def create_df(analyzer, test_sequence, max_length):
    all_logits, all_probs = [], []   
    input_ids = analyzer.tokenizer.encode(test_sequence.strip(), add_special_tokens = False, \
                     return_tensors = "pt", max_length = max_length, truncation = True).to('cuda')
    
    outputs = analyzer.model(input_ids[0:1, max_length//2:], output_hidden_states=True)
    logits = outputs.logits.squeeze()
    probs = torch.softmax(logits, dim = -1)
    all_logits.append(logits)
    all_probs.append(probs)
    
    outputs = analyzer.model(input_ids, output_hidden_states=True)
    logits = outputs.logits.squeeze()[max_length//2:]
    probs = torch.softmax(logits, dim = -1)
    all_logits.append(logits)
    all_probs.append(probs)

    stats = calculate_cumulants(all_logits, all_probs)
    
    entropy_com = stats['entropy_com'].cpu().numpy() if hasattr(stats['entropy_com'], 'cpu') else np.array(stats['entropy_com'])
    avg_entropy = stats['avg_entropy'].cpu().numpy() if hasattr(stats['avg_entropy'], 'cpu') else np.array(stats['avg_entropy'])
    kld_mean = stats['kld_center'].mean(-1).cpu().numpy() if hasattr(stats['kld_center'], 'cpu') else np.array(stats['kld_center'].mean(-1))
    
    # Extract first 5 cumulants (normalized)
    avg_cumulants = stats['avg_normalized_cumulants'].cpu().numpy() 
    
    # Create the DataFrame with entropy/KLD stats and first 5 cumulants
    df = pd.DataFrame({
        'avg_entropy': avg_entropy,
        'entropy_com': entropy_com,
        'kld_center': kld_mean,
        'κ₂': avg_cumulants[:, 0],  # 2nd cumulant (index 0)
        'κ₃': avg_cumulants[:, 1],  # 3rd cumulant (index 1) 
        'κ₄': avg_cumulants[:, 2],  # 4th cumulant (index 2)
        'κ₅': avg_cumulants[:, 3],  # 5th cumulant (index 3)
        'κ₆': avg_cumulants[:, 4],  # 6th cumulant (index 4)
    }, index=['Free', 'Context'])
    
    difference_row = df.iloc[0] - df.iloc[1]
    difference_row.name = 'Free - Context'
    df = pd.concat([df, difference_row.to_frame().T])

    return {'df': df, 'all_probs': all_probs, 'stats': stats}

def create_interactive_token_display(test_sequence, tokenizer, all_probs, max_length, k=3):
    """
    Create an interactive HTML display with hoverable tokens showing top-k predictions.
    Optimized for Jupyter Notebook.
    """
    # Tokenize the sequence
    input_ids = tokenizer.encode(test_sequence.strip(), add_special_tokens=False, 
                                return_tensors="pt", max_length=max_length, truncation=True)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
    
    # Convert probabilities to numpy if needed
    probs_with_context = to_numpy(all_probs[0])
    probs_no_context = to_numpy(all_probs[1])
    
    def get_topk_for_position(probs, position, k=3):
        """Get top-k predictions for a specific position."""
        if position >= len(probs):
            return []
        topk_probs, topk_indices = torch.topk(torch.tensor(probs[position]), k)
        predictions = []
        for j in range(k):
            token = tokenizer.decode([topk_indices[j].item()])
            prob = topk_probs[j].item()
            predictions.append((token, prob))
        return predictions
    
    def create_token_html(tokens_list, start_idx, end_idx, probs_context, probs_no_context):
        """Create HTML for tokens with hover information."""
        html_tokens = []
        
        for i in range(start_idx, min(end_idx, len(tokens_list))):
            token = tokens_list[i]
            
            # Get relative position for probability lookup
            if start_idx == 0:  # Context section
                prob_idx = i
            else:  # Query section
                prob_idx = i - max_length//2
            
            # Get top-k predictions if within bounds
            hover_parts = [f"Token: {token}"]
            
            if prob_idx < len(probs_context):
                # Add predictions with context
                preds_context = get_topk_for_position(probs_context, prob_idx, k)
                if preds_context:
                    hover_parts.append("\nWITH CONTEXT:")
                    for j, (pred_token, pred_prob) in enumerate(preds_context, 1):
                        hover_parts.append(f"{j}. {pred_token} ({pred_prob:.1%})")
                
                # Add predictions without context
                preds_no_context = get_topk_for_position(probs_no_context, prob_idx, k)
                if preds_no_context:
                    hover_parts.append("\nNO CONTEXT:")
                    for j, (pred_token, pred_prob) in enumerate(preds_no_context, 1):
                        hover_parts.append(f"{j}. {pred_token} ({pred_prob:.1%})")
            
            # Join with | for better display in Jupyter tooltips
            hover_text = " | ".join(hover_parts)
            
            # Escape special characters for HTML
            display_token = token.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '\\n').replace(' ', '&nbsp;')
            hover_text = hover_text.replace('"', '&quot;').replace("'", '&#39;')
            
            # Create span with hover using data attribute for better Jupyter compatibility
            html_tokens.append(
                f'<span class="token" title="{hover_text}" '
                f'data-toggle="tooltip" data-placement="top">'
                f'{display_token}</span>'
            )
        
        return ''.join(html_tokens)
    
    # Create the HTML with CSS
    html_output = """
    <style>
        .token {
            display: inline;
            padding: 2px 4px;
            margin: 1px;
            background-color: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 3px;
            cursor: help;
            transition: all 0.2s;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        .token:hover {
            background-color: #3498db;
            color: white;
            border-color: #2980b9;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .token-container {
            line-height: 2.0;
            padding: 15px;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            margin-bottom: 15px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        h4 {
            margin-bottom: 10px;
            color: #2c3e50;
        }
    </style>
    """
    
    # Display the styled content
    display(HTML(html_output))
    
    # Context section
    # display(HTML("<h4><b>Context</b></h4>"))
    # context_html = create_token_html(tokens, 0, max_length//2, 
    #                                probs_with_context, 
    #                                probs_with_context)
    # display(HTML(f'<div class="token-container">{context_html}</div>'))
    
    # Query section  
    display(HTML("<h4><b>Query</b></h4>"))
    query_html = create_token_html(tokens, max_length//2, max_length,
                                 probs_with_context, 
                                 probs_no_context)
    display(HTML(f'<div class="token-container">{query_html}</div>'))