#!/usr/bin/env python3
"""
Export script for multilingual-e5-large-instruct model to Qualcomm AI Engine Direct backend.

This script exports the Hugging Face multilingual-e5-large-instruct model for deployment
on Qualcomm SM8650 (Snapdragon 8 Gen 3) devices with Hexagon NPU acceleration.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add ExecutorCH to path
EXECUTORCH_ROOT = os.environ.get('EXECUTORCH_ROOT')
if EXECUTORCH_ROOT:
    sys.path.append(EXECUTORCH_ROOT)

try:
    from transformers import AutoTokenizer, AutoModel
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from torch.export import export, Dim
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and EXECUTORCH_ROOT is set correctly.")
    sys.exit(1)


class E5ModelWrapper(torch.nn.Module):
    """
    Wrapper for the multilingual-e5-large-instruct model optimized for mobile deployment.
    """
    
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large-instruct'):
        super().__init__()
        print(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use FP32 for better mobile compatibility
            trust_remote_code=True
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        # Model configuration
        self.max_length = 512
        self.hidden_size = 1024
        
        print(f"Model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Average pooling implementation optimized for mobile deployment.
        """
        # Mask hidden states
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        
        # Compute average
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass optimized for mobile deployment.
        
        Args:
            input_ids: Token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            
        Returns:
            embeddings: Normalized embeddings [batch_size, hidden_size]
        """
        # Model forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Average pooling
        embeddings = self.average_pool(outputs.last_hidden_state, attention_mask)
        
        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


def prepare_example_inputs(wrapper: E5ModelWrapper, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare example inputs for model export.
    """
    # Example instruction-tuned input
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    query = f'Instruct: {task}\nQuery: How to optimize deep learning models for mobile devices?'
    
    # Tokenize
    inputs = wrapper.tokenizer(
        query,
        max_length=wrapper.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Expand for batch size
    input_ids = inputs['input_ids'].expand(batch_size, -1)
    attention_mask = inputs['attention_mask'].expand(batch_size, -1)
    
    return input_ids, attention_mask


def export_model_to_qnn(
    wrapper: E5ModelWrapper,
    output_path: str,
    quantize: bool = True,
    profile_memory: bool = True
) -> str:
    """
    Export the model to Qualcomm AI Engine Direct backend.
    
    Args:
        wrapper: Model wrapper
        output_path: Output file path for .pte file
        quantize: Whether to apply quantization
        profile_memory: Whether to enable memory profiling
        
    Returns:
        Path to exported .pte file
    """
    print("Starting model export process...")
    
    # Prepare example inputs
    input_ids, attention_mask = prepare_example_inputs(wrapper)
    example_inputs = (input_ids, attention_mask)
    
    print(f"Example input shapes: {input_ids.shape}, {attention_mask.shape}")
    
    # Test model with example inputs
    with torch.no_grad():
        original_output = wrapper(*example_inputs)
        print(f"Original model output shape: {original_output.shape}")
    
    # Define dynamic shapes for variable sequence lengths
    dynamic_shapes = {
        "input_ids": {
            0: Dim("batch_size", min=1, max=8),
            1: Dim("seq_len", min=1, max=512)
        },
        "attention_mask": {
            0: Dim("batch_size", min=1, max=8),
            1: Dim("seq_len", min=1, max=512)
        }
    }
    
    try:
        # Export model
        print("Exporting model with torch.export...")
        exported_program = export(
            wrapper,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False  # Allow some flexibility for mobile deployment
        )
        print("Model exported successfully")
        
        # Apply quantization if requested
        if quantize:
            print("Applying quantization...")
            try:
                # Prepare model for quantization
                qconfig_mapping = get_default_qconfig_mapping("qnnpack")
                prepared_model = prepare_fx(wrapper, qconfig_mapping, example_inputs)
                
                # Calibrate with example data
                with torch.no_grad():
                    prepared_model(*example_inputs)
                
                # Convert to quantized model
                quantized_model = convert_fx(prepared_model)
                print("Quantization applied successfully")
                
                # Re-export quantized model
                exported_program = export(
                    quantized_model,
                    example_inputs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False
                )
                
            except Exception as e:
                print(f"Quantization failed: {e}")
                print("Continuing with full precision model...")
        
        # Lower to Qualcomm backend
        print("Lowering to Qualcomm AI Engine Direct backend...")
        
        partitioner_config = {
            "skip_node_id_set": set(),  # Don't skip any nodes initially
            "skip_node_op_set": set(),  # Can add problematic ops here if needed
        }
        
        edge_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[
                QnnPartitioner(
                    skip_node_id_set=partitioner_config["skip_node_id_set"],
                    skip_node_op_set=partitioner_config["skip_node_op_set"]
                )
            ],
            compile_config=None
        )
        
        print("Model lowered successfully")
        
        # Convert to ExecutorCH format
        print("Converting to ExecutorCH format...")
        executorch_program = edge_program.to_executorch()
        
        # Save the model
        pte_filename = output_path if output_path.endswith('.pte') else f"{output_path}.pte"
        
        with open(pte_filename, "wb") as f:
            f.write(executorch_program.buffer)
        
        print(f"Model exported successfully to: {pte_filename}")
        
        # Print model statistics
        buffer_size_mb = len(executorch_program.buffer) / (1024 * 1024)
        print(f"Model size: {buffer_size_mb:.2f} MB")
        
        return pte_filename
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_exported_model(pte_path: str, wrapper: E5ModelWrapper) -> bool:
    """
    Validate the exported model by comparing outputs.
    """
    try:
        from executorch.runtime import Runtime
        
        print("Validating exported model...")
        
        # Load runtime
        runtime = Runtime.get()
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        
        # Prepare test inputs
        input_ids, attention_mask = prepare_example_inputs(wrapper)
        
        # Run original model
        with torch.no_grad():
            original_output = wrapper(input_ids, attention_mask)
        
        # Run exported model
        exported_output = method.execute([input_ids, attention_mask])
        
        # Compare outputs
        if len(exported_output) > 0:
            cosine_sim = F.cosine_similarity(
                original_output.flatten(),
                torch.tensor(exported_output[0]).flatten(),
                dim=0
            )
            print(f"Cosine similarity: {cosine_sim:.4f}")
            
            if cosine_sim > 0.95:
                print("✅ Model validation passed")
                return True
            else:
                print("⚠️ Model validation warning: low similarity")
                return False
        else:
            print("❌ Model validation failed: no output")
            return False
            
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export multilingual-e5-large-instruct to Qualcomm QNN")
    parser.add_argument(
        "--model_name",
        type=str,
        default="intfloat/multilingual-e5-large-instruct",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="e5_model_qnn.pte",
        help="Output path for .pte file"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization for mobile deployment"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="SM8650",
        help="Target device (SM8650, SM8550, etc.)"
    )
    
    args = parser.parse_args()
    
    # Check environment
    if not EXECUTORCH_ROOT:
        print("Error: EXECUTORCH_ROOT environment variable not set")
        sys.exit(1)
    
    print(f"Exporting {args.model_name} for {args.device}")
    print(f"Quantization: {'enabled' if args.quantize else 'disabled'}")
    print(f"Output: {args.output_path}")
    print("-" * 50)
    
    # Load and wrap model
    try:
        wrapper = E5ModelWrapper(args.model_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Export model
    pte_path = export_model_to_qnn(
        wrapper,
        args.output_path,
        quantize=args.quantize
    )
    
    if pte_path is None:
        print("Export failed")
        sys.exit(1)
    
    # Validate if requested
    if args.validate:
        validation_success = validate_exported_model(pte_path, wrapper)
        if not validation_success:
            print("⚠️ Validation failed, but model was exported")
    
    print("\n" + "=" * 50)
    print("Export completed successfully!")
    print(f"Model file: {pte_path}")
    print(f"Target device: {args.device}")
    print("Next steps:")
    print("1. Deploy to Android device using the deployment script")
    print("2. Test inference performance")
    print("3. Integrate into your Android application")


if __name__ == "__main__":
    main()
