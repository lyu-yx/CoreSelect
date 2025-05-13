#!/usr/bin/env python
"""
Test script for comparing the performance of the original vs optimized subset selection implementations.
"""

import argparse
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Test optimized selection performance")
    parser.add_argument("--size", type=int, default=5000, help="Number of samples")
    parser.add_argument("--dims", type=int, default=128, help="Feature dimensions")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--selection", type=float, default=0.1, help="Selection fraction")
    parser.add_argument("--dpp-weight", type=float, default=0.3, help="DPP weight (0-1)")
    args = parser.parse_args()
    
    print(f"Generating random dataset with {args.size} samples, {args.dims} dimensions")
    
    # Generate random features
    features = np.random.randn(args.size, args.dims)
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    
    # Generate random labels
    labels = np.random.randint(0, args.classes, size=args.size)
    
    # Compute selection target
    target_size = int(args.size * args.selection)
    print(f"Selecting {target_size} samples ({args.selection * 100:.1f}% of data)")
    
    # Class counts
    class_counts = np.bincount(labels, minlength=args.classes)
    targets_per_class = {}
    for c in range(args.classes):
        targets_per_class[c] = int(class_counts[c] * args.selection)
    
    print(f"Class distribution: {class_counts}")
    print(f"Targets per class: {[targets_per_class[c] for c in range(args.classes)]}")
    
    # Run standard implementation
    try:
        from utils.submodular import get_orders_and_weights_hybrid
        
        start_time = time.time()
        order_mg, weights_mg, _, _, _, _ = get_orders_and_weights_hybrid(
            B=target_size,
            X=features,
            metric="cosine",
            y=labels,
            dpp_weight=args.dpp_weight
        )
        standard_time = time.time() - start_time
        
        print(f"Standard implementation: selected {len(order_mg)} samples in {standard_time:.2f}s")
    except ImportError:
        print("Standard implementation not available.")
        standard_time = float('inf')
    
    # Run optimized implementation
    try:
        from utils.optimized_selection import get_orders_and_weights_optimized
        
        start_time = time.time()
        order_mg, weights_mg, _, _, _, _ = get_orders_and_weights_optimized(
            B=target_size,
            X=features,
            metric="cosine",
            y=labels,
            dpp_weight=args.dpp_weight,
            verbose=True
        )
        optimized_time = time.time() - start_time
        
        print(f"Optimized implementation: selected {len(order_mg)} samples in {optimized_time:.2f}s")
        
        if standard_time != float('inf'):
            print(f"Speedup: {standard_time / optimized_time:.2f}x")
    except ImportError:
        print("Optimized implementation not available.")
    
if __name__ == "__main__":
    main()
