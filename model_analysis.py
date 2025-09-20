"""
Model Analysis Script for S5 Assignment
Analyzes the CNN model for specific requirements:
1. Total Parameter Count Test
2. Use of Batch Normalization
3. Use of Dropout
4. Use of Fully Connected Layer or GAP
"""

import torch
import torch.nn as nn
import json
from collections import defaultdict

class DSConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, s, p, groups=in_c, bias=False, dilation=dilation)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SE(nn.Module):
    def __init__(self, c, r=12):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // r, 1, bias=True)
        self.fc2 = nn.Conv2d(c // r, c, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),   # 1 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 28x28 -> 28x28
        self.block1 = DSConv(16, 32, k=3, s=1, p=1, dilation=1)   # 16 -> 32
        self.pool1  = nn.MaxPool2d(2, 2)                          # 28 -> 14

        # 14x14 -> 14x14
        self.block2 = DSConv(32, 64, k=3, s=1, p=1, dilation=1)   # 32 -> 64
        self.block3 = DSConv(64, 64, k=3, s=1, p=2, dilation=2)   # dilation to grow RF
        self.pool2  = nn.MaxPool2d(2, 2)                          # 14 -> 7

        # 7x7 -> 7x7
        self.block4 = DSConv(64, 96, k=3, s=1, p=2, dilation=2)   # 64 -> 96
        self.se     = SE(96, r=12)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.03)
        self.fc   = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.stem(x)            # 16x28x28
        x = self.block1(x)          # 32x28x28
        x = self.pool1(x)           # 32x14x14

        x = self.block2(x)          # 64x14x14
        x = self.block3(x)          # 64x14x14 (dilated)
        x = self.pool2(x)           # 64x7x7

        x = self.block4(x)          # 96x7x7 (dilated)
        x = self.se(x)              # 96x7x7

        x = self.gap(x).view(x.size(0), -1)  # 96
        x = self.drop(x)
        x = self.fc(x)              # 10
        return x

def analyze_model():
    """Analyze the CNN model for specific requirements"""
    
    # Create model instance
    model = CNN()
    
    # Initialize analysis results
    analysis_results = {
        "total_parameters": 0,
        "batch_normalization": {"found": False, "layers": [], "count": 0},
        "dropout": {"found": False, "layers": [], "count": 0},
        "gap_or_fc": {"gap_found": False, "fc_found": False, "gap_layers": [], "fc_layers": []},
        "requirements_met": {}
    }
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    analysis_results["total_parameters"] = total_params
    
    # Analyze each module
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # Check for Batch Normalization
        if isinstance(module, nn.BatchNorm2d):
            analysis_results["batch_normalization"]["found"] = True
            analysis_results["batch_normalization"]["count"] += 1
            analysis_results["batch_normalization"]["layers"].append({
                "name": name,
                "type": module_type,
                "num_features": module.num_features
            })
        
        # Check for Dropout
        elif isinstance(module, nn.Dropout):
            analysis_results["dropout"]["found"] = True
            analysis_results["dropout"]["count"] += 1
            analysis_results["dropout"]["layers"].append({
                "name": name,
                "type": module_type,
                "p": module.p
            })
        
        # Check for Global Average Pooling
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            analysis_results["gap_or_fc"]["gap_found"] = True
            analysis_results["gap_or_fc"]["gap_layers"].append({
                "name": name,
                "type": module_type,
                "output_size": module.output_size
            })
        
        # Check for Fully Connected Layer
        elif isinstance(module, nn.Linear):
            analysis_results["gap_or_fc"]["fc_found"] = True
            analysis_results["gap_or_fc"]["fc_layers"].append({
                "name": name,
                "type": module_type,
                "in_features": module.in_features,
                "out_features": module.out_features
            })
    
    # Check requirements
    analysis_results["requirements_met"] = {
        "parameter_count": {
            "requirement": "< 20,000",
            "actual": total_params,
            "passed": total_params < 20000
        },
        "batch_normalization": {
            "requirement": "Must use Batch Normalization",
            "found": analysis_results["batch_normalization"]["found"],
            "count": analysis_results["batch_normalization"]["count"],
            "passed": analysis_results["batch_normalization"]["found"]
        },
        "dropout": {
            "requirement": "Must use Dropout",
            "found": analysis_results["dropout"]["found"],
            "count": analysis_results["dropout"]["count"],
            "passed": analysis_results["dropout"]["found"]
        },
        "gap_or_fc": {
            "requirement": "Must use GAP or Fully Connected Layer",
            "gap_found": analysis_results["gap_or_fc"]["gap_found"],
            "fc_found": analysis_results["gap_or_fc"]["fc_found"],
            "passed": analysis_results["gap_or_fc"]["gap_found"] or analysis_results["gap_or_fc"]["fc_found"]
        }
    }
    
    return analysis_results

def print_analysis_report(results):
    """Print a formatted analysis report"""
    
    print("=" * 80)
    print("MNIST CNN MODEL ANALYSIS REPORT")
    print("=" * 80)
    
    # Total Parameter Count Test
    print("\n1️⃣ TOTAL PARAMETER COUNT TEST")
    print("-" * 40)
    param_req = results["requirements_met"]["parameter_count"]
    print(f"Requirement: {param_req['requirement']} parameters")
    print(f"Actual Count: {param_req['actual']:,} parameters")
    print(f"Status: {'✅ PASSED' if param_req['passed'] else '❌ FAILED'}")
    
    # Batch Normalization
    print("\n2️⃣ BATCH NORMALIZATION USAGE")
    print("-" * 40)
    bn_req = results["requirements_met"]["batch_normalization"]
    print(f"Requirement: Must use Batch Normalization")
    print(f"Found: {bn_req['found']}")
    print(f"Count: {bn_req['count']} BatchNorm2d layers")
    if results["batch_normalization"]["layers"]:
        print("Locations:")
        for layer in results["batch_normalization"]["layers"]:
            print(f"  - {layer['name']}: {layer['type']} ({layer['num_features']} features)")
    print(f"Status: {'✅ PASSED' if bn_req['passed'] else '❌ FAILED'}")
    
    # Dropout
    print("\n3️⃣ DROPOUT USAGE")
    print("-" * 40)
    dropout_req = results["requirements_met"]["dropout"]
    print(f"Requirement: Must use Dropout")
    print(f"Found: {dropout_req['found']}")
    print(f"Count: {dropout_req['count']} Dropout layers")
    if results["dropout"]["layers"]:
        print("Locations:")
        for layer in results["dropout"]["layers"]:
            print(f"  - {layer['name']}: {layer['type']} (p={layer['p']})")
    print(f"Status: {'✅ PASSED' if dropout_req['passed'] else '❌ FAILED'}")
    
    # GAP or FC Layer
    print("\n4️⃣ GAP OR FULLY CONNECTED LAYER USAGE")
    print("-" * 40)
    gap_fc_req = results["requirements_met"]["gap_or_fc"]
    print(f"Requirement: Must use GAP or Fully Connected Layer")
    print(f"GAP Found: {gap_fc_req['gap_found']}")
    print(f"FC Found: {gap_fc_req['fc_found']}")
    
    if results["gap_or_fc"]["gap_layers"]:
        print("GAP Layers:")
        for layer in results["gap_or_fc"]["gap_layers"]:
            print(f"  - {layer['name']}: {layer['type']} (output_size={layer['output_size']})")
    
    if results["gap_or_fc"]["fc_layers"]:
        print("FC Layers:")
        for layer in results["gap_or_fc"]["fc_layers"]:
            print(f"  - {layer['name']}: {layer['type']} ({layer['in_features']}→{layer['out_features']})")
    
    print(f"Status: {'✅ PASSED' if gap_fc_req['passed'] else '❌ FAILED'}")
    
    # Overall Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    all_passed = all([
        param_req['passed'],
        bn_req['passed'],
        dropout_req['passed'],
        gap_fc_req['passed']
    ])
    
    print(f"All Requirements Met: {'✅ YES' if all_passed else '❌ NO'}")
    print(f"Model is compliant with all specified requirements: {'✅ YES' if all_passed else '❌ NO'}")

def save_analysis_to_file(results, filename="model_analysis_results.json"):
    """Save analysis results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis results saved to: {filename}")

if __name__ == "__main__":
    # Run analysis
    results = analyze_model()
    
    # Print report
    print_analysis_report(results)
    
    # Save results
    save_analysis_to_file(results)
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check the generated JSON file for detailed results.")
    print("=" * 80)
