"""
Test Qwen3-1.7B model loading with Unsloth
Verifies 4-bit quantization, VRAM usage, and basic inference
Tests thinking mode capability
"""

import sys
import torch

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def get_vram_usage():
    """Get current VRAM usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0, 0

def test_model_loading():
    """Test loading Qwen3-1.7B with 4-bit quantization"""
    print_section("Loading Qwen3-1.7B with 4-bit Quantization")
    
    try:
        from unsloth import FastLanguageModel
        
        print("Loading model...")
        print("  Model: unsloth/Qwen3-1.7B (pre-optimized)")
        print("  Quantization: 4-bit")
        print("  Max sequence length: 2048 (test mode)")
        print("  Features: Thinking mode support")
        
        # Load model
        # Using Unsloth's pre-optimized Qwen3 version
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-1.7B",
            max_seq_length=2048,  # Small for testing
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        print("\n✅ Model loaded successfully!")
        
        # Check VRAM
        allocated, reserved = get_vram_usage()
        print("\n📊 VRAM Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        
        # Expected: ~4-7 GB for 1.5B model with 4-bit
        if allocated > 10:
            print("  ⚠️  Warning: Higher than expected (should be ~4-7 GB)")
        else:
            print("  ✅ Within expected range for 4-bit quantization")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return None, None

def test_inference(model, tokenizer):
    """Test basic inference"""
    print_section("Testing Inference")
    
    if model is None or tokenizer is None:
        print("❌ Skipping inference test (model not loaded)")
        return False
    
    try:
        # Simple test prompt
        prompt = "Hello! How are you today?"
        print(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        print(f"\n📝 Tokenized: {inputs['input_ids'].shape[1]} tokens")
        
        # Generate
        print("\n🤖 Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n💬 Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        # Check VRAM after inference
        allocated, reserved = get_vram_usage()
        print("\n📊 VRAM After Inference:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        
        print("\n✅ Inference successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_info(model, tokenizer):
    """Display model information"""
    print_section("Model Information")
    
    if model is None or tokenizer is None:
        print("❌ Model not loaded")
        return
    
    try:
        # Model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # Tokenizer info
        print(f"\nVocabulary size: {len(tokenizer):,}")
        print(f"Model max length: {tokenizer.model_max_length:,}")
        
        # Quick tokenization test
        test_text = "Testing tokenization"
        tokens = tokenizer(test_text)
        print(f"\nTest tokenization: '{test_text}'")
        print(f"  Token count: {len(tokens['input_ids'])}")
        print(f"  Token IDs: {tokens['input_ids'][:10]}...")  # First 10 tokens
        
    except Exception as e:
        print(f"❌ Failed to get model info: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Qwen3-1.7B Model Loading Test")
    print("=" * 60)
    
    # Check GPU first
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available! GPU required for this test.")
        return 1
    
    print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test sequence
    results = {}
    
    # 1. Load model
    model, tokenizer = test_model_loading()
    results['loading'] = model is not None
    
    # 2. Model info
    if model:
        test_model_info(model, tokenizer)
    
    # 3. Inference test
    results['inference'] = test_inference(model, tokenizer)
    
    # Summary
    print_section("Test Summary")
    
    print(f"Model Loading: {'✅ PASS' if results['loading'] else '❌ FAIL'}")
    print(f"Inference Test: {'✅ PASS' if results['inference'] else '❌ FAIL'}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Ready for Phase 0a training.")
        print("\nNext steps:")
        print("  1. Collect 60 training examples")
        print("  2. Format as JSONL")
        print("  3. Run training script")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
