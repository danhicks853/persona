"""
Test basic Unsloth functionality
Verifies that Unsloth can be imported and initialized successfully
"""

import sys

def test_unsloth_import():
    """Test that Unsloth can be imported"""
    print("Testing Unsloth import...")
    try:
        from unsloth import FastLanguageModel  # noqa: F401
        print("✅ Unsloth imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import Unsloth: {e}")
        return False

def test_pytorch_cuda():
    """Verify PyTorch CUDA is available"""
    print("\nTesting PyTorch CUDA...")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print("✅ PyTorch CUDA working")
            return True
        else:
            print("⚠️  CUDA not available")
            return False
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_unsloth_components():
    """Test that key Unsloth components are available"""
    print("\nTesting Unsloth components...")
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: F401
        print("   FastLanguageModel: ✅")
        print("   is_bfloat16_supported: ✅")
        print(f"   BFloat16 supported: {is_bfloat16_supported()}")
        print("✅ Unsloth components available")
        return True
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Unsloth Basic Functionality Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("PyTorch CUDA", test_pytorch_cuda()))
    results.append(("Unsloth Import", test_unsloth_import()))
    results.append(("Unsloth Components", test_unsloth_components()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Unsloth is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
