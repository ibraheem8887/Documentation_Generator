#!/usr/bin/env python3
"""
Demo script for the Python Documentation Generation System

This script demonstrates the key features of the integrated system.
"""

import sys
import os

# Add integration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'integration'))

from doc_generator import DocumentationGenerator


def demo_basic_functionality():
    """Demonstrate basic documentation generation"""
    print("🚀 Python Documentation Generation System Demo")
    print("=" * 60)
    
    # Sample Python functions to demonstrate
    sample_functions = [
        """
def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)
    return bmi
""",
        """
def find_max(numbers):
    max_value = numbers[0]
    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value
""",
        """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
    ]
    
    try:
        # Initialize the generator
        print("🔧 Initializing Documentation Generator...")
        generator = DocumentationGenerator()
        
        # Show model info
        model_info = generator.get_model_info()
        print(f"✅ System ready!")
        print(f"📊 Tokenizer: {'✅' if model_info.get('tokenizer_loaded') else '❌'}")
        print(f"📊 Language Model: {'✅' if model_info.get('language_model_loaded') else '❌'}")
        print(f"📊 Device: {model_info.get('device', 'Unknown')}")
        print(f"📊 Vocabulary Size: {model_info.get('vocab_size', 'Unknown')}")
        print()
        
        # Process each sample function
        for i, func_code in enumerate(sample_functions, 1):
            print(f"📝 Demo {i}: Processing Function")
            print("-" * 40)
            
            # Show original code
            print("🔍 Original Code:")
            print(func_code.strip())
            print()
            
            # Generate documentation
            result = generator.generate_from_string(func_code)
            
            if result['status'] == 'success':
                func_info = result.get('function_info', {})
                
                print("✅ Generated Documentation:")
                print(f"📋 Function: {func_info.get('name', 'Unknown')}")
                print(f"📋 Parameters: {', '.join(func_info.get('args', []))}")
                print()
                print("📄 Summary:")
                print(result.get('summary', 'No summary available'))
                print()
                print("📝 Generated Docstring:")
                print(result.get('docstring', 'No docstring generated'))
                print()
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                print()
            
            print("=" * 60)
            print()
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_file_processing():
    """Demonstrate file processing capabilities"""
    print("📁 File Processing Demo")
    print("=" * 60)
    
    # Create a sample Python file
    sample_file_content = '''
def greet_user(name, greeting="Hello"):
    """This docstring will be replaced"""
    message = f"{greeting}, {name}!"
    return message

def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

class StringUtils:
    def reverse_string(self, text):
        return text[::-1]
    
    def count_words(self, text):
        return len(text.split())
'''
    
    try:
        # Save sample file
        sample_file = 'demo_sample.py'
        with open(sample_file, 'w') as f:
            f.write(sample_file_content)
        
        print(f"📄 Created sample file: {sample_file}")
        print()
        
        # Initialize generator
        generator = DocumentationGenerator()
        
        # Process the file
        print("🔄 Processing file...")
        result = generator.generate_from_file(sample_file)
        
        if result['status'] == 'success':
            print("✅ File processed successfully!")
            print()
            
            functions = result.get('functions', [])
            print(f"📊 Found {len(functions)} function(s)")
            print()
            
            # Show brief results
            for i, func_result in enumerate(functions):
                if func_result.get('status') == 'success':
                    func_info = func_result.get('function_info', {})
                    func_name = func_info.get('name', f'Function_{i+1}')
                    print(f"✅ {func_name}: {func_result.get('summary', 'No summary')}")
                else:
                    print(f"❌ Function {i+1}: {func_result.get('error', 'Unknown error')}")
            
            print()
            print("📄 Complete documentation available in result['documentation']")
        
        else:
            print(f"❌ Error processing file: {result.get('error', 'Unknown error')}")
        
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)
            print(f"🗑️ Cleaned up: {sample_file}")
    
    except Exception as e:
        print(f"❌ File processing demo failed: {e}")


def demo_cli_usage():
    """Show CLI usage examples"""
    print("💻 CLI Usage Examples")
    print("=" * 60)
    
    examples = [
        "# Generate documentation from code string:",
        'python run_system.py cli -s "def add(a, b): return a + b" -o docs.md',
        "",
        "# Generate documentation from Python file:",
        "python run_system.py cli my_script.py -o documentation.md",
        "",
        "# Run integration tests:",
        "python run_system.py test",
        "",
        "# Start web interface:",
        "python run_system.py web",
        "",
        "# Check system status:",
        "python run_system.py check"
    ]
    
    for example in examples:
        print(example)
    
    print()


def main():
    """Run all demos"""
    print("🎯 Python Documentation Generation System")
    print("🔗 Phase 5: System Integration Demo")
    print("=" * 60)
    print()
    
    # Demo 1: Basic functionality
    demo_basic_functionality()
    
    print("\n" + "=" * 60 + "\n")
    
    # Demo 2: File processing
    demo_file_processing()
    
    print("\n" + "=" * 60 + "\n")
    
    # Demo 3: CLI usage
    demo_cli_usage()
    
    print("=" * 60)
    print("🎉 Demo completed!")
    print()
    print("Next steps:")
    print("1. Run 'python run_system.py web' to start the web interface")
    print("2. Try the CLI with your own Python files")
    print("3. Check README_INTEGRATION.md for detailed documentation")


if __name__ == "__main__":
    main()
