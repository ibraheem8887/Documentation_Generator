"""
Test script for the integrated documentation generation system.

This script tests the complete pipeline with sample Python functions
to validate the system integration.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from doc_generator import DocumentationGenerator


def test_sample_functions():
    """Test the system with various sample Python functions"""
    
    # Sample functions to test
    test_functions = [
        # Simple function
        """
def add_numbers(a, b):
    result = a + b
    return result
""",
        
        # Function with type hints
        """
def calculate_area(length: float, width: float) -> float:
    area = length * width
    return area
""",
        
        # More complex function
        """
def process_data(data_list, threshold=10):
    filtered_data = []
    for item in data_list:
        if item > threshold:
            filtered_data.append(item * 2)
    return filtered_data
""",
        
        # Function with docstring (to be replaced)
        """
def fibonacci(n):
    \"\"\"Old docstring to be replaced\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        
        # Class method
        """
class Calculator:
    def multiply(self, x, y):
        return x * y
"""
    ]
    
    print("🧪 Testing Documentation Generation System")
    print("=" * 50)
    
    try:
        # Initialize generator
        print("Initializing documentation generator...")
        generator = DocumentationGenerator()
        
        # Print model info
        model_info = generator.get_model_info()
        print(f"✅ Generator initialized successfully")
        print(f"📊 Model Info: {model_info}")
        print()
        
        # Test each function
        for i, func_code in enumerate(test_functions, 1):
            print(f"🔍 Testing Function {i}")
            print("-" * 30)
            
            # Show original code
            print("Original Code:")
            print(func_code.strip())
            print()
            
            # Generate documentation
            result = generator.generate_from_string(func_code)
            
            if result['status'] == 'success':
                print("✅ Documentation generated successfully!")
                print()
                
                # Show results
                func_info = result.get('function_info', {})
                print(f"Function Name: {func_info.get('name', 'Unknown')}")
                print(f"Parameters: {func_info.get('args', [])}")
                print()
                
                print("Generated Summary:")
                print(result.get('summary', 'No summary'))
                print()
                
                print("Generated Docstring:")
                print(result.get('docstring', 'No docstring'))
                print()
                
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                print()
            
            print("=" * 50)
            print()
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_file_processing():
    """Test processing a Python file with multiple functions"""
    
    print("🧪 Testing File Processing")
    print("=" * 50)
    
    # Create a test file
    test_file_content = '''
def greet(name):
    message = f"Hello, {name}!"
    return message

def calculate_discount(price, discount_percent):
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount
    return final_price

class MathUtils:
    def square(self, x):
        return x * x
    
    def cube(self, x):
        return x * x * x
'''
    
    try:
        # Save test file
        test_file_path = 'test_sample.py'
        with open(test_file_path, 'w') as f:
            f.write(test_file_content)
        
        print(f"📄 Created test file: {test_file_path}")
        
        # Initialize generator
        generator = DocumentationGenerator()
        
        # Process file
        result = generator.generate_from_file(test_file_path)
        
        if result['status'] == 'success':
            print("✅ File processed successfully!")
            print()
            
            functions = result.get('functions', [])
            print(f"Found {len(functions)} function(s)")
            print()
            
            # Show combined documentation
            print("Combined Documentation:")
            print(result.get('documentation', 'No documentation'))
            
        else:
            print(f"❌ Error processing file: {result.get('error', 'Unknown error')}")
        
        # Clean up
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"🗑️ Cleaned up test file: {test_file_path}")
    
    except Exception as e:
        print(f"❌ File processing test failed: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling with invalid code"""
    
    print("🧪 Testing Error Handling")
    print("=" * 50)
    
    invalid_codes = [
        # Syntax error
        "def invalid_function(\n    return 'missing colon'",
        
        # Empty string
        "",
        
        # Not a function
        "x = 5\ny = 10\nprint(x + y)",
    ]
    
    try:
        generator = DocumentationGenerator()
        
        for i, code in enumerate(invalid_codes, 1):
            print(f"Testing invalid code {i}:")
            print(f"Code: {repr(code)}")
            
            result = generator.generate_from_string(code)
            
            if result['status'] == 'error':
                print(f"✅ Error handled correctly: {result.get('error', 'Unknown error')}")
            else:
                print(f"⚠️ Expected error but got: {result['status']}")
            
            print()
    
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")


def main():
    """Run all tests"""
    print("🚀 Starting Integration Tests")
    print("=" * 60)
    print()
    
    # Test 1: Sample functions
    test_sample_functions()
    
    print("\n" + "=" * 60 + "\n")
    
    # Test 2: File processing
    test_file_processing()
    
    print("\n" + "=" * 60 + "\n")
    
    # Test 3: Error handling
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("🏁 Integration tests completed!")


if __name__ == "__main__":
    main()
