"""
Documentation Generation System

This module provides the main interface for generating documentation
from Python code using the integrated pipeline.
"""

import os
import sys
import ast
import re
from typing import List, Dict, Optional, Union
import tempfile
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from pipeline import DocumentationPipeline


class DocumentationGenerator:
    """
    Main documentation generator that provides high-level interface
    for generating documentation from Python code.
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the documentation generator.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.pipeline = DocumentationPipeline(model_dir)
        self.supported_extensions = ['.py']
    
    def generate_from_string(self, code: str) -> Dict:
        """
        Generate documentation from a Python code string.
        
        Args:
            code: Python code as string
            
        Returns:
            Dictionary containing generated documentation
        """
        try:
            result = self.pipeline.generate_documentation(code)
            result['status'] = 'success'
            result['error'] = None
            return result
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'function_info': None,
                'summary': None,
                'docstring': None,
                'full_documentation': f"Error generating documentation: {str(e)}"
            }
    
    def generate_from_file(self, file_path: str) -> Dict:
        """
        Generate documentation from a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary containing generated documentation for all functions
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not any(file_path.endswith(ext) for ext in self.supported_extensions):
                raise ValueError(f"Unsupported file type. Supported: {self.supported_extensions}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Extract all functions from the file
            functions = self.extract_functions_from_code(code)
            
            if not functions:
                return {
                    'status': 'warning',
                    'error': 'No functions found in file',
                    'file_path': file_path,
                    'functions': [],
                    'documentation': "No functions found to document."
                }
            
            # Generate documentation for each function
            results = []
            for func_code in functions:
                doc_result = self.generate_from_string(func_code)
                results.append(doc_result)
            
            return {
                'status': 'success',
                'error': None,
                'file_path': file_path,
                'functions': results,
                'documentation': self._combine_documentation(results)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'file_path': file_path,
                'functions': [],
                'documentation': f"Error processing file: {str(e)}"
            }
    
    def extract_functions_from_code(self, code: str) -> List[str]:
        """
        Extract individual function definitions from Python code.
        
        Args:
            code: Python code as string
            
        Returns:
            List of function code strings
        """
        try:
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get the source code for this function
                    func_lines = code.split('\n')[node.lineno-1:node.end_lineno]
                    func_code = '\n'.join(func_lines)
                    functions.append(func_code)
            
            return functions
            
        except Exception as e:
            print(f"Error extracting functions: {e}")
            # Fallback: return entire code as single function
            return [code]
    
    def generate_batch(self, code_snippets: List[str]) -> List[Dict]:
        """
        Generate documentation for multiple code snippets.
        
        Args:
            code_snippets: List of Python code strings
            
        Returns:
            List of documentation results
        """
        results = []
        for i, code in enumerate(code_snippets):
            print(f"Processing snippet {i+1}/{len(code_snippets)}...")
            result = self.generate_from_string(code)
            results.append(result)
        
        return results
    
    def _combine_documentation(self, results: List[Dict]) -> str:
        """
        Combine documentation from multiple functions into a single document.
        
        Args:
            results: List of documentation results
            
        Returns:
            Combined documentation string
        """
        if not results:
            return "No documentation generated."
        
        doc = "# Generated Documentation\n\n"
        
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            doc += "No functions were successfully processed.\n\n"
            doc += "## Errors:\n"
            for i, result in enumerate(results):
                if result.get('error'):
                    doc += f"{i+1}. {result['error']}\n"
            return doc
        
        doc += f"Generated documentation for {len(successful_results)} function(s).\n\n"
        
        for i, result in enumerate(successful_results):
            func_info = result.get('function_info', {})
            func_name = func_info.get('name', f'Function_{i+1}')
            
            doc += f"## {func_name}\n\n"
            doc += result.get('full_documentation', 'No documentation available') + "\n\n"
            doc += "---\n\n"
        
        # Add error summary if any
        error_results = [r for r in results if r.get('status') == 'error']
        if error_results:
            doc += "## Processing Errors\n\n"
            for i, result in enumerate(error_results):
                doc += f"{i+1}. {result.get('error', 'Unknown error')}\n"
        
        return doc
    
    def save_documentation(self, documentation: str, output_path: str):
        """
        Save generated documentation to a file.
        
        Args:
            documentation: Documentation string
            output_path: Path to save the documentation
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(documentation)
            
            print(f"Documentation saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving documentation: {e}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'tokenizer_loaded': self.pipeline.tokenizer is not None,
            'word2vec_loaded': self.pipeline.word2vec_model is not None,
            'language_model_loaded': self.pipeline.language_model is not None,
            'device': str(self.pipeline.device)
        }
        
        if self.pipeline.tokenizer and hasattr(self.pipeline.tokenizer, 'vocab'):
            info['vocab_size'] = len(self.pipeline.tokenizer.vocab)
        
        return info


def main():
    """
    Command line interface for the documentation generator.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate documentation for Python code')
    parser.add_argument('input', help='Input Python file or code string')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-s', '--string', action='store_true', 
                       help='Treat input as code string instead of file path')
    parser.add_argument('--model-dir', help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Initialize generator
    try:
        generator = DocumentationGenerator(args.model_dir)
        print("✅ Documentation generator initialized")
        print("Model info:", generator.get_model_info())
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return
    
    # Generate documentation
    if args.string:
        # Input is code string
        result = generator.generate_from_string(args.input)
        documentation = result.get('full_documentation', 'No documentation generated')
    else:
        # Input is file path
        result = generator.generate_from_file(args.input)
        documentation = result.get('documentation', 'No documentation generated')
    
    # Output results
    if args.output:
        generator.save_documentation(documentation, args.output)
    else:
        print("\n" + "="*50)
        print("GENERATED DOCUMENTATION")
        print("="*50)
        print(documentation)
    
    # Print status
    print(f"\nStatus: {result.get('status', 'unknown')}")
    if result.get('error'):
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
