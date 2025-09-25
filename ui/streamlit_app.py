"""
Streamlit UI for Documentation Generation System

Interactive web interface for generating Python function documentation
using BPE tokenization, Word2Vec embeddings, and BiLSTM language model.
"""

import streamlit as st
import sys
import os
import tempfile
import traceback
from io import StringIO

# Add integration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'integration'))

try:
    from doc_generator import DocumentationGenerator
except ImportError as e:
    st.error(f"Error importing documentation generator: {e}")
    st.stop()


def initialize_generator():
    """Initialize the documentation generator with caching"""
    if 'generator' not in st.session_state:
        try:
            with st.spinner("Loading models... This may take a moment."):
                st.session_state.generator = DocumentationGenerator()
                st.session_state.model_info = st.session_state.generator.get_model_info()
        except Exception as e:
            st.error(f"Failed to initialize documentation generator: {e}")
            st.error("Please ensure all model files are present in the correct directories.")
            st.stop()


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Python Documentation Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("📚 Python Documentation Generator")
    st.markdown("""
    Generate professional documentation for your Python functions using AI!
    
    This tool uses:
    - **BPE Tokenization** for code preprocessing
    - **Word2Vec Embeddings** for semantic understanding
    - **BiLSTM Language Model** for docstring generation
    """)
    
    # Initialize generator
    initialize_generator()
    
    # Sidebar with model information
    with st.sidebar:
        st.header("🔧 System Status")
        
        if 'model_info' in st.session_state:
            info = st.session_state.model_info
            
            st.success("✅ System Ready")
            
            with st.expander("Model Details"):
                st.write(f"**Device:** {info.get('device', 'Unknown')}")
                st.write(f"**Tokenizer:** {'✅' if info.get('tokenizer_loaded') else '❌'}")
                st.write(f"**Word2Vec:** {'✅' if info.get('word2vec_loaded') else '❌'}")
                st.write(f"**Language Model:** {'✅' if info.get('language_model_loaded') else '❌'}")
                if 'vocab_size' in info:
                    st.write(f"**Vocabulary Size:** {info['vocab_size']}")
        
        st.header("📖 How to Use")
        st.markdown("""
        1. **Choose input method** (paste code or upload file)
        2. **Enter your Python function**
        3. **Click Generate Documentation**
        4. **Review and download results**
        """)
        
        st.header("💡 Tips")
        st.markdown("""
        - Functions should be syntactically correct
        - Include type hints for better results
        - Complex functions may take longer to process
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Paste Code", "Upload File"],
            horizontal=True
        )
        
        code_input = ""
        
        if input_method == "Paste Code":
            # Text area for code input
            code_input = st.text_area(
                "Enter Python function code:",
                height=300,
                placeholder="""def example_function(x, y):
    \"\"\"This docstring will be replaced\"\"\"
    result = x + y
    return result""",
                help="Paste your Python function code here"
            )
        
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Choose a Python file",
                type=['py'],
                help="Upload a .py file containing Python functions"
            )
            
            if uploaded_file is not None:
                # Read file content
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                code_input = stringio.read()
                
                # Show file preview
                with st.expander("📄 File Preview"):
                    st.code(code_input[:500] + "..." if len(code_input) > 500 else code_input, language="python")
        
        # Generation options
        st.subheader("⚙️ Options")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            max_length = st.slider("Max docstring length", 50, 200, 100)
        with col_opt2:
            include_summary = st.checkbox("Include function summary", value=True)
        
        # Generate button
        generate_button = st.button(
            "🚀 Generate Documentation",
            type="primary",
            disabled=not code_input.strip(),
            use_container_width=True
        )
    
    with col2:
        st.header("📄 Generated Documentation")
        
        if generate_button and code_input.strip():
            
            with st.spinner("Generating documentation... Please wait."):
                try:
                    # Generate documentation
                    if input_method == "Upload File":
                        # Process as file (may contain multiple functions)
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                            tmp_file.write(code_input)
                            tmp_file_path = tmp_file.name
                        
                        result = st.session_state.generator.generate_from_file(tmp_file_path)
                        os.unlink(tmp_file_path)  # Clean up temp file
                        
                    else:
                        # Process as code string
                        result = st.session_state.generator.generate_from_string(code_input)
                    
                    # Display results
                    if result['status'] == 'success':
                        st.success("✅ Documentation generated successfully!")
                        
                        if input_method == "Upload File":
                            # Multiple functions
                            documentation = result.get('documentation', '')
                            st.markdown(documentation)
                            
                            # Show individual function results
                            functions = result.get('functions', [])
                            if functions:
                                st.subheader("📋 Individual Functions")
                                for i, func_result in enumerate(functions):
                                    if func_result.get('status') == 'success':
                                        func_info = func_result.get('function_info', {})
                                        func_name = func_info.get('name', f'Function {i+1}')
                                        
                                        with st.expander(f"🔍 {func_name}"):
                                            if include_summary:
                                                st.write("**Summary:**", func_result.get('summary', 'N/A'))
                                            st.code(func_result.get('docstring', ''), language="python")
                        
                        else:
                            # Single function
                            func_info = result.get('function_info', {})
                            
                            if include_summary:
                                st.subheader("📊 Summary")
                                st.info(result.get('summary', 'No summary available'))
                            
                            st.subheader("📝 Generated Docstring")
                            docstring = result.get('docstring', '')
                            st.code(docstring, language="python")
                            
                            st.subheader("📋 Complete Documentation")
                            st.markdown(result.get('full_documentation', ''))
                        
                        # Download button
                        documentation_text = result.get('full_documentation') or result.get('documentation', '')
                        if documentation_text:
                            st.download_button(
                                label="💾 Download Documentation",
                                data=documentation_text,
                                file_name="generated_documentation.md",
                                mime="text/markdown"
                            )
                    
                    else:
                        # Error occurred
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                        
                        if st.checkbox("Show detailed error"):
                            st.code(result.get('traceback', 'No traceback available'))
                
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    if st.checkbox("Show error details"):
                        st.code(traceback.format_exc())
        
        elif not code_input.strip():
            st.info("👆 Please enter some Python code or upload a file to get started!")
        
        else:
            st.info("Click 'Generate Documentation' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit | Powered by BPE + Word2Vec + BiLSTM</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
