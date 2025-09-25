import ast

def generate_docs_from_file(filename: str):
    with open(filename, "r") as f:
        tree = ast.parse(f.read())
    
    docs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            docstring = ast.get_docstring(node)
            args = [arg.arg for arg in node.args.args]
            docs.append({
                "Function": func_name,
                "Arguments": args,
                "Docstring": docstring or "No docstring provided"
            })
    return docs

# Example usage:
if __name__ == "__main__":
    for d in generate_docs_from_file("example.py"):
        print(d)
