#!/usr/bin/env python3
"""Verification script to validate all created files."""

import ast
import sys
from pathlib import Path


def verify_python_syntax(filepath: Path) -> tuple[bool, str]:
    """Verify Python file syntax and return result."""
    try:
        with open(filepath) as f:
            ast.parse(f.read())
        return True, f"✓ {filepath.name}"
    except SyntaxError as e:
        return False, f"✗ {filepath.name}: {e}"


def verify_imports(filepath: Path) -> tuple[bool, str]:
    """Verify imports in file."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return True, f"✓ {filepath.name}: {len(set(imports))} unique imports"
    except Exception as e:
        return False, f"✗ {filepath.name}: {e}"


def main():
    """Run verification."""
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    print("AgentArmy File Verification")
    print("=" * 60)
    
    if not src_dir.exists():
        print("✗ src/ directory not found!")
        sys.exit(1)
    
    # Find all Python files
    py_files = list(src_dir.rglob("*.py"))
    
    if not py_files:
        print("✗ No Python files found!")
        sys.exit(1)
    
    print(f"\nFound {len(py_files)} Python files\n")
    
    # Verify syntax
    print("Syntax Verification:")
    print("-" * 60)
    syntax_ok = True
    for filepath in sorted(py_files):
        ok, msg = verify_python_syntax(filepath)
        print(msg)
        if not ok:
            syntax_ok = False
    
    if not syntax_ok:
        print("\n✗ Syntax errors found!")
        sys.exit(1)
    
    # Check file sizes
    print("\n\nFile Sizes:")
    print("-" * 60)
    total_lines = 0
    for filepath in sorted(py_files):
        with open(filepath) as f:
            lines = len(f.readlines())
        total_lines += lines
        rel_path = filepath.relative_to(project_root)
        print(f"{str(rel_path):50} {lines:5} lines")
    
    print("-" * 60)
    print(f"{'TOTAL':50} {total_lines:5} lines")
    
    # Check for required classes
    print("\n\nClass Verification:")
    print("-" * 60)
    
    required_classes = {
        "src/models/schemas.py": [
            "ModelProvider", "ModelTier", "SensitivityLevel",
            "LLMRequest", "LLMResponse", "ToolDefinition", "ToolCall"
        ],
        "src/models/router.py": [
            "RoutingRule", "ModelRouter", "CircuitBreakerState"
        ],
        "src/models/claude_client.py": ["ClaudeClient"],
        "src/models/ollama_client.py": ["OllamaClient"],
        "src/bridges/whatsapp.py": [
            "WhatsAppMessage", "WhatsAppBridge", "WhatsAppMessageType"
        ],
        "src/bridges/webhook_handler.py": [
            "WebhookEvent", "WebhookHandler", "WebhookSource"
        ],
        "src/protocols/a2a.py": [
            "A2AMessage", "A2AMessageType", "A2AProtocol", "A2ACapability"
        ],
    }
    
    for filepath_str, classes in required_classes.items():
        filepath = project_root / filepath_str
        try:
            with open(filepath) as f:
                content = f.read()
            
            found = []
            missing = []
            for cls in classes:
                if f"class {cls}" in content:
                    found.append(cls)
                else:
                    missing.append(cls)
            
            status = "✓" if not missing else "✗"
            print(f"{status} {filepath_str}")
            if missing:
                print(f"  Missing: {', '.join(missing)}")
        except FileNotFoundError:
            print(f"✗ {filepath_str} not found!")
    
    print("\n" + "=" * 60)
    print("✓ All verifications passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
