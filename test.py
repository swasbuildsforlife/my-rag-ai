"""
My RAG AI - Project Tests

Run:
    python test.py
"""

import ast
import os
import sys


APP_FILE = "app.py"


def load_app_source():
    """Read app.py safely for static analysis."""
    with open(APP_FILE, "r", encoding="utf-8") as file:
        return file.read()


def load_app_tree():
    """Parse app.py and verify Python syntax."""
    source = load_app_source()
    return ast.parse(source)


def test_required_files():
    """Check that essential project files exist."""
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
    ]

    missing = [
        file for file in required_files
        if not os.path.exists(file)
    ]

    assert not missing, f"Missing files: {missing}"


def test_app_syntax():
    """Check that app.py has valid Python syntax."""
    load_app_tree()


def test_required_functions():
    """Check that important RAG functions exist."""
    tree = load_app_tree()

    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    required_functions = [
        "get_relevant_documents",
        "build_context",
        "generate_with_fallback",
        "create_local_fallback_answer",
        "normalize_text",
        "document_key",
    ]

    missing = [
        name
        for name in required_functions
        if name not in functions
    ]

    assert not missing, (
        f"Missing RAG functions: {missing}"
    )


def test_rag_components():
    """Check that the main RAG technologies are used."""
    source = load_app_source().lower()

    required_components = {
        "chromadb": "chroma",
        "huggingface embeddings": "huggingfaceembeddings",
        "similarity search": "similarity_search_with_score",
        "mmr retrieval": 'search_type="mmr"',
        "gemini generation": "generate_content",
        "pdf extraction": "pdfreader",
        "text chunking": "recursivecharactertextsplitter",
    }

    missing = [
        name
        for name, code in required_components.items()
        if code not in source
    ]

    assert not missing, (
        f"Missing RAG components: {missing}"
    )


def test_rag_retrieval_configuration():
    """Check that similarity + MMR retrieval are configured."""
    source = load_app_source()

    assert "similarity_search_with_score" in source
    assert 'search_type="mmr"' in source
    assert "fetch_k" in source
    assert "lambda_mult" in source


def test_readme():
    """Check that README exists and contains core project information."""
    assert os.path.exists("README.md")

    with open("README.md", "r", encoding="utf-8") as file:
        readme = file.read().lower()

    required_sections = [
        "my rag ai",
        "rag pipeline",
        "tech stack",
        "live demo",
    ]

    missing = [
        section
        for section in required_sections
        if section not in readme
    ]

    assert not missing, (
        f"README is missing: {missing}"
    )


def test_requirements():
    """Check that requirements.txt exists and is not empty."""
    assert os.path.exists("requirements.txt")
    assert os.path.getsize("requirements.txt") > 0


def run_tests():
    tests = [
        test_required_files,
        test_app_syntax,
        test_required_functions,
        test_rag_components,
        test_rag_retrieval_configuration,
        test_readme,
        test_requirements,
    ]

    passed = 0

    print("\n🧪 My RAG AI — Project Tests\n")

    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1

        except Exception as error:
            print(f"❌ {test.__name__}")
            print(f"   {error}")

    print(
        f"\nResult: {passed}/{len(tests)} tests passed."
    )

    if passed == len(tests):
        print("🎉 All project tests passed!")
        return 0

    print("⚠️ Some tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(run_tests())