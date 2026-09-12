"""
Basic tests for My RAG AI.

Run:
    python test.py
"""

import os
import sys


def test_required_files():
    """Check that important project files exist."""
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
    ]

    missing = [file for file in required_files if not os.path.exists(file)]

    assert not missing, f"Missing required files: {missing}"


def test_app_import():
    """Check that app.py can be imported without a syntax error."""
    import py_compile

    py_compile.compile("app.py", doraise=True)


def test_readme_exists():
    """Check that README is not empty."""
    assert os.path.getsize("README.md") > 0, "README.md is empty"


def test_requirements_exists():
    """Check that requirements.txt is not empty."""
    assert os.path.getsize("requirements.txt") > 0, "requirements.txt is empty"


def run_tests():
    tests = [
        test_required_files,
        test_app_import,
        test_readme_exists,
        test_requirements_exists,
    ]

    passed = 0

    print("\n🧪 My RAG AI — Basic Tests\n")

    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as error:
            print(f"❌ {test.__name__}: {error}")

    print(f"\nResult: {passed}/{len(tests)} tests passed.")

    if passed == len(tests):
        print("🎉 All tests passed!")
        return 0

    print("⚠️ Some tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(run_tests())