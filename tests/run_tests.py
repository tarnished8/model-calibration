"""
Test runner script for the model calibration package.

This script runs all unit tests and provides a summary of results.
"""

import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_all_tests():
    """Run all unit tests and return results."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    return result


def main():
    """Main function to run tests."""
    print("=" * 60)
    print("Model Calibration Package - Test Suite")
    print("=" * 60)
    
    # Run tests
    result = run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Return exit code
    if result.failures or result.errors:
        print(f"\nSome tests failed. Exit code: 1")
        return 1
    else:
        print(f"\nAll tests passed! Exit code: 0")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
