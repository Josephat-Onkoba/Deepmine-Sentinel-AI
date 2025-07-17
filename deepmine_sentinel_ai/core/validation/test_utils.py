"""
Test utilities for validation scripts

Provides base classes and utilities for testing and validation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import traceback


@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ValidationTest:
    """Base class for validation tests"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    def run(self, context: Dict[str, Any]) -> TestResult:
        """
        Run the validation test
        
        Args:
            context: Test context and shared data
            
        Returns:
            TestResult with test outcome
        """
        raise NotImplementedError("Subclasses must implement run method")


class ValidationSuite:
    """Collection of validation tests"""
    
    def __init__(self, name: str):
        self.name = name
        self.tests: List[ValidationTest] = []
        self.context: Dict[str, Any] = {}
    
    def add_test(self, test: ValidationTest):
        """Add a test to the suite"""
        self.tests.append(test)
    
    def setup(self):
        """Setup the test environment (override in subclasses)"""
        pass
    
    def teardown(self):
        """Cleanup after tests (override in subclasses)"""
        pass
    
    def run(self) -> List[TestResult]:
        """
        Run all tests in the suite
        
        Returns:
            List of TestResult objects
        """
        results = []
        
        try:
            # Setup test environment
            self.setup()
            
            # Prepare context
            self.context.update({
                'suite_name': self.name,
                'timestamp': datetime.now(),
                'cleanup_files': []
            })
            
            # Run each test
            for test in self.tests:
                print(f"\nRunning: {test.name}")
                try:
                    start_time = datetime.now()
                    result = test.run(self.context)
                    end_time = datetime.now()
                    
                    result.test_name = test.name
                    result.execution_time = (end_time - start_time).total_seconds()
                    
                    results.append(result)
                    
                except Exception as e:
                    error_result = TestResult(
                        test_name=test.name,
                        passed=False,
                        message=f"Test execution failed: {str(e)}",
                        details={'error': traceback.format_exc()},
                        execution_time=0.0
                    )
                    results.append(error_result)
        
        finally:
            # Cleanup
            self.teardown()
        
        return results


def assert_equals(actual, expected, message: str = ""):
    """Assert that two values are equal"""
    if actual != expected:
        raise AssertionError(f"{message} Expected {expected}, got {actual}")


def assert_true(condition: bool, message: str = ""):
    """Assert that condition is true"""
    if not condition:
        raise AssertionError(f"{message} Expected True, got False")


def assert_false(condition: bool, message: str = ""):
    """Assert that condition is false"""
    if condition:
        raise AssertionError(f"{message} Expected False, got True")


def assert_not_none(value, message: str = ""):
    """Assert that value is not None"""
    if value is None:
        raise AssertionError(f"{message} Expected non-None value, got None")


def assert_isinstance(obj, expected_type, message: str = ""):
    """Assert that object is instance of expected type"""
    if not isinstance(obj, expected_type):
        raise AssertionError(f"{message} Expected {expected_type}, got {type(obj)}")


def assert_in(item, container, message: str = ""):
    """Assert that item is in container"""
    if item not in container:
        raise AssertionError(f"{message} Expected {item} to be in {container}")


def assert_greater(value, minimum, message: str = ""):
    """Assert that value is greater than minimum"""
    if value <= minimum:
        raise AssertionError(f"{message} Expected {value} > {minimum}")


def assert_less(value, maximum, message: str = ""):
    """Assert that value is less than maximum"""
    if value >= maximum:
        raise AssertionError(f"{message} Expected {value} < {maximum}")
