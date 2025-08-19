"""
State management for the LeetCode Assistant workflow.

This module defines the shared state structure that gets passed between agents
in the LangGraph workflow. Each agent can read from and write to this state.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DifficultyLevel(str, Enum):
    """LeetCode problem difficulty levels."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class ProblemType(str, Enum):
    """Common LeetCode problem categories."""
    ARRAY = "Array"
    STRING = "String"
    LINKED_LIST = "Linked List"
    TREE = "Tree"
    GRAPH = "Graph"
    DYNAMIC_PROGRAMMING = "Dynamic Programming"
    GREEDY = "Greedy"
    BACKTRACKING = "Backtracking"
    BINARY_SEARCH = "Binary Search"
    SORTING = "Sorting"
    HASH_TABLE = "Hash Table"
    TWO_POINTERS = "Two Pointers"
    SLIDING_WINDOW = "Sliding Window"
    STACK = "Stack"
    QUEUE = "Queue"
    HEAP = "Heap"
    MATH = "Math"
    OTHER = "Other"


class TestCase(BaseModel):
    """A single test case for the problem."""
    input: str = Field(..., description="Input for the test case")
    expected_output: str = Field(..., description="Expected output")
    explanation: Optional[str] = Field(None, description="Optional explanation")


class ProblemDetails(BaseModel):
    """Structured information about the LeetCode problem."""
    title: Optional[str] = Field(None, description="Problem title")
    number: Optional[int] = Field(None, description="Problem number")
    difficulty: Optional[DifficultyLevel] = Field(None, description="Problem difficulty")
    description: Optional[str] = Field(None, description="Problem description")
    constraints: List[str] = Field(default_factory=list, description="Problem constraints")
    examples: List[TestCase] = Field(default_factory=list, description="Example test cases")
    tags: List[ProblemType] = Field(default_factory=list, description="Problem categories/tags")


class SolutionApproach(BaseModel):
    """A potential solution approach."""
    name: str = Field(..., description="Name of the approach")
    description: str = Field(..., description="Description of the approach")
    time_complexity: Optional[str] = Field(None, description="Time complexity (e.g., O(n))")
    space_complexity: Optional[str] = Field(None, description="Space complexity (e.g., O(1))")
    pros: List[str] = Field(default_factory=list, description="Advantages of this approach")
    cons: List[str] = Field(default_factory=list, description="Disadvantages of this approach")
    difficulty_to_implement: Optional[str] = Field(None, description="Implementation difficulty")


class CodeSolution(BaseModel):
    """Generated code solution."""
    language: str = Field(default="python", description="Programming language")
    code: str = Field(..., description="The actual code implementation")
    approach_used: str = Field(..., description="Which approach this implements")
    key_insights: List[str] = Field(default_factory=list, description="Key programming insights")


class TestResult(BaseModel):
    """Result of running a test case."""
    test_case: TestCase = Field(..., description="The test case that was run")
    actual_output: str = Field(..., description="Actual output from the code")
    passed: bool = Field(..., description="Whether the test passed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if test failed")


class Explanation(BaseModel):
    """Detailed explanation of the solution."""
    algorithm_explanation: str = Field(..., description="How the algorithm works")
    complexity_analysis: str = Field(..., description="Time and space complexity analysis")
    key_concepts: List[str] = Field(default_factory=list, description="Important concepts used")
    alternative_approaches: List[str] = Field(default_factory=list, description="Other possible approaches")
    common_pitfalls: List[str] = Field(default_factory=list, description="Common mistakes to avoid")


class WorkflowState(BaseModel):
    """
    The complete state object that flows through the LangGraph workflow.
    
    Each agent can read from and update relevant parts of this state.
    """
    
    # Input
    raw_problem_text: str = Field(..., description="Original problem text from user")
    
    # Problem Analysis (populated by ProblemAnalyzer)
    problem: Optional[ProblemDetails] = Field(None, description="Structured problem information")
    analysis_complete: bool = Field(False, description="Whether problem analysis is done")
    
    # Strategy Development (populated by SolutionStrategist) 
    potential_approaches: List[SolutionApproach] = Field(
        default_factory=list, description="All possible solution approaches"
    )
    selected_approach: Optional[SolutionApproach] = Field(
        None, description="The chosen approach to implement"
    )
    strategy_complete: bool = Field(False, description="Whether strategy phase is done")
    
    # Code Generation (populated by CodeGenerator)
    solution: Optional[CodeSolution] = Field(None, description="Generated code solution")
    code_complete: bool = Field(False, description="Whether code generation is done")
    
    # Testing (populated by TestRunner)
    test_results: List[TestResult] = Field(
        default_factory=list, description="Results from running test cases"
    )
    all_tests_passed: bool = Field(False, description="Whether all tests passed")
    testing_complete: bool = Field(False, description="Whether testing phase is done")
    
    # Explanation (populated by Explainer)
    explanation: Optional[Explanation] = Field(None, description="Detailed solution explanation")
    explanation_complete: bool = Field(False, description="Whether explanation is done")
    
    # Workflow metadata
    current_agent: Optional[str] = Field(None, description="Currently active agent")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def add_error(self, error: str) -> None:
        """Add an error to the error list."""
        self.errors.append(error)
    
    def is_complete(self) -> bool:
        """Check if the entire workflow is complete."""
        return (
            self.analysis_complete and
            self.strategy_complete and 
            self.code_complete and
            self.testing_complete and
            self.explanation_complete
        )
    
    def get_progress_summary(self) -> Dict[str, bool]:
        """Get a summary of workflow progress."""
        return {
            "analysis": self.analysis_complete,
            "strategy": self.strategy_complete,
            "coding": self.code_complete,
            "testing": self.testing_complete,
            "explanation": self.explanation_complete,
            "overall": self.is_complete()
        }