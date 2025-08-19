"""
LeetCode problem parser utility.

This module provides functions to parse raw LeetCode problem text and extract
structured information like examples, constraints, and problem details.
"""

import re
from typing import List, Optional, Tuple
from ..graph.state import TestCase, ProblemDetails, DifficultyLevel, ProblemType


class LeetCodeParser:
    """Parser for LeetCode problem text."""
    
    def __init__(self):
        # Common patterns for parsing LeetCode problems
        self.example_pattern = re.compile(
            r'Example\s*(\d+)?:?\s*\n(.*?)(?=Example|\n\n|\Z)', 
            re.DOTALL | re.IGNORECASE
        )
        self.constraint_pattern = re.compile(
            r'Constraints?:?\s*\n(.*?)(?=\n\n|\Z)', 
            re.DOTALL | re.IGNORECASE
        )
        self.title_pattern = re.compile(r'^(\d+)\.\s*(.+)', re.MULTILINE)
        
    def parse_problem(self, raw_text: str) -> ProblemDetails:
        """
        Parse raw problem text and return structured problem details.
        
        Args:
            raw_text: Raw problem text from LeetCode
            
        Returns:
            ProblemDetails object with parsed information
        """
        # Extract title and number
        title, number = self._extract_title_and_number(raw_text)
        
        # Extract examples
        examples = self._extract_examples(raw_text)
        
        # Extract constraints
        constraints = self._extract_constraints(raw_text)
        
        # Extract description (everything before examples)
        description = self._extract_description(raw_text)
        
        # Try to infer difficulty and tags (basic heuristics)
        difficulty = self._infer_difficulty(raw_text)
        tags = self._infer_tags(raw_text, description)
        
        return ProblemDetails(
            title=title,
            number=number,
            difficulty=difficulty,
            description=description,
            constraints=constraints,
            examples=examples,
            tags=tags
        )
    
    def _extract_title_and_number(self, text: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract problem number and title."""
        match = self.title_pattern.search(text)
        if match:
            number = int(match.group(1))
            title = match.group(2).strip()
            return title, number
        
        # Fallback: look for title-like patterns
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a title
            if len(first_line) < 100 and not first_line.lower().startswith(('given', 'return', 'find')):
                return first_line, None
        
        return None, None
    
    def _extract_examples(self, text: str) -> List[TestCase]:
        """Extract example test cases from the problem text."""
        examples = []
        
        # Find all example blocks
        example_matches = self.example_pattern.findall(text)
        
        for example_num, example_text in example_matches:
            # Parse input/output from example text
            input_val, output_val, explanation = self._parse_example_content(example_text)
            
            if input_val and output_val:
                examples.append(TestCase(
                    input=input_val,
                    expected_output=output_val,
                    explanation=explanation
                ))
        
        return examples
    
    def _parse_example_content(self, example_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse input, output, and explanation from example text."""
        input_val = None
        output_val = None
        explanation = None
        
        # Look for Input: and Output: patterns
        input_match = re.search(r'Input:\s*(.*?)(?=Output:|Explanation:|\n\n|\Z)', 
                               example_text, re.DOTALL | re.IGNORECASE)
        if input_match:
            input_val = input_match.group(1).strip()
        
        output_match = re.search(r'Output:\s*(.*?)(?=Explanation:|\n\n|\Z)', 
                                example_text, re.DOTALL | re.IGNORECASE)
        if output_match:
            output_val = output_match.group(1).strip()
        
        explanation_match = re.search(r'Explanation:\s*(.*?)(?=\n\n|\Z)', 
                                     example_text, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        return input_val, output_val, explanation
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraint lines from the problem text."""
        constraints = []
        
        constraint_match = self.constraint_pattern.search(text)
        if constraint_match:
            constraint_text = constraint_match.group(1).strip()
            
            # Split into individual constraint lines
            lines = constraint_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.isspace():
                    # Remove bullet points or numbering
                    line = re.sub(r'^[-*•]\s*', '', line)
                    line = re.sub(r'^\d+\.\s*', '', line)
                    if line:
                        constraints.append(line)
        
        return constraints
    
    def _extract_description(self, text: str) -> Optional[str]:
        """Extract the main problem description."""
        # Find the start of examples to know where description ends
        example_match = self.example_pattern.search(text)
        
        if example_match:
            description_end = example_match.start()
            description = text[:description_end].strip()
        else:
            # If no examples found, look for constraints
            constraint_match = self.constraint_pattern.search(text)
            if constraint_match:
                description_end = constraint_match.start()
                description = text[:description_end].strip()
            else:
                # Use the whole text
                description = text.strip()
        
        # Remove title from description if present
        title_match = self.title_pattern.search(description)
        if title_match:
            description = description[title_match.end():].strip()
        
        return description if description else None
    
    def _infer_difficulty(self, text: str) -> Optional[DifficultyLevel]:
        """Try to infer difficulty level from problem text."""
        text_lower = text.lower()
        
        # Look for explicit difficulty mentions
        if 'easy' in text_lower:
            return DifficultyLevel.EASY
        elif 'medium' in text_lower:
            return DifficultyLevel.MEDIUM
        elif 'hard' in text_lower:
            return DifficultyLevel.HARD
        
        # Basic heuristics based on problem complexity indicators
        complexity_indicators = [
            'dynamic programming', 'dp', 'optimal substructure',
            'backtrack', 'recursion', 'memoization',
            'binary search', 'divide and conquer',
            'graph', 'tree traversal', 'dfs', 'bfs',
            'sliding window', 'two pointers'
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators 
                            if indicator in text_lower)
        
        if indicator_count >= 3:
            return DifficultyLevel.HARD
        elif indicator_count >= 1:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY
    
    def _infer_tags(self, text: str, description: Optional[str]) -> List[ProblemType]:
        """Infer problem tags based on keywords in the text."""
        tags = []
        full_text = (text + ' ' + (description or '')).lower()
        
        # Keyword mappings for different problem types
        tag_keywords = {
            ProblemType.ARRAY: ['array', 'list', 'nums', 'elements'],
            ProblemType.STRING: ['string', 'char', 'character', 'substring'],
            ProblemType.LINKED_LIST: ['linked list', 'listnode', 'node', 'next'],
            ProblemType.TREE: ['tree', 'binary tree', 'treenode', 'root', 'leaf'],
            ProblemType.GRAPH: ['graph', 'edge', 'vertex', 'node', 'connected'],
            ProblemType.DYNAMIC_PROGRAMMING: ['dynamic programming', 'dp', 'optimal', 'subproblem'],
            ProblemType.GREEDY: ['greedy', 'optimal choice'],
            ProblemType.BACKTRACKING: ['backtrack', 'backtracking', 'permutation', 'combination'],
            ProblemType.BINARY_SEARCH: ['binary search', 'sorted', 'log', 'divide'],
            ProblemType.SORTING: ['sort', 'sorted', 'order', 'arrange'],
            ProblemType.HASH_TABLE: ['hash', 'map', 'dictionary', 'count'],
            ProblemType.TWO_POINTERS: ['two pointer', 'left', 'right', 'pointer'],
            ProblemType.SLIDING_WINDOW: ['sliding window', 'window', 'subarray'],
            ProblemType.STACK: ['stack', 'push', 'pop', 'lifo'],
            ProblemType.QUEUE: ['queue', 'deque', 'fifo'],
            ProblemType.HEAP: ['heap', 'priority', 'minimum', 'maximum'],
            ProblemType.MATH: ['math', 'formula', 'calculate', 'sum', 'product']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                tags.append(tag)
        
        # If no tags found, default to OTHER
        if not tags:
            tags.append(ProblemType.OTHER)
        
        return tags


def parse_leetcode_problem(raw_text: str) -> ProblemDetails:
    """
    Convenience function to parse a LeetCode problem.
    
    Args:
        raw_text: Raw problem text from LeetCode
        
    Returns:
        ProblemDetails object with parsed information
    """
    parser = LeetCodeParser()
    return parser.parse_problem(raw_text)