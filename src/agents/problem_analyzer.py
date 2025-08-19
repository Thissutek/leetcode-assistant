"""
Problem Analyzer Agent

This agent takes raw problem text and performs intelligent analysis to extract
structured information, classify the problem, and provide initial insights.
"""

import os
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..graph.state import WorkflowState, ProblemDetails, DifficultyLevel, ProblemType
from ..utils.leetcode_parser import parse_leetcode_problem


class ProblemAnalyzer:
    """Agent responsible for analyzing and structuring LeetCode problems."""
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4o-mini"):
        """
        Initialize the Problem Analyzer agent.
        
        Args:
            model_provider: "openai" or "anthropic"
            model_name: Specific model to use
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self) -> Optional[object]:
        """Initialize the LLM with proper error handling for missing API keys."""
        try:
            if self.model_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("⚠️  Warning: OPENAI_API_KEY not found. LLM analysis will be skipped.")
                    return None
                return ChatOpenAI(model=self.model_name, api_key=api_key)
            
            elif self.model_provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    print("⚠️  Warning: ANTHROPIC_API_KEY not found. LLM analysis will be skipped.")
                    return None
                return ChatAnthropic(model=self.model_name, api_key=api_key)
            
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize LLM: {e}")
            return None
    
    def analyze_problem(self, state: WorkflowState) -> WorkflowState:
        """
        Main analysis function that processes the problem and updates state.
        
        Args:
            state: Current workflow state with raw problem text
            
        Returns:
            Updated state with problem analysis
        """
        try:
            # Update current agent
            state.current_agent = "problem_analyzer"
            
            # Step 1: Basic parsing using our utility
            print("🔍 Parsing problem structure...")
            basic_problem = parse_leetcode_problem(state.raw_problem_text)
            
            # Step 2: Enhanced analysis with LLM (if available)
            if self.llm:
                print("🤖 Performing AI-enhanced analysis...")
                enhanced_problem = self._enhance_with_llm(basic_problem, state.raw_problem_text)
                state.problem = enhanced_problem
            else:
                print("📋 Using basic parsing (no LLM available)")
                state.problem = basic_problem
            
            # Step 3: Validate and finalize analysis
            self._validate_analysis(state)
            
            # Mark analysis as complete
            state.analysis_complete = True
            print("✅ Problem analysis complete!")
            
        except Exception as e:
            error_msg = f"Problem analysis failed: {str(e)}"
            state.add_error(error_msg)
            print(f"❌ {error_msg}")
        
        return state
    
    def _enhance_with_llm(self, basic_problem: ProblemDetails, raw_text: str) -> ProblemDetails:
        """Use LLM to enhance and validate the basic parsing results."""
        
        # Create analysis prompt
        system_prompt = self._create_analysis_prompt()
        human_prompt = self._create_human_prompt(basic_problem, raw_text)
        
        try:
            # Get LLM response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse LLM response and enhance problem details
            enhanced_problem = self._parse_llm_response(response.content, basic_problem)
            return enhanced_problem
            
        except Exception as e:
            print(f"⚠️  LLM enhancement failed, using basic parsing: {e}")
            return basic_problem
    
    def _create_analysis_prompt(self) -> str:
        """Create the system prompt for problem analysis."""
        return """You are an expert at analyzing LeetCode coding problems. Your task is to:

1. **Validate and improve** the basic parsing results
2. **Classify** the problem type and difficulty accurately  
3. **Identify** key algorithmic concepts and patterns
4. **Extract** additional insights not caught by basic parsing

Focus on:
- Problem classification (Array, Tree, Graph, DP, etc.)
- Difficulty assessment (Easy/Medium/Hard)
- Key algorithmic patterns and techniques needed
- Time/space complexity considerations
- Common edge cases and gotchas

Respond in a structured format that enhances the provided analysis."""
    
    def _create_human_prompt(self, basic_problem: ProblemDetails, raw_text: str) -> str:
        """Create the human prompt with problem details."""
        return f"""Please analyze this LeetCode problem:

**Raw Problem Text:**
{raw_text}

**Basic Parsing Results:**
- Title: {basic_problem.title}
- Number: {basic_problem.number}
- Detected Difficulty: {basic_problem.difficulty}
- Detected Tags: {[tag.value for tag in basic_problem.tags]}
- Examples Found: {len(basic_problem.examples)}
- Constraints Found: {len(basic_problem.constraints)}

**Please provide:**
1. **Corrected/Enhanced Classification**: Verify difficulty and tags
2. **Key Insights**: What algorithmic patterns do you see?
3. **Problem Category**: Primary and secondary problem types
4. **Complexity Hints**: Expected time/space complexity ranges
5. **Common Approaches**: What solution strategies would work?

Format your response clearly with each section marked."""
    
    def _parse_llm_response(self, response: str, basic_problem: ProblemDetails) -> ProblemDetails:
        """Parse LLM response and create enhanced problem details."""
        
        # For now, we'll enhance the basic problem with LLM insights
        # In a full implementation, you'd parse the structured LLM response
        
        # This is a simplified version - you could make this more sophisticated
        response_lower = response.lower()
        
        # Try to extract enhanced difficulty
        enhanced_difficulty = basic_problem.difficulty
        if "hard" in response_lower and "difficulty" in response_lower:
            enhanced_difficulty = DifficultyLevel.HARD
        elif "medium" in response_lower and "difficulty" in response_lower:
            enhanced_difficulty = DifficultyLevel.MEDIUM
        elif "easy" in response_lower and "difficulty" in response_lower:
            enhanced_difficulty = DifficultyLevel.EASY
        
        # Try to extract additional tags from LLM analysis
        enhanced_tags = list(basic_problem.tags)  # Start with basic tags
        
        tag_keywords = {
            ProblemType.DYNAMIC_PROGRAMMING: ["dynamic programming", "dp", "memoization"],
            ProblemType.GREEDY: ["greedy"],
            ProblemType.BACKTRACKING: ["backtrack", "backtracking"],
            ProblemType.BINARY_SEARCH: ["binary search"],
            ProblemType.TWO_POINTERS: ["two pointer", "two-pointer"],
            ProblemType.SLIDING_WINDOW: ["sliding window"],
            ProblemType.GRAPH: ["graph", "dfs", "bfs"],
            ProblemType.TREE: ["tree", "binary tree"],
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                if tag not in enhanced_tags:
                    enhanced_tags.append(tag)
        
        # Create enhanced problem details
        enhanced_problem = ProblemDetails(
            title=basic_problem.title,
            number=basic_problem.number,
            difficulty=enhanced_difficulty,
            description=basic_problem.description,
            constraints=basic_problem.constraints,
            examples=basic_problem.examples,
            tags=enhanced_tags
        )
        
        return enhanced_problem
    
    def _validate_analysis(self, state: WorkflowState) -> None:
        """Validate the analysis results and ensure completeness."""
        if not state.problem:
            raise ValueError("Problem analysis produced no results")
        
        # Ensure we have at least basic information
        if not state.problem.description and not state.problem.title:
            raise ValueError("Problem analysis failed to extract title or description")
        
        # Ensure we have examples (most LeetCode problems have them)
        if not state.problem.examples:
            print("⚠️  Warning: No examples found in problem")
        
        # Ensure we have some tags
        if not state.problem.tags:
            state.problem.tags = [ProblemType.OTHER]
        
        print(f"📊 Analysis Summary:")
        print(f"   Title: {state.problem.title}")
        print(f"   Difficulty: {state.problem.difficulty}")
        print(f"   Tags: {[tag.value for tag in state.problem.tags]}")
        print(f"   Examples: {len(state.problem.examples)}")
        print(f"   Constraints: {len(state.problem.constraints)}")


def analyze_problem(state: WorkflowState) -> WorkflowState:
    """
    Convenience function for LangGraph integration.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with problem analysis
    """
    analyzer = ProblemAnalyzer()
    return analyzer.analyze_problem(state)