#!/usr/bin/env python3
"""
LeetCode Assistant CLI

A command-line interface for the AI-powered LeetCode problem solver.
"""

import sys
import os
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.state import WorkflowState
from src.agents.problem_analyzer import analyze_problem

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()


class LeetCodeAssistant:
    """Main CLI application class."""
    
    def __init__(self):
        self.console = console
        
    def display_banner(self):
        """Display the application banner."""
        banner_text = Text()
        banner_text.append("🚀 LeetCode Assistant", style="bold blue")
        banner_text.append("\n")
        banner_text.append("AI-Powered Problem Solver", style="italic cyan")
        
        panel = Panel(
            banner_text,
            title="Welcome",
            title_align="center",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        self.console.print()
    
    def get_problem_input(self, problem_text: Optional[str] = None) -> str:
        """Get problem input from user."""
        if problem_text:
            return problem_text
        
        self.console.print("📝 [bold cyan]Enter your LeetCode problem:[/bold cyan]")
        self.console.print("   [dim]Paste the problem text, then press Enter twice to finish[/dim]")
        self.console.print("   [dim]Or type 'demo' to use a sample problem[/dim]")
        self.console.print()
        
        lines = []
        empty_line_count = 0
        
        while True:
            try:
                line = input()
                if line.strip().lower() == 'demo':
                    return self.get_demo_problem()
                    
                if line.strip() == "":
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                else:
                    empty_line_count = 0
                    
                lines.append(line)
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Cancelled by user[/yellow]")
                return ""
        
        return "\n".join(lines).strip()
    
    def get_demo_problem(self) -> str:
        """Return a demo problem for testing."""
        return """1. Two Sum

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

Constraints:
2 <= nums.length <= 10^4
-10^9 <= nums[i] <= 10^9
-10^9 <= target <= 10^9
Only one valid answer exists."""
    
    def display_analysis_results(self, state: WorkflowState):
        """Display problem analysis results in a formatted way."""
        if not state.problem:
            self.console.print("[red]❌ No analysis results available[/red]")
            return
        
        problem = state.problem
        
        # Title and basic info
        title_text = Text()
        if problem.number:
            title_text.append(f"{problem.number}. ", style="dim")
        title_text.append(problem.title or "Untitled Problem", style="bold blue")
        
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column(style="cyan", width=12)
        info_table.add_column()
        
        info_table.add_row("📊 Difficulty:", str(problem.difficulty.value) if problem.difficulty else "Unknown")
        info_table.add_row("🏷️  Tags:", ", ".join([tag.value for tag in problem.tags]) if problem.tags else "None")
        info_table.add_row("📝 Examples:", str(len(problem.examples)))
        info_table.add_row("⚙️  Constraints:", str(len(problem.constraints)))
        
        # Create main panel
        main_panel = Panel(
            info_table,
            title=title_text,
            title_align="left",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(main_panel)
        
        # Display examples if available
        if problem.examples:
            self.console.print("\n[bold cyan]📚 Examples:[/bold cyan]")
            for i, example in enumerate(problem.examples, 1):
                example_table = Table(show_header=False, box=None, padding=(0, 1))
                example_table.add_column(style="yellow", width=10)
                example_table.add_column()
                
                example_table.add_row("Input:", example.input)
                example_table.add_row("Output:", example.expected_output)
                if example.explanation:
                    example_table.add_row("Explanation:", example.explanation)
                
                example_panel = Panel(
                    example_table,
                    title=f"Example {i}",
                    border_style="yellow",
                    padding=(0, 1)
                )
                self.console.print(example_panel)
        
        # Display constraints if available
        if problem.constraints:
            self.console.print("\n[bold cyan]⚙️ Constraints:[/bold cyan]")
            constraint_text = "\n".join(f"• {constraint}" for constraint in problem.constraints)
            constraint_panel = Panel(
                constraint_text,
                border_style="blue",
                padding=(1, 2)
            )
            self.console.print(constraint_panel)
    
    def display_progress(self, state: WorkflowState):
        """Display workflow progress."""
        progress_data = state.get_progress_summary()
        
        progress_table = Table(title="🔄 Workflow Progress")
        progress_table.add_column("Phase", style="cyan", width=15)
        progress_table.add_column("Status", width=10)
        
        for phase, completed in progress_data.items():
            if phase == "overall":
                continue
            status = "✅ Done" if completed else "⏳ Pending"
            style = "green" if completed else "yellow"
            progress_table.add_row(phase.title(), status, style=style)
        
        self.console.print(progress_table)
    
    def run_analysis(self, problem_text: str) -> WorkflowState:
        """Run the problem analysis with progress indicators."""
        # Create initial state
        state = WorkflowState(raw_problem_text=problem_text)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            
            # Problem Analysis
            task = progress.add_task("🔍 Analyzing problem structure...", total=1)
            
            try:
                state = analyze_problem(state)
                progress.update(task, completed=1, description="✅ Problem analysis complete")
            except Exception as e:
                progress.update(task, completed=1, description=f"❌ Analysis failed: {e}")
                state.add_error(f"Analysis failed: {e}")
        
        return state


@click.group()
@click.version_option(version="0.1.0", prog_name="LeetCode Assistant")
def cli():
    """🚀 AI-Powered LeetCode Problem Solver
    
    Analyze problems, generate strategies, write code, and get explanations
    using advanced AI agents powered by LangGraph.
    """
    pass


@cli.command()
@click.option('--problem', '-p', help='Problem text (or use interactive mode)')
@click.option('--file', '-f', type=click.Path(exists=True), help='Read problem from file')
@click.option('--demo', is_flag=True, help='Use demo problem for testing')
def analyze(problem: Optional[str], file: Optional[str], demo: bool):
    """🔍 Analyze a LeetCode problem structure and classify it."""
    
    assistant = LeetCodeAssistant()
    assistant.display_banner()
    
    # Get problem text
    if demo:
        problem_text = assistant.get_demo_problem()
        console.print("[green]📋 Using demo problem: Two Sum[/green]\n")
    elif file:
        problem_text = Path(file).read_text()
        console.print(f"[green]📁 Loaded problem from: {file}[/green]\n")
    elif problem:
        problem_text = problem
    else:
        problem_text = assistant.get_problem_input()
    
    if not problem_text.strip():
        console.print("[red]❌ No problem text provided[/red]")
        return
    
    # Run analysis
    console.print("[bold blue]🚀 Starting Problem Analysis...[/bold blue]\n")
    
    try:
        state = assistant.run_analysis(problem_text)
        
        console.print()
        
        # Display results
        if state.analysis_complete and not state.errors:
            console.print("[green]🎉 Analysis Complete![/green]\n")
            assistant.display_analysis_results(state)
        else:
            console.print("[red]❌ Analysis encountered issues:[/red]")
            for error in state.errors:
                console.print(f"   • {error}")
        
        # Show progress
        console.print()
        assistant.display_progress(state)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Analysis cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {e}[/red]")


@cli.command()
def interactive():
    """🎮 Start interactive mode with guided problem solving."""
    
    assistant = LeetCodeAssistant()
    assistant.display_banner()
    
    console.print("[bold cyan]🎮 Interactive Mode[/bold cyan]")
    console.print("This mode will guide you through the complete problem-solving workflow.\n")
    
    # Get problem
    problem_text = assistant.get_problem_input()
    if not problem_text.strip():
        console.print("[red]❌ No problem text provided[/red]")
        return
    
    try:
        # Step 1: Analysis
        console.print("[bold blue]Step 1: Problem Analysis[/bold blue]")
        state = assistant.run_analysis(problem_text)
        
        if state.analysis_complete:
            assistant.display_analysis_results(state)
            
            # Future: Add more steps here
            console.print("\n[yellow]🚧 Strategy, Coding, Testing, and Explanation phases coming soon![/yellow]")
            console.print("For now, you have a complete problem analysis. 🎉")
        else:
            console.print("[red]❌ Analysis failed. Please check your problem text and try again.[/red]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interactive mode cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error in interactive mode: {e}[/red]")


@cli.command()
def config():
    """⚙️ Show configuration and API key status."""
    
    console.print("[bold cyan]⚙️ Configuration Status[/bold cyan]\n")
    
    # Check API keys
    config_table = Table()
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Status", style="green")
    config_table.add_column("Value", style="dim")
    
    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_status = "✅ Set" if openai_key else "❌ Missing"
    openai_value = f"sk-...{openai_key[-4:]}" if openai_key else "Not configured"
    config_table.add_row("OpenAI API Key", openai_status, openai_value)
    
    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_status = "✅ Set" if anthropic_key else "❌ Missing"
    anthropic_value = f"sk-...{anthropic_key[-4:]}" if anthropic_key else "Not configured"
    config_table.add_row("Anthropic API Key", anthropic_status, anthropic_value)
    
    # LangSmith
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    langsmith_status = "✅ Set" if langsmith_key else "❌ Missing" 
    langsmith_value = f"ls__...{langsmith_key[-4:]}" if langsmith_key else "Not configured"
    config_table.add_row("LangSmith API Key", langsmith_status, langsmith_value)
    
    # Tavily
    tavily_key = os.getenv("TAVILY_API_KEY")
    tavily_status = "✅ Set" if tavily_key else "❌ Missing"
    tavily_value = f"tvly-...{tavily_key[-4:]}" if tavily_key else "Not configured"
    config_table.add_row("Tavily API Key", tavily_status, tavily_value)
    
    console.print(config_table)
    
    # Provider settings
    console.print(f"\n[cyan]🤖 AI Provider:[/cyan] {os.getenv('AI_PROVIDER', 'openai')}")
    console.print(f"[cyan]🧠 AI Model:[/cyan] {os.getenv('AI_MODEL', 'gpt-4o-mini')}")


def main():
    """Entry point for the CLI application."""
    cli()


if __name__ == "__main__":
    main()