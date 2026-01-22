"""
Interactive Wizard for MusicTruth.

Uses Questionary and Rich to guide the user through setting up an analysis session.
Supports "Back" navigation and Multi-Source inputs.
"""

from typing import Dict, Any, List, Optional
import os
import argparse
import sys

# Load config to access API defaults
try:
    from ..config import config
except ImportError:
    from src.config import config

try:
    import questionary
    from questionary import Choice
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    print("Warning: 'rich' and 'questionary' not installed. Interactive mode will be basic.")

def run_wizard() -> argparse.Namespace:
    """
    Run the interactive wizard loop.
    """
    if not UI_AVAILABLE:
        return _run_basic_wizard()

    console = Console()
    
    # Session State
    state = {
        "project": "My_Analysis_Project",
        "inputs": [], # List[Dict] {type, value, group_id}
        "mode": config.api.default_analysis_mode,
        "use_llm": bool(config.api.gemini_api_key or config.api.openai_api_key or config.api.anthropic_api_key),
        "llm_config": {
            "provider": config.api.default_llm_provider,
            "model": None,
            "key": None,
            "base_url": None
        },
        "report_formats": config.api.default_output_formats
    }
    
    steps = [
        "Welcome", "Project", "Inputs", "MetadataReview", "AnalysisMode", "LLM", 
        "Reporting", "ReportStyle", "SystemCheck", "Confirmation"
    ]
    
    current_step_idx = 0
    
    while 0 <= current_step_idx < len(steps):
        step = steps[current_step_idx]
        
        # Clear screen for cleaner UI? 
        # os.system('cls' if os.name == 'nt' else 'clear') 
        # Keeping history is better for context usually.
        
        console.rule(f"[bold blue]{step}[/bold blue]")
        
        next_action = "next"
        
        if step == "Welcome":
            console.print(Panel.fit(
                "[bold cyan]MusicTruth 2.0[/bold cyan]\n"
                "[dim]Advanced AI Music Forensic Tool[/dim]",
                subtitle="Interactive Mode"
            ))
            next_action = "next"
            
        elif step == "Project":
            ans = questionary.text(
                "Project Name:",
                default=state["project"],
                instruction="(Press Ctrl+C to go back)"
            ).ask()
            
            if ans is None:  # User pressed Ctrl+C
                next_action = "back"
            else:
                state["project"] = ans
            
        elif step == "Inputs":
            # Loop for multiple inputs
            while True:
                # Show current inputs
                if state["inputs"]:
                    rprint("\n[bold]Current Sources:[/bold]")
                    for idx, inp in enumerate(state["inputs"]):
                        rprint(f"  {idx+1}. [{inp['type']}] {inp['value']}")
                    rprint("")
                
                action = questionary.select(
                    "Manage Inputs:",
                    choices=[
                        "Add Source",
                        Choice("Finished adding sources", disabled=len(state["inputs"]) == 0),
                        Choice("Remove last source", disabled=len(state["inputs"]) == 0),
                        "Back to previous step"
                    ]
                ).ask()
                
                if action == "Back to previous step":
                    next_action = "back"
                    break
                elif action == "Finished adding sources":
                    break
                elif action == "Remove last source":
                    state["inputs"].pop()
                elif action == "Add Source":
                    _add_source_flow(state)
                    
        elif step == "MetadataReview":
            # Scan files with mutagen
            from .metadata_scanner import MetadataScanner
            from ..layers.input.handler import InputHandler
            
            scanner = MetadataScanner()
            input_handler = InputHandler(os.getcwd()) # Base dir not critical for scan_directory_path
            
            # Flatten inputs into a list of file paths
            file_paths = []
            for inp in state["inputs"]:
                if inp["type"] == "Local Folder":
                    file_paths.extend(input_handler.scan_directory_path(inp["value"]))
                elif inp["type"] == "Local File":
                    file_paths.append(inp["value"])
            
            if not file_paths:
                rprint("[yellow]⚠️  No audio files found in the provided sources.[/yellow]")
                if not questionary.confirm("Continue anyway?").ask():
                    next_action = "back"
                    continue
            
            tracks = scanner.scan_files(file_paths)
            
            # Auto-fetch if configured (and if user wants to for this session)
            if any(file_paths):
                if questionary.confirm("Attempt to auto-fetch missing metadata from MusicBrainz/Spotify?", default=True).ask():
                    scanner.auto_fetch_metadata(tracks)
            
            # Interactive review
            reviewed = scanner.interactive_review(tracks)
            state["metadata_reviewed"] = reviewed
            
            # Use artist/album from first track as default for project if not set/generic
            if reviewed:
                artist = reviewed[0].get("artist")
                album = reviewed[0].get("album")
                if artist and artist != "Unknown Artist":
                    state["artist"] = artist
                if album and album != "Unknown Album":
                    state["album"] = album

        elif step == "AnalysisMode":
            mode_choices = [
                {"name": "Quick (30s) - Basic spectral & tempo checks", "value": "quick"},
                {"name": "Standard (2m) - Comprehensive analysis", "value": "standard"},
                {"name": "Deep (10m) - Source separation + full feature set", "value": "deep"},
                {"name": "Forensic (30m+) - Exhaustive scan + Fingerprinting", "value": "forensic"},
                Choice("← Back", value="BACK")
            ]
            
            ans = questionary.select(
                 "Select Analysis Mode:",
                 choices=mode_choices,
                 default=Choice(state["mode"]) if state["mode"] != "standard" else None
            ).ask()
            
            if ans == "BACK":
                next_action = "back"
            else:
                state["mode"] = ans
                
        elif step == "LLM":
            use_llm = questionary.confirm("Enable AI Intelligence (LLM)?", default=state["use_llm"]).ask()
            state["use_llm"] = use_llm
            
            if use_llm:
                # Provider Selection
                providers = [
                    "OpenRouter", "OpenAI", "Anthropic", "Google Gemini", 
                    "DeepSeek", "Custom OpenAI-Compatible", "Local (Ollama/LM Studio)", 
                    Choice("← Back", value="BACK")
                ]
                prov = questionary.select("Select Provider:", choices=providers).ask()
                
                if prov == "BACK":
                    next_action = "back"
                else:
                    # Normalize provider
                    p_code = prov.split(" ")[0].lower() # default logic
                    if "gemini" in prov.lower(): p_code = "gemini" 
                    if "custom" in prov.lower(): p_code = "custom"
                    if "local" in prov.lower(): p_code = "ollama"
                    if "openrouter" in prov.lower(): p_code = "openrouter"
                    
                    state["llm_config"]["provider"] = p_code
                    
                    # 2b. Credentials and Base URL (Ask FIRST per user request)
                    _configure_credentials(state["llm_config"])
                    
                    # 2c. Model Selection - SKIP if already set from .env
                    if not state["llm_config"].get("model"):
                        model = _select_model_for_provider(p_code)
                        if model == "BACK":
                            # logic to retry provider selection? 
                            # simpler just to re-loop this step
                            continue 
                            
                        state["llm_config"]["model"] = model

        elif step == "Reporting":
            choices = [
                Choice("HTML", checked="html" in state["report_formats"]),
                Choice("JSON", checked="json" in state["report_formats"]),
                Choice("PDF", checked="pdf" in state["report_formats"]),
                Choice("CSV", checked="csv" in state["report_formats"]),
                Choice("← Back", value="BACK")
            ]
            
            ans = questionary.checkbox("Select output formats:", choices=choices).ask()
            
            if "BACK" in ans: 
                next_action = "back"
            else:
                state["report_formats"] = ",".join([f.lower() for f in ans if f != "BACK"])
                
        elif step == "ReportStyle":
            styles = [
                Choice("Combined (Technical + Summary) [DEFAULT]", value="combined"),
                Choice("Technical - Full feature tables & raw metrics", value="technical"),
                Choice("Human-Readable - Plain language & key findings", value="human"),
                Choice("Executive Summary - 1-page overview", value="summary"),
                Choice("Forensic Expert - Everything + raw JSON export", value="forensic"),
                Choice("← Back", value="BACK")
            ]
            
            ans = questionary.select(
                "Select Report Style:",
                choices=styles,
                default="combined"
            ).ask()
            
            if ans == "BACK":
                next_action = "back"
            else:
                state["report_style"] = ans

        elif step == "SystemCheck":
            from .validators import SystemValidator
            validator = SystemValidator()
            
            while True:
                rprint("\n[bold blue]Running System Checks...[/bold blue]")
                results = []
                
                # Run validations
                if state["use_llm"]:
                    results.append(validator.validate_llm(state["llm_config"]["provider"], state["llm_config"]))
                    
                results.append(validator.validate_spotify())
                results.append(validator.validate_musicbrainz())
                results.append(validator.validate_disk_space())
                local_paths = [inp["value"] for inp in state["inputs"] if inp["type"] in ["Local File", "Local Folder"]]
                results.append(validator.validate_inputs(local_paths))
                
                # Show results table
                table = Table(title="System Validation Results")
                table.add_column("Component", style="bold")
                table.add_column("Status", style="bold")
                table.add_column("Details")
                
                for res in results:
                    status_style = "green" if res.passed else ("yellow" if res.severity == "warning" else "red")
                    status_icon = "✅" if res.passed else ("⚠️" if res.severity == "warning" else "❌")
                    table.add_row(res.component, f"[{status_style}]{status_icon} {res.severity.upper()}[/{status_style}]", res.message)
                
                console.print(table)
                
                # Check for failures
                failures = [r for r in results if not r.passed]
                if not failures:
                    break
                
                critical_failures = [f for f in failures if f.severity == "critical"]
                
                if critical_failures:
                    rprint(f"\n[bold red]Critical validation errors found![/bold red]")
                    action = questionary.select(
                        "How would you like to proceed?",
                        choices=[
                            "Retry Checks",
                            "Edit Configuration (Back)",
                            "Cancel Analysis"
                        ]
                    ).ask()
                else:
                    # Only warnings
                    action = questionary.select(
                        "Warnings found. Proceed anyway?",
                        choices=[
                            "Continue",
                            "Retry Checks",
                            "Edit Configuration (Back)"
                        ],
                        default="Continue"
                    ).ask()
                
                if action == "Continue":
                    break
                elif action == "Retry Checks":
                    continue
                elif action == "Edit Configuration (Back)":
                    next_action = "back"
                    break
                elif action == "Cancel Analysis":
                    exit(0)

        elif step == "Confirmation":
            rprint(f"\n[bold green]Configuration Summary:[/bold green]")
            rprint(f"Project: [cyan]{state['project']}[/cyan]")
            rprint(f"Inputs:  [cyan]{len(state['inputs'])} sources[/cyan]")
            rprint(f"Mode:    [cyan]{state['mode']}[/cyan]")
            if state["use_llm"]:
                rprint(f"LLM:     [cyan]{state['llm_config']['provider']} / {state['llm_config']['model']}[/cyan]")
            
            confirm = questionary.select(
                "Start Analysis?",
                choices=["Yes", "Edit Configuration", "Cancel"]
            ).ask()
            
            if confirm == "Cancel":
                print("Cancelled.")
                exit(0)
            elif confirm == "Edit Configuration":
                current_step_idx = 1 # Go back to Project
                continue
            else:
                break # Done!

        # Navigation
        if next_action == "next":
            current_step_idx += 1
        elif next_action == "back":
            current_step_idx -= 1

    # Convert state to Namespace
    return _state_to_namespace(state)

def _add_source_flow(state):
    """Helper to add a single source."""
    typ = questionary.select(
        "Source Type:", 
        choices=["Local File", "Local Folder", "URL"]
    ).ask()
    
    val = ""
    if typ == "Local File":
        val = questionary.path("Path:").ask()
    elif typ == "Local Folder":
        val = questionary.path("Folder Path:", only_directories=True).ask()
    else:
        val = questionary.text("URL:").ask()
        
    gid = None
    if questionary.confirm("Assign a Group ID? (Useful for comparing versions)").ask():
        gid = questionary.text("Group ID (e.g. 'song1'):").ask()
        
    if val:
        state["inputs"].append({"type": typ, "value": val, "group_id": gid})

def _select_model_for_provider(provider):
    """Show model menu for provider."""
    models_map = {
        "openai": ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "Custom..."],
        "anthropic": ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620", "claude-3-sonnet-20240229", "Custom..."],
        "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "Custom..."],
        "deepseek": ["deepseek-chat", "deepseek-coder", "Custom..."],
        "openrouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "meta-llama/llama-3-70b-instruct", "Custom..."],
        "ollama": ["llama3", "mistral", "gemma", "Custom..."],
        "custom": ["Custom..."]
    }
    
    choices = models_map.get(provider, ["Custom..."]) + ["Back"]
    
    ans = questionary.select(f"Select Model for {provider}:", choices=choices).ask()
    
    if ans == "Back": return "BACK"
    if ans == "Custom...":
        return questionary.text("Enter Model Name:").ask()
    return ans

def _configure_credentials(config_dict):
    """Get Key/BaseURL."""
    provider = config_dict["provider"]
    
    # Base URL for custom/openrouter
    if provider == "custom":
        config_dict["base_url"] = questionary.text("API Base URL:", default="https://api.openai.com/v1").ask()
    elif provider == "ollama":
        config_dict["base_url"] = questionary.text("Ollama URL:", default="http://localhost:11434/v1").ask()
    
    # API Key - check .env first
    if provider not in ["ollama"]:
        # Try to get from config.api first
        env_key = None
        if provider == "gemini":
            env_key = config.api.gemini_api_key
            config_dict["model"] = config_dict.get("model") or config.api.gemini_model
        elif provider == "openai":
            env_key = config.api.openai_api_key
            config_dict["model"] = config_dict.get("model") or config.api.openai_model
        elif provider == "anthropic":
            env_key = config.api.anthropic_api_key
            config_dict["model"] = config_dict.get("model") or config.api.anthropic_model
        
        if env_key:
            use_env = questionary.confirm(f"Found {provider.upper()} credentials in .env. Use them?", default=True).ask()
            if use_env:
                config_dict["key"] = env_key
                return
        
        # Fallback to manual entry
        config_dict["key"] = questionary.password(f"Enter {provider} API Key:").ask()

def _state_to_namespace(state):
    """Convert valid state to argparse namespace."""
    # Main.py expects 'input' arg generally. 
    # With multi-source, we might need to pass a special structure.
    # We'll join them or modify main.py to accept list.
    # Let's attach the raw list to the namespace as 'input_list'
    
    ns = argparse.Namespace(
        project=state["project"],
        mode=state["mode"],
        llm_provider=state["llm_config"]["provider"] if state["use_llm"] else None,
        llm_model=state["llm_config"]["model"] if state["use_llm"] else None,
        llm_key=state["llm_config"]["key"] if state["use_llm"] else None,
        llm_base_url=state["llm_config"]["base_url"] if state["use_llm"] else None,
        report_formats=state["report_formats"],
        report_style=state.get("report_style", "combined"),
        artist=state.get("artist"),
        album=state.get("album"),
        input=None, # Legacy field
        input_list=state["inputs"], # Raw inputs list
        metadata_reviewed=state.get("metadata_reviewed", []) # Enriched metadata
    )
    return ns

def _run_basic_wizard():
    print("Interactive mode requires 'rich' and 'questionary'.")
    exit(1)
