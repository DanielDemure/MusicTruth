"""
MusicTruth 2.0 - Advanced AI Music Detection & Forensic Tool
"""

import sys
import os
import argparse
from typing import List, Optional
import time
from src.utils.logger import logger


# Remove the sys.path hack that breaks package structure
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Relative imports for package execution (python -m src.main)
    from .config import config, AnalysisMode
    from .layers.orchestration.history import history_manager
    from .layers.input.handler import get_input_handler, AudioSource
    from .layers.analysis.core import Analyzer
    from .layers.reporting.generator import MultiFormatReporter
    from .layers.orchestration.llm.client import LLMClient
    from .layers.orchestration.llm.agents import CriticAgent, PublicReporterAgent
    from .layers.orchestration.llm.researcher import ResearcherAgent
    from .layers.analysis.comparator import CrossCheckComparator
except ImportError:
    # Fallback for script execution (python src/main.py) - though discouraged
    # If we are running as script, we need to fix path to see 'src' package
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import config, AnalysisMode
    from src.layers.orchestration.history import history_manager
    from src.layers.input.handler import get_input_handler, AudioSource
    from src.layers.analysis.core import Analyzer
    from src.layers.reporting.generator import MultiFormatReporter
    from src.layers.orchestration.llm.client import LLMClient
    from src.layers.orchestration.llm.agents import CriticAgent, PublicReporterAgent
    from src.layers.orchestration.llm.researcher import ResearcherAgent
    from src.layers.analysis.comparator import CrossCheckComparator

def main():
    parser = argparse.ArgumentParser(description="MusicTruth 2.0: Advanced AI Music Forensics")
    
    # Input/Output
    parser.add_argument("--input", "-i", help="Input file, directory, or URL")
    parser.add_argument("--project", "-p", default="Default_Project", help="Project name for history tracking")
    parser.add_argument("--mode", "-m", choices=[m.value for m in AnalysisMode], default="standard", help="Analysis mode")
    
    # Multi-source
    parser.add_argument("--group-id", help="Manually specify a Group ID for multi-source verification")
    
    # LLM Options
    parser.add_argument("--llm-provider", default=None, choices=["openai", "anthropic", "gemini", "deepseek", "ollama", "lm_studio"], help="LLM Provider")
    parser.add_argument("--llm-model", help="Specific model name")
    parser.add_argument("--llm-key", help="API Key for LLM")
    
    # Reporting
    parser.add_argument("--report-formats", default="html,json", help="Comma-separated output formats (pdf,html,json,csv)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive wizard")
    
    args = parser.parse_args()
    
    # 0. Check for Interactive Mode
    if args.interactive or (not args.input and len(sys.argv) == 1):
        try:
            # Import wizard lazily
            try:
                from .ui.wizard import run_wizard
            except ImportError:
                from src.ui.wizard import run_wizard
                
            print("🚀 Launching Interactive Wizard...")
            args = run_wizard()
        except ImportError as e:
            print(f"Error loading UI wizard: {e}")
            print("Please install 'rich' and 'questionary' or provide CLI arguments.")
            return

    # 1. Initialize System
    print(f"🎵 MusicTruth 2.0 | Mode: {args.mode.upper()} | Project: {args.project}")
    
    # Create Session
    artist = getattr(args, 'artist', None)
    album = getattr(args, 'album', None)
    session_dir = history_manager.create_session(args.project, artist=artist, album=album)
    print(f"📂 Session created: {session_dir}")
    
    # 2. Handle Inputs
    input_handler = get_input_handler(config.paths.input_dir)
    
    # Check if we have a list from wizard or single arg from CLI
    if hasattr(args, 'input_list') and args.input_list:
        for inp in args.input_list:
            if inp['type'] == 'URL':
                input_handler.add_source_url(inp['value'], group_id=inp['group_id'])
            else:
                # File or Folder
                path = inp['value']
                g_id = inp['group_id']
                if os.path.isdir(path):
                    # For folder, we add files but grouping might be tricky if one group_id for whole folder
                    # For now assume group_id applies to all (e.g. album)
                    files = input_handler.scan_directory_path(path) # Need to implement this helper
                    input_handler.add_sources_from_paths(files, group_id=g_id)
                else:
                    input_handler.add_sources_from_paths([path], group_id=g_id)
                    
    elif args.input:
        # Legacy CLI argument
        if str(args.input).startswith(("http", "www")):
            input_handler.add_source_url(args.input, group_id=args.group_id)
        elif os.path.exists(args.input):
            if os.path.isdir(args.input):
                # We need to manually scan here since input_handler scan defaults to internal input dir
                # Let's just use glob
                import glob
                files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('.mp3', '.wav', '.flac'))] 
                input_handler.add_sources_from_paths(files)
            else:
                input_handler.add_sources_from_paths([args.input], group_id=args.group_id)
    else:
        print("Error: No input provided.")
        return

    # Download remote sources if any
    print("⬇️  Preparing sources...")
    
    # User requested downloads go to Input Folder with Artist/Song info
    # We'll creating a folder structure: input / Project_Name
    # This persists the files for future use
    project_input_dir = os.path.join(config.paths.input_dir, args.project.replace(" ", "_"))
    if not os.path.exists(project_input_dir):
        os.makedirs(project_input_dir)
        print(f"📂 Created input directory: {project_input_dir}")
    
    # Retrieve files (downloads happen here if URL)
    downloaded_items = input_handler.download_remote_sources(project_input_dir)
    
    ready_sources = []
    for source, local_path in downloaded_items:
        # Create a proxy source object or just use the source but analyzed on local_path
        # We need to tell the analyzer to use local_path, but maybe keep original metadata
        # Let's create a new AudioSource pointing to the file, but copying metadata
        # Inject reviewed metadata if available
        meta = {}
        if hasattr(args, 'metadata_reviewed') and args.metadata_reviewed:
            # Match by filename
            fname = os.path.basename(local_path)
            for m in args.metadata_reviewed:
                if m['filename'] == fname:
                    meta = m
                    break
        
        new_source = AudioSource(
            path_or_url=local_path,
            source_type='file', # It is now a local file
            group_id=source.group_id,
            metadata=meta or source.metadata or {}
        )
        if 'original_url' not in new_source.metadata:
            new_source.metadata['original_url'] = source.path_or_url
        ready_sources.append(new_source)
    
    if not ready_sources:
        print("❌ No valid audio sources ready (downloads failed or no inputs).")
        return

    # 3. Initialize Engines
    analyzer = Analyzer()
    
    # Initialize LLM Agents if provider selected
    critic_agent = None
    researcher_agent = None
    reporter_agent = None
    
    if args.llm_provider:
        print(f"🧠 Initializing LLM ({args.llm_provider})...")
        # Handle base_url if present
        base_url = getattr(args, 'llm_base_url', None)
        
        llm_client = LLMClient(
            provider=args.llm_provider, 
            model=args.llm_model, 
            api_key=args.llm_key,
            base_url=base_url
        )
        if llm_client.check_availability():
            researcher_agent = ResearcherAgent(llm_client)
            critic_agent = CriticAgent(llm_client)
            reporter_agent = PublicReporterAgent(llm_client)
        else:
            print("⚠️ LLM Client failed to initialize. Proceeding without AI intelligence.")

    # 4. Run Analysis Loop
    all_results = {}
    
    for source in ready_sources:
        print(f"🔍 Analyzing: {source.path_or_url}")
        
        # Determine features based on mode
        # analyzer.analyze_file(file, mode)
        try:
            # Placeholder for the actual call - we need to refactor analyzer.py to expose this cleanly
            results = analyzer.analyze_audio(source.path_or_url, mode=AnalysisMode(args.mode))
            
            # Enrich with Metadata (LLM)
            if researcher_agent:
                print("   🤖 Researching context...")
                # We need artist/title from metadata or filename
                # For now using filename
                fname = os.path.basename(source.path_or_url)
                context = researcher_agent.research_context(artist="Unknown", title=fname)
                results['context'] = context
                
            # Critical Review (LLM)
            if critic_agent and 'context' in results:
                print("   🤔 AI Critic reviewing...")
                critique = critic_agent.critique(results, results['context'])
                results['critique'] = critique
                
            all_results[source.path_or_url] = results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.debug(e, exc_info=True)

    # 5. Cross-Check (if Multiple Sources)
    comparator = CrossCheckComparator()
    # Group sources by ID and check
    # cross_check_results = comparator.compare(...)
    
    # 6. Generate Reports
    print("📄 Generating Reports...")
    reporter = MultiFormatReporter(session_dir)
    
    for src_path, res in all_results.items():
        # Generate Public Report if Agents available
        if reporter_agent and 'critique' in res:
             public_text = reporter_agent.write_report(
                 technical_data=str(res.get('ai_probability', 'N/A')), # simplified
                 critique=res['critique'],
                 context=res.get('context', '')
             )
             res['public_report_content'] = public_text
             
        # Import style enum
        from .layers.reporting.generator import ReportStyle
        style_str = getattr(args, 'report_style', 'combined')
        try:
            style = ReportStyle(style_str)
        except ValueError:
            style = ReportStyle.COMBINED
            
        reporter.generate(res, output_formats=args.report_formats.split(','), style=style)
        
    # Save raw data via HistoryManager
    history_manager.save_results(all_results, filename="full_session_data.json")
    
    print(f"✅ Analysis Complete! Results saved to: {session_dir}")

if __name__ == "__main__":
    main()
