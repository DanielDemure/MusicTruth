import os
import sys
import click
from typing import List

# Add project root to python path if running from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.input_handler import scan_files, download_from_sources
from src.analyzer import analyze_audio
from src.reporter import generate_reports

@click.command()
@click.option('--input-dir', default='input', help='Directory to scan for audio files.')
@click.option('--output-dir', default='output', help='Directory to save reports.')
@click.option('--sources', default='sources.txt', help='File containing URLs to download.')
@click.option('--download/--no-download', default=True, help='Download from sources.txt before analysis.')
def main(input_dir, output_dir, sources, download):
    """
    MusicTruth: AI Music Analysis Tool
    """
    click.echo("🎵 MusicTruth - AI Music Analysis Tool 🎵")
    click.echo("========================================")

    # 1. Download phase
    if download and os.path.exists(sources):
        if click.confirm(f"Found {sources}. Do you want to check/download new files?"):
            click.echo("Downloading/Updating from sources...")
            download_from_sources(sources, input_dir)
    
    # 2. Scan phase
    files = scan_files(input_dir)
    if not files:
        click.echo(f"No audio files found in {input_dir}. Please add files or URLs.")
        return

    click.echo(f"\nFound {len(files)} audio files.")
    # 3. Select analysis scope
    scope = click.prompt("Select analysis scope (all, select)", default="all")
    
    files_to_analyze = []
    if scope == 'all':
        files_to_analyze = files
    else:
        # Simple selection logic
        print("Available files:")
        for i, f in enumerate(files):
            print(f"{i+1}. {os.path.basename(f)}")
        indices = click.prompt("Enter file numbers to analyze (comma separated)", type=str)
        try:
            indices = [int(x.strip()) - 1 for x in indices.split(',')]
            files_to_analyze = [files[i] for i in indices if 0 <= i < len(files)]
        except:
            print("Invalid selection. Analyzing all.")
            files_to_analyze = files

    # 4. Ask for Deep Analysis (Source Separation)
    deep_analysis = click.confirm("Perform Deep Analysis? (Source Separation & Vocal Forensics) [Slow]", default=False)
    
    # 5. Ask for Album Comparison
    compare_album = False
    if len(files_to_analyze) > 1:
        compare_album = click.confirm("Perform Album Consistency Check? (Compare tracks)", default=True)

    print(f"\nStarting analysis on {len(files_to_analyze)} files...")
    
    results = []
    
    # Album Comparison
    album_results = {}
    if compare_album:
        from src.comparator import compare_album as run_comparison
        album_results = run_comparison(files_to_analyze)
        if 'outliers' in album_results:
            print(f"Album Analysis: Found {len(album_results['outliers'])} outliers.")

    with click.progressbar(files_to_analyze, label="Analyzing") as bar:
        for file_path in bar:
            print(f"Analyzing {os.path.basename(file_path)}...")
            
            # Standard Analysis
            res = analyze_audio(file_path)
            
            # Deep Analysis
            if deep_analysis:
                from src.separator import separate_audio
                from src.analyzer import check_vocal_forensics
                
                stems = separate_audio(file_path)
                if 'vocals' in stems:
                    vocal_res = check_vocal_forensics(stems['vocals'])
                    res['vocal_analysis'] = vocal_res
                    if vocal_res['score'] > 0:
                        res['ai_probability'] += 0.2 * vocal_res['score']
                        res['flags'].extend(vocal_res['flags'])
                    
                    # Cleanup
                    try:
                        import shutil
                        shutil.rmtree(os.path.dirname(stems['vocals']))
                    except:
                        pass
            
            # Add Album Context
            if compare_album and 'track_scores' in album_results:
                sim_score = album_results['track_scores'].get(file_path, 0)
                res['album_similarity'] = sim_score
                if sim_score < album_results.get('group_mean_similarity', 0) - 0.1:
                    res['flags'].append(f"Low album consistency (Sim: {sim_score:.2f})")
                    res['ai_probability'] += 0.1

            res['ai_probability'] = min(1.0, res['ai_probability'])
            results.append(res)

    print("\nGenerating reports...")
    generate_reports(results, output_dir)
    
    print(f"Reports generated at {output_dir}")

if __name__ == '__main__':
    main()
