"""
Reporter Module.

Generates multi-format reports (HTML, JSON, PDF) with templates and visualizations.
"""

import os
import json
from typing import Dict, List
import datetime
from enum import Enum

class ReportStyle(Enum):
    TECHNICAL = "technical"
    HUMAN = "human"
    SUMMARY = "summary"
    COMBINED = "combined"
    FORENSIC = "forensic"

# Try imports
try:
    import jinja2
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False
    
try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class MultiFormatReporter:
    """Generates reports in requested formats."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        
    def generate(self, results: Dict, output_formats: List[str], style: ReportStyle = ReportStyle.COMBINED):
        """Generate all requested report formats."""
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        base_filename = f"report_{timestamp}"
        
        if 'json' in output_formats:
            self._generate_json(results, base_filename)
        
        if 'html' in output_formats:
            self._generate_html(results, base_filename, style)
            
        if 'csv' in output_formats:
            self._generate_csv(results, base_filename)
            
    def _generate_json(self, results: Dict, filename: str):
        path = os.path.join(self.output_dir, filename + ".json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"   📄 JSON report saved: {path}")

    def _generate_html(self, results: Dict, filename: str, style: ReportStyle):
        path = os.path.join(self.output_dir, filename + ".html")
        
        if not JINJA_AVAILABLE:
            print("❌ Jinja2 not installed. Skipping HTML report.")
            return

        # Load Template
        try:
            from src.config import config
            from .visualizations import generate_spectrogram_plot, generate_feature_radar_chart
            
            # Map style to template filename
            style_templates = {
                ReportStyle.TECHNICAL: "report_technical.html",
                ReportStyle.HUMAN: "report_human.html",
                ReportStyle.SUMMARY: "report_summary.html",
                ReportStyle.COMBINED: "report_template.html", # Default existing
                ReportStyle.FORENSIC: "report_forensic.html"
            }
            
            template_filename = style_templates.get(style, "report_template.html")
            template_path = os.path.join(config.paths.templates_dir, template_filename)
            
            if not os.path.exists(template_path):
                # Try fallback to main template
                template_path = os.path.join(config.paths.templates_dir, "report_template.html")
                
            if not os.path.exists(template_path):
                print(f"⚠️ Template not found: {template_path}. Using fallback HTML.")
                self._generate_fallback_html(results, filename)
                return
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_str = f.read()
                
            template = jinja2.Template(template_str)
            
            # Generate visualizations
            spectrogram_html = None
            radar_html = None
            
            # Try to generate spectrogram if we have audio path
            audio_path = results.get('audio_path') or results.get('filename')
            if audio_path and os.path.exists(audio_path):
                spectrogram_html = generate_spectrogram_plot(audio_path)
            
            # Generate feature radar chart
            if 'features' in results:
                radar_html = generate_feature_radar_chart(results['features'])
            
            # Context for template
            ctx = {
                'filename': results.get('filename'),
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                'ai_probability': results.get('ai_probability', 0),
                'context': results.get('context'),
                'public_report_content': results.get('public_report_content'),
                'flags': results.get('flags', []),
                'results': results,
                'spectrogram_html': spectrogram_html,
                'radar_html': radar_html
            }
            
            rendered = template.render(**ctx)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(rendered)
            print(f"   📄 HTML report saved: {path}")
            
        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()
            self._generate_fallback_html(results, filename)
    
    def _generate_fallback_html(self, results: Dict, filename: str):
        """Generate minimal HTML report as fallback."""
        path = os.path.join(self.output_dir, filename + ".html")
        
        content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MusicTruth Report</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; }}
        .score {{ font-size: 3em; color: {'red' if results.get('ai_probability', 0) > 0.5 else 'green'}; }}
    </style>
</head>
<body>
    <h1>MusicTruth Analysis</h1>
    <p><strong>File:</strong> {results.get('filename', 'Unknown')}</p>
    <div class="score">AI Probability: {results.get('ai_probability', 0)*100:.1f}%</div>
    <h2>Flags</h2>
    <ul>
        {''.join(f'<li>{flag}</li>' for flag in results.get('flags', []))}
    </ul>
</body>
</html>"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   📄 Fallback HTML report saved: {path}")

    def _generate_csv(self, results: Dict, filename: str):
        # Placeholder
        pass
