import os
from typing import Dict, List
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

def generate_reports(results_list: List[Dict], output_dir: str):
    """
    Generates Markdown and PDF reports for the analyzed files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate Markdown Report
    md_path = os.path.join(output_dir, "report.md")
    generate_markdown_report(results_list, md_path)
    
    # 2. Generate PDF Report
    pdf_path = os.path.join(output_dir, "report.pdf")
    generate_pdf_report(results_list, pdf_path)
    
    print(f"Reports generated at {output_dir}")

def generate_markdown_report(results_list: List[Dict], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("# MusicTruth Analysis Report\n\n")
        
        for res in results_list:
            f.write(f"## {os.path.basename(res['filename'])}\n")
            f.write(f"**AI Probability Score:** {res['ai_probability']:.2f}\n\n")
            
            if res['flags']:
                f.write("### 🚩 Suspicious Indicators\n")
                for flag in res['flags']:
                    f.write(f"- {flag}\n")
                f.write("\n")
                
            f.write("### 📊 Metrics\n")
            for k, v in res['metrics'].items():
                f.write(f"- **{k}:** {v}\n")
            f.write("\n")
            
            if 'vocal_analysis' in res:
                f.write("### 🎤 Vocal Forensics\n")
                v = res['vocal_analysis']
                f.write(f"- **Pitch Deviation:** {v.get('pitch_deviation', 'N/A')}\n")
                if v.get('flags'):
                    for flag in v['flags']:
                        f.write(f"- ⚠️ {flag}\n")
                f.write("\n")

            if 'album_similarity' in res:
                f.write(f"### 💿 Album Consistency\n")
                f.write(f"- **Similarity Score:** {res['album_similarity']:.3f} (Higher is better)\n\n")
                
            f.write("---\n")

def generate_pdf_report(results_list: List[Dict], file_path: str):
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("MusicTruth Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    for res in results_list:
        filename = os.path.basename(res['filename'])
        story.append(Paragraph(f"File: {filename}", styles['Heading2']))
        
        # Summary Data
        data = [
            ["Metric", "Value"],
            ["AI Probability", f"{res['ai_probability']:.2f}"],
            ["Duration", f"{res['duration']:.2f}s"],
            ["Sample Rate", f"{res['sample_rate']} Hz"]
        ]
        
        # Add metrics
        for k, v in res['metrics'].items():
            if isinstance(v, float):
                data.append([k, f"{v:.4f}"])
            else:
                data.append([k, str(v)])
                
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
        
        if res['flags']:
            story.append(Paragraph("Flags:", styles['Heading3']))
            for flag in res['flags']:
                story.append(Paragraph(f"• {flag}", styles['BodyText']))
            story.append(Spacer(1, 12))
            
        story.append(Spacer(1, 24))
        
    doc.build(story)
