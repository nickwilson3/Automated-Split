import os
from weasyprint import HTML
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

def convert_html_to_pdf(html_path, pdf_path):
    """
    Convert HTML report to PDF
    
    Parameters:
    html_path (str): Path to the HTML file
    pdf_path (str): Path where the PDF will be saved
    
    Returns:
    str: Path to the generated PDF
    """
    # Get directory of HTML file to resolve image paths
    html_dir = os.path.dirname(html_path)
    base_url = f"file://{html_dir}/"
    
    # Convert HTML to PDF
    HTML(html_path, base_url=base_url).write_pdf(pdf_path)
    
    return pdf_path