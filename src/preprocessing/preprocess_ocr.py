import os
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm


DPI = 300
  
def preprocess_pdf_4_ocr(pdf_path, out_dir):
    # Extract the pdf name
    file_name = os.path.basename(pdf_path)
    pdf_name = os.path.splitext(file_name)[0]
    
    out_dir = Path(out_dir + '/' + pdf_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # convert the pdf into pages
    pages = convert_from_path(pdf_path, dpi=DPI)
    
    # save the pages so that ocr model can read it
    page_paths = []
    for i, img in tqdm(enumerate(pages, start=1), total=len(pages), desc="saving the Extracted images from the pdf"):
        p = out_dir / f"page_{i:04d}.png"
        img.save(p)
        page_paths.append(p)
        
    return pdf_name, page_paths
