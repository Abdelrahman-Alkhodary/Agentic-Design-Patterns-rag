##############################################################
###################  STEP ONE ################################
######### Read the PDF and extract text into markdown ########
##############################################################
import os
from src.preprocessing.preprocess_ocr import preprocess_pdf_4_ocr
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

# pdf path to extract its text
pdf_path = './data/raw/Agentic_Design_Patterns.pdf'
# the base directory to extract the pdf as images
output_pdf_path_imgs = './data/raw'

pdf_name, pdf_data = preprocess_pdf_4_ocr(
    pdf_path=pdf_path,
    out_dir=output_pdf_path_imgs
)

# the extracted data from the ocr model
pdf_ocr = './data/ocr/' + pdf_name
os.makedirs(pdf_ocr, exist_ok=True)

# The ocr model from deepseek
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="./models/ocr/")
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True, cache_dir="./models/ocr/")
model = model.eval().cuda().to(torch.bfloat16)

pdf_content_markdown = "./data/processed/" + pdf_name + ".mmd"

with open(pdf_content_markdown, "w", encoding='utf-8') as outfile:
    # loop on the saved images to extract the texts
    for idx, img in tqdm(enumerate(pdf_data), total=len(pdf_data), desc="Extracting the text with OCR"):
        # prompt = "<image>\nFree OCR. "
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        image_file = img
        output_ocr_path = pdf_ocr + f"/page_{idx:04d}"

        res = model.infer(
            tokenizer,
            prompt=prompt, 
            image_file=image_file, 
            output_path=output_ocr_path, 
            base_size=1024, 
            image_size=640, 
            crop_mode=True, 
            save_results=True, 
            test_compress=True
        )
        
        result_file = os.path.join(output_ocr_path, "result.mmd")
        with open(result_file, "r", encoding="utf-8") as infile:
            content = infile.read()
            
            outfile.write(content)
            outfile.write(f"\n\n\n\n")