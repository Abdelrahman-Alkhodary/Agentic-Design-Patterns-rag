##############################################################
###################  STEP ONE ################################
######### Read the PDF and extract text into markdown ########
##############################################################
from src.preprocessing.preprocess_ocr import preprocess_pdf_4_ocr
from src.ocr.ocr import load_ocr_model, run_ocr_on_images


def extract_pdf_to_markdown(
    pdf_path: str,
    raw_output_dir: str = "./data/raw",
    model_name: str = "deepseek-ai/DeepSeek-OCR"
):
    pdf_name, pdf_data = preprocess_pdf_4_ocr(
        pdf_path=pdf_path,
        out_dir=raw_output_dir
    )

    tokenizer, model = load_ocr_model(model_name)

    output_markdown = run_ocr_on_images(
        pdf_data=pdf_data,
        pdf_name=pdf_name,
        tokenizer=tokenizer,
        model=model
    )

    return output_markdown


def main():
    
    # pdf path to extract its text
    pdf_path = './data/raw/Agentic_Design_Patterns.pdf'
    
    # The pdf text as markdown
    output_file = extract_pdf_to_markdown(pdf_path)
    print("Markdown saved to:", output_file)
    

if __name__ == "__main__":
    main()
    