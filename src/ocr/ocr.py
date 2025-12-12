import os
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm



def load_ocr_model(model_name: str, device: str = "0"):

    os.environ["CUDA_VISIBLE_DEVICES"] = device

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./models/ocr/"
    )

    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        cache_dir="./models/ocr/"
    )

    model = model.eval().cuda().to(torch.bfloat16)
    return tokenizer, model

def run_ocr_on_images(
    pdf_data,
    pdf_name: str,
    tokenizer,
    model,
    output_dir_ocr: str = "./data/ocr",
    output_dir_processed: str = "./data/processed"
):

    pdf_ocr = os.path.join(output_dir_ocr, pdf_name)
    os.makedirs(pdf_ocr, exist_ok=True)

    pdf_content_markdown = os.path.join(
        output_dir_processed, f"{pdf_name}.mmd"
    )

    with open(pdf_content_markdown, "w", encoding="utf-8") as outfile:
        for idx, img in tqdm(
            enumerate(pdf_data),
            total=len(pdf_data),
            desc="Extracting the text with OCR"
        ):
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            output_ocr_path = os.path.join(pdf_ocr, f"page_{idx:04d}")

            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=img,
                output_path=output_ocr_path,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=True
            )

            result_file = os.path.join(output_ocr_path, "result.mmd")
            with open(result_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n\n\n")

    return pdf_content_markdown
