import gradio as gr
import torch
import fitz  # PyMuPDF
import os
import time
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up project relative path for model storage
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get model directory from environment variable or use default
model_dir_name = os.environ.get("MODEL_DIR", "models-1.8b")
model_dir = os.path.join(current_dir, model_dir_name)
print(f"Using model directory: {model_dir} (set via {'environment variable' if 'MODEL_DIR' in os.environ else 'default'})")

# Create models directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Model directory created: {model_dir}")
else:
    print(f"Model directory exists: {model_dir}")

# Local model loading mode - no server download
# Check if model files exist locally
if not os.path.exists(os.path.join(model_dir, "config.json")):
    print(f"❌ Error: Model files not found in {model_dir}")
    print("Please ensure the model files are placed in the correct directory.")
    exit(1)

print(f"✅ Model files found locally in: {model_dir}")

# Auto-select device and precision based on environment
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Loading model on {device} with {dtype}...")

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map=device,
    dtype=dtype
)

# Supported languages (both English and Chinese names)
supported_languages = {
    "English": "English",
    "Chinese": "中文",
    "Japanese": "日本語",
    "Korean": "한국어",
    "French": "French",
    "German": "German",
    "Spanish": "Spanish",
    "Portuguese": "Portuguese",
    "Italian": "Italian",
    "Russian": "Russian",
    "Arabic": "Arabic",
    "Turkish": "Turkish",
    "Thai": "Thai",
    "Vietnamese": "Vietnamese",
    "Malay": "Malay",
    "Indonesian": "Indonesian",
    "Filipino": "Filipino",
    "Hindi": "Hindi",
    "Traditional Chinese": "繁体中文",
    "Polish": "Polish",
    "Czech": "Czech",
    "Dutch": "Dutch",
    "Khmer": "Khmer",
    "Burmese": "Burmese",
    "Persian": "Persian",
    "Gujarati": "Gujarati",
    "Urdu": "Urdu",
    "Telugu": "Telugu",
    "Marathi": "Marathi",
    "Hebrew": "Hebrew",
    "Bengali": "Bengali",
    "Tamil": "Tamil",
    "Ukrainian": "Ukrainian",
    "Tibetan": "Tibetan",
    "Kazakh": "Kazakh",
    "Mongolian": "Mongolian",
    "Uyghur": "Uyghur",
    "Cantonese": "粤语"
}

# Create display names for dropdown with native script for specific languages
display_language_map = {
    "English": "English",
    "Chinese": "中文",
    "Japanese": "日本語",
    "Korean": "한국어",
    "French": "French",
    "German": "German",
    "Spanish": "Spanish",
    "Portuguese": "Portuguese",
    "Italian": "Italian",
    "Russian": "Russian",
    "Arabic": "Arabic",
    "Turkish": "Turkish",
    "Thai": "Thai",
    "Vietnamese": "Vietnamese",
    "Malay": "Malay",
    "Indonesian": "Indonesian",
    "Filipino": "Filipino",
    "Hindi": "Hindi",
    "Traditional Chinese": "繁体中文",
    "Polish": "Polish",
    "Czech": "Czech",
    "Dutch": "Dutch",
    "Khmer": "Khmer",
    "Burmese": "Burmese",
    "Persian": "Persian",
    "Gujarati": "Gujarati",
    "Urdu": "Urdu",
    "Telugu": "Telugu",
    "Marathi": "Marathi",
    "Hebrew": "Hebrew",
    "Bengali": "Bengali",
    "Tamil": "Tamil",
    "Ukrainian": "Ukrainian",
    "Tibetan": "Tibetan",
    "Kazakh": "Kazakh",
    "Mongolian": "Mongolian",
    "Uyghur": "Uyghur",
    "Cantonese": "粤语"
}

# Reverse mapping for display names to model language names
display_to_model_map = {v: k for k, v in display_language_map.items()}

# Get language list for dropdown with display names
language_list = list(display_language_map.values())


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    if pdf_file is None:
        return ""
    
    try:
        doc = fitz.open(pdf_file.name)
        full_text = ""
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                full_text += f"\n--- Page {page_num} ---\n{text.strip()}\n"
        
        doc.close()
        return full_text.strip()
    
    except Exception as e:
        return f"❌ Error extracting text from PDF: {str(e)}"


def translate_text(source_text, target_lang):
    """Translate text using the HY-MT1.5-1.8B model"""
    if not source_text or not source_text.strip():
        return "No input text provided."
    
    # Start execution time tracking
    start_time = time.time()
    
    # Convert display name to model language name
    model_target_lang = display_to_model_map.get(target_lang, target_lang)
    
    # Create appropriate prompt based on target language
    prompt = f"Translate the following segment into {model_target_lang}, without additional explanation.\n{source_text}"
    
    messages = [{"role": "user", "content": prompt}]
    
    # Process input
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(device)
    
    # Get input token count
    input_tokens = text_input.shape[1]
    
    # Log generation parameters
    generation_params = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.6,
        "repetition_penalty": 1.05
    }
    
    # Generate translation
    with torch.no_grad():
        generated_ids = model.generate(
            text_input,
            **generation_params
        )
    
    # Get generated token count
    generated_tokens = generated_ids[0].shape[0] - input_tokens
    
    # Process output
    input_length = text_input.shape[1]
    response = generated_ids[0][input_length:]
    decoded_output = tokenizer.decode(response, skip_special_tokens=True)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Log technical details (debug level)
    print(f"  ⏱️  Chunk translation time: {execution_time:.2f}s")
    print(f"  🔢 Input tokens: {input_tokens}, Generated tokens: {generated_tokens}")
    print(f"  📊 Tokens/sec: {generated_tokens/execution_time:.1f}" if execution_time > 0 else "  📊 Tokens/sec: N/A")
    print(f"  💻 Device: {device}")
    
    return decoded_output


def translate_long_text(source_text, target_lang, chunk_size=1500, progress=None):
    """Translate long text by splitting into chunks"""
    if not source_text or not source_text.strip():
        return "No input text provided."
    
    # Translate short text directly with tqdm progress
    if len(source_text) <= chunk_size:
        # Show tqdm progress bar for single chunk
        print(f"Debug: Text is short (length {len(source_text)}), using single chunk")
        with tqdm(total=1, desc="Translating text chunks", unit="chunk") as pbar:
            if progress:
                progress(0.0, desc="Starting translation")
            
            result = translate_text(source_text, target_lang)
            
            pbar.update(1)
            
            if progress:
                progress(1.0, desc="Translation Complete")
            
        return result
    
    # Split long text by paragraphs
    paragraphs = source_text.split('\n\n')
    print(f"Debug: Detected {len(paragraphs)} paragraphs")
    for idx, para in enumerate(paragraphs):
        print(f"Debug: Paragraph {idx+1} length: {len(para)} characters")
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + '\n\n'
        else:
            if current_chunk:
                print(f"Debug: Creating chunk of length {len(current_chunk)}")
                chunks.append(current_chunk.strip())
            current_chunk = para + '\n\n'
    
    if current_chunk:
        print(f"Debug: Creating final chunk of length {len(current_chunk)}")
        chunks.append(current_chunk.strip())
    
    # Translate each chunk with tqdm progress bar in terminal
    translated_chunks = []
    total_chunks = len(chunks)
    print(f"Debug: Total chunks created: {total_chunks}")
    
    for i, chunk in tqdm(enumerate(chunks), total=total_chunks, desc="Translating text chunks", unit="chunk"):
        # Update Gradio progress bar with only description (no estimated time)
        if progress:
            progress(0.0, desc=f"Translating chunk {i+1}/{total_chunks}")
        
        translated = translate_text(chunk, target_lang)
        translated_chunks.append(translated)
    
    # Mark progress as complete
    if progress:
        progress(1.0, desc="Translation Complete")
    
    return "\n\n".join(translated_chunks)


def process_pdf_and_translate(pdf_file, target_lang):
    """Process PDF file: extract text and translate"""
    # Create progress object
    progress = gr.Progress()
    
    if pdf_file is None:
        return "", "Please upload a PDF file."
    
    # Start execution time tracking
    start_time = time.time()
    
    # Log PDF translation request details
    print(f"\n=== PDF Translation Request ===")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"PDF File: {pdf_file.name}")
    print(f"PDF File Size: {os.path.getsize(pdf_file.name) / (1024*1024):.2f} MB")
    print(f"Target Language (Display): {target_lang}")
    model_target_lang = display_to_model_map.get(target_lang, target_lang)
    print(f"Target Language (Model): {model_target_lang}")
    
    progress(0.1, desc="Extracting text from PDF")
    
    # Extract text from PDF
    extract_start = time.time()
    extracted_text = extract_text_from_pdf(pdf_file)
    extract_time = time.time() - extract_start
    
    if extracted_text.startswith("❌"):
        return "", extracted_text
    
    if not extracted_text.strip():
        return "", "No text could be extracted from the PDF."
    
    print(f"PDF Text Extraction Time: {extract_time:.2f} seconds")
    print(f"Extracted Text Length: {len(extracted_text)} characters")
    print(f"Extracted Text (truncated): {extracted_text[:500]}{'...' if len(extracted_text) > 500 else ''}")
    
    # Translate the extracted text
    progress(0.2, desc="Starting translation")
    translate_start = time.time()
    translated_text = translate_long_text(extracted_text, target_lang, progress=progress)
    translate_time = time.time() - translate_start
    
    # Calculate total execution time
    total_time = time.time() - start_time
    
    # Log translation results
    print(f"=== PDF Translation Complete ===")
    print(f"Translation Time: {translate_time:.2f} seconds")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Translated Text Length: {len(translated_text)} characters")
    print(f"Translated Text (truncated): {translated_text[:500]}{'...' if len(translated_text) > 500 else ''}")
    print("=" * 50)
    
    return extracted_text, translated_text


def translate_input_text(source_text, target_lang):
    """Translate input text"""
    # Create progress object
    progress = gr.Progress()
    
    # Start execution time tracking
    start_time = time.time()
    
    # Log translation request details
    print(f"\n=== Translation Request ===")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Target Language (Display): {target_lang}")
    model_target_lang = display_to_model_map.get(target_lang, target_lang)
    print(f"Target Language (Model): {model_target_lang}")
    print(f"Input Text Length: {len(source_text)} characters")
    print(f"Input Text (truncated): {source_text[:500]}{'...' if len(source_text) > 500 else ''}")
    
    # Perform translation
    result = translate_long_text(source_text, target_lang, progress=progress)
    
    # Calculate and log execution time
    execution_time = time.time() - start_time
    print(f"=== Translation Complete ===")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Output Text Length: {len(result)} characters")
    print(f"Output Text (truncated): {result[:500]}{'...' if len(result) > 500 else ''}")
    print("=" * 50)
    
    return result


# Determine model size from directory name
model_size = "1.8B"
if "7b" in model_dir_name.lower():
    model_size = "7B"
elif "1.5b" in model_dir_name.lower():
    model_size = "1.8B"

print(f"Detected model size: {model_size}")

# Create Gradio interface
with gr.Blocks(title=f"HY-MT1.5-{model_size} Translator") as demo:
    gr.Markdown(f"# 🚀 HY-MT1.5-{model_size} Translator")
    gr.Markdown(f"A powerful translation tool supporting text and PDF files, built with Tencent's HY-MT1.5-{model_size} model.")
    gr.Markdown(f"支持文本和PDF文件的强大翻译工具，使用腾讯HY-MT1.5-{model_size}模型构建。")
    
    with gr.Tabs():
        # Tab 1: Text Translation
        with gr.TabItem("📝 Text Translation | 文本翻译"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Original Text | 原文",
                        lines=10,
                        placeholder="Enter text to translate... | 请输入要翻译的文本..."
                    )
                    input_char_count = gr.Textbox(
                        label="",
                        value="0 characters",
                        interactive=False,
                        container=False,
                        scale=1
                    )
                    target_lang_text = gr.Dropdown(
                        choices=language_list,
                        value="中文",
                        label="Target Language | 目标语言"
                    )
                    translate_btn = gr.Button("🔄 Translate | 翻译", variant="primary")
                
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Translation Result | 翻译结果",
                        lines=10,
                        interactive=False
                    )
                    output_char_count = gr.Textbox(
                        label="",
                        value="0 characters",
                        interactive=False,
                        container=False,
                        scale=1
                    )
            
            # Update character count when input text changes
            def update_char_count(text):
                return f"{len(text)} characters"
            
            input_text.change(
                fn=update_char_count,
                inputs=input_text,
                outputs=input_char_count
            )
            
            # Update character counts after translation
            def translate_and_count(source_text, target_lang):
                result = translate_input_text(source_text, target_lang)
                return [result, update_char_count(result)]
            
            translate_btn.click(
                fn=translate_and_count,
                inputs=[input_text, target_lang_text],
                outputs=[output_text, output_char_count]
            )
        
        # Tab 2: PDF Translation
        with gr.TabItem("📄 PDF Translation | PDF翻译"):
            gr.Markdown("### Upload a PDF file to extract text and translate it. | 上传PDF文件以提取文本并翻译。")
            
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label="📄 Upload PDF File | 上传PDF文件",
                        file_types=[".pdf"]
                    )
                    target_lang_pdf = gr.Dropdown(
                        choices=language_list,
                        value="中文",
                        label="Target Language | 目标语言"
                    )
                    translate_pdf_btn = gr.Button("🔄 Translate PDF | 翻译PDF", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    extracted_text = gr.Textbox(
                        label="📋 Extracted Text | 提取的原文",
                        lines=15,
                        interactive=False
                    )
                    extracted_char_count = gr.Textbox(
                        label="",
                        value="0 characters",
                        interactive=False,
                        container=False,
                        scale=1
                    )
                
                with gr.Column():
                    translated_pdf_text = gr.Textbox(
                        label="📋 Translation Result | 翻译结果",
                        lines=15,
                        interactive=False
                    )
                    translated_pdf_char_count = gr.Textbox(
                        label="",
                        value="0 characters",
                        interactive=False,
                        container=False,
                        scale=1
                    )
            
            # Update PDF text character counts
            def update_pdf_char_counts(extracted, translated):
                return [
                    f"{len(extracted)} characters",
                    f"{len(translated)} characters"
                ]
            
            # Process PDF and update character counts
            def process_pdf_and_count(pdf_file, target_lang):
                extracted, translated = process_pdf_and_translate(pdf_file, target_lang)
                counts = update_pdf_char_counts(extracted, translated)
                return [extracted, translated, counts[0], counts[1]]
            
            translate_pdf_btn.click(
                fn=process_pdf_and_count,
                inputs=[pdf_input, target_lang_pdf],
                outputs=[extracted_text, translated_pdf_text, extracted_char_count, translated_pdf_char_count]
            )

# Launch the application
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch()