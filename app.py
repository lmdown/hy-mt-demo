import gradio as gr
import torch
import fitz  # PyMuPDF
import os
import time
import locale
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

# Create bilingual display names for dropdown when system language is Chinese
bilingual_display_map = {
    "English": "English (英语)",
    "Chinese": "Chinese (中文)",
    "Japanese": "Japanese (日语)",
    "Korean": "Korean (韩语)",
    "French": "French (法语)",
    "German": "German (德语)",
    "Spanish": "Spanish (西班牙语)",
    "Portuguese": "Portuguese (葡萄牙语)",
    "Italian": "Italian (意大利语)",
    "Russian": "Russian (俄语)",
    "Arabic": "Arabic (阿拉伯语)",
    "Turkish": "Turkish (土耳其语)",
    "Thai": "Thai (泰语)",
    "Vietnamese": "Vietnamese (越南语)",
    "Malay": "Malay (马来语)",
    "Indonesian": "Indonesian (印尼语)",
    "Filipino": "Filipino (菲律宾语)",
    "Hindi": "Hindi (印地语)",
    "Traditional Chinese": "Traditional Chinese (繁体中文)",
    "Polish": "Polish (波兰语)",
    "Czech": "Czech (捷克语)",
    "Dutch": "Dutch (荷兰语)",
    "Khmer": "Khmer (高棉语)",
    "Burmese": "Burmese (缅甸语)",
    "Persian": "Persian (波斯语)",
    "Gujarati": "Gujarati (古吉拉特语)",
    "Urdu": "Urdu (乌尔都语)",
    "Telugu": "Telugu (泰卢固语)",
    "Marathi": "Marathi (马拉地语)",
    "Hebrew": "Hebrew (希伯来语)",
    "Bengali": "Bengali (孟加拉语)",
    "Tamil": "Tamil (泰米尔语)",
    "Ukrainian": "Ukrainian (乌克兰语)",
    "Tibetan": "Tibetan (藏语)",
    "Kazakh": "Kazakh (哈萨克语)",
    "Mongolian": "Mongolian (蒙古语)",
    "Uyghur": "Uyghur (维吾尔语)",
    "Cantonese": "Cantonese (粤语)"
}

# Determine model size from directory name
model_size = "1.8B"
if "7b" in model_dir_name.lower():
    model_size = "7B"
elif "1.5b" in model_dir_name.lower():
    model_size = "1.8B"

print(f"Detected model size: {model_size}")

# Detect system language
try:
    system_lang = locale.getdefaultlocale()[0]
    if system_lang and 'zh' in system_lang:
        current_lang = 'zh'
    else:
        current_lang = 'en'
except:
    current_lang = 'en'

print(f"Detected system language: {current_lang}")

# Reverse mapping for display names to model language names
display_to_model_map = {v: k for k, v in display_language_map.items()}

# Get language list for dropdown with display names
if current_lang == 'zh':
    # Use bilingual display names for Chinese system language
    language_list = list(bilingual_display_map.values())
    # Update display_to_model_map for bilingual names
    display_to_model_map = {v: k for k, v in bilingual_display_map.items()}
else:
    # Use original display names for English system language
    language_list = list(display_language_map.values())

# Language dictionaries
lang_dict = {
    'en': {
        'app_title': f"HY-MT1.5-{model_size} Translator",
        'app_description1': f"A powerful translation tool supporting text and PDF files, built with Tencent's HY-MT1.5-{model_size} model.",
        'app_description2': "",
        'text_tab': "📝 Text Translation",
        'original_text': "Original Text",
        'target_language': "Target Language",
        'gen_params': "⚙️ Generation Parameters",
        'max_new_tokens': "Max New Tokens",
        'temperature': "Temperature",
        'top_p': "Top P",
        'repetition_penalty': "Repetition Penalty",
        'do_sample': "Do Sample",
        'translate_btn': "🔄 Translate",
        'translation_result': "Translation Result",
        'pdf_tab': "📄 PDF Translation",
        'pdf_description': "### Upload a PDF file to extract text and translate it.",
        'upload_pdf': "📄 Upload PDF File",
        'translate_pdf_btn': "🔄 Translate PDF",
        'extracted_text': "📋 Extracted Text",
        'characters': "characters",
        'placeholder': "Enter text to translate...",
        'templates_help_tab': "📖 Templates Help",
        'templates_help_title': "Translation Templates Usage Guide",
        'terminology_template_title': "Terminology Intervention Template",
        'context_template_title': "Context Translation Template",
        'format_template_title': "Format-Preserving Translation Template",
        'prompt_tips_title': "Prompt Usage Tips",
        'prompt_tip_1': "1. The prompt is sent to the model along with your text",
        'prompt_tip_2': "2. You can modify the prompt to control translation behavior",
        'prompt_tip_3': "3. The prompt updates automatically when you change the target language",
        'prompt_tip_4': "4. For best results, keep prompts clear and concise",
        'general_tips': "General Tips",
        'prompt_note': "**Note:** This prompt dynamically changes based on the target language selection. You can edit it freely and use the templates from the 'Templates Help' tab.",
        'translation_prompt': "Translation Prompt"
    },
    'zh': {
        'app_title': f"HY-MT1.5-{model_size} 翻译器",
        'app_description1': "",
        'app_description2': f"支持文本和PDF文件的强大翻译工具，使用腾讯HY-MT1.5-{model_size}模型构建。",
        'text_tab': "📝 文本翻译",
        'original_text': "原文",
        'target_language': "目标语言",
        'gen_params': "⚙️ 生成参数",
        'max_new_tokens': "最大新 tokens",
        'temperature': "温度",
        'top_p': "核采样概率",
        'repetition_penalty': "重复惩罚",
        'do_sample': "启用采样",
        'translate_btn': "🔄 翻译",
        'translation_result': "翻译结果",
        'pdf_tab': "📄 PDF翻译",
        'pdf_description': "### 上传PDF文件以提取文本并翻译。",
        'upload_pdf': "📄 上传PDF文件",
        'translate_pdf_btn': "🔄 翻译PDF",
        'extracted_text': "📋 提取的原文",
        'characters': "字符",
        'placeholder': "请输入要翻译的文本...",
        'templates_help_tab': "📖 模板帮助",
        'templates_help_title': "翻译模板使用指南",
        'terminology_template_title': "术语干预模板",
        'context_template_title': "上下文翻译模板",
        'format_template_title': "格式保留翻译模板",
        'prompt_tips_title': "提示使用技巧",
        'prompt_tip_1': "1. 提示会与您的文本一起发送给模型",
        'prompt_tip_2': "2. 您可以修改提示来控制翻译行为",
        'prompt_tip_3': "3. 当您更改目标语言时，提示会自动更新",
        'prompt_tip_4': "4. 为获得最佳效果，请保持提示清晰简洁",
        'general_tips': "通用技巧",
        'prompt_note': "**注意：** 以下提示词会根据目标语言选择动态变化。您可以自由编辑它，参考'模板帮助'标签中的模板。",
        'translation_prompt': "翻译提示词"
    }
}

# Get current language dictionary
lang = lang_dict[current_lang]

# Translation templates
terminology_template = "参考下面的翻译：\n{source_term} 翻译成 {target_term}\n\n将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n{source_text}"

context_template = "{context}\n参考上面的信息，把下面的文本翻译成{target_language}，注意不需要翻译上文，也不要额外解释：\n{source_text}"

format_template = "将以下<source></source>之间的文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释，原文中的<sn></sn>标签表示标签内文本包含格式信息，需要在译文中相应的位置尽量保留该标签。输出格式为：<target>str</target>\n\n<source>{src_text_with_format}</source>"

# Function to update prompt when target language changes
def update_prompt(target_lang):
    # Get model language name first
    model_target_lang = display_to_model_map.get(target_lang, target_lang)
    # Then get display name from display_language_map
    display_target_lang = display_language_map.get(model_target_lang, model_target_lang)
    return f"Translate the following segment into {display_target_lang}, without additional explanation.\n"


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


def translate_text(source_text, target_lang, prompt=None, max_new_tokens=1024, temperature=0.7, top_p=0.6, repetition_penalty=1.05, do_sample=True):
    """Translate text using the HY-MT1.5-1.8B model"""
    if not source_text or not source_text.strip():
        return "No input text provided."
    
    # Start execution time tracking
    start_time = time.time()
    
    # Convert display name to model language name
    model_target_lang = display_to_model_map.get(target_lang, target_lang)
    # Then get display name from display_language_map
    display_target_lang = display_language_map.get(model_target_lang, model_target_lang)
    
    # Use provided prompt or create default
    if prompt:
        full_prompt = f"{prompt}{source_text}"
    else:
        full_prompt = f"Translate the following segment into {display_target_lang}, without additional explanation.\n{source_text}"
    
    messages = [{"role": "user", "content": full_prompt}]
    
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
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
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


def translate_long_text(source_text, target_lang, chunk_size=1500, progress=None, prompt=None, max_new_tokens=1024, temperature=0.7, top_p=0.6, repetition_penalty=1.05, do_sample=True):
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
            
            result = translate_text(source_text, target_lang, prompt, max_new_tokens, temperature, top_p, repetition_penalty, do_sample)
            
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
        
        translated = translate_text(chunk, target_lang, prompt, max_new_tokens, temperature, top_p, repetition_penalty, do_sample)
        translated_chunks.append(translated)
    
    # Mark progress as complete
    if progress:
        progress(1.0, desc="Translation Complete")
    
    return "\n\n".join(translated_chunks)


def process_pdf_and_translate(pdf_file, target_lang, max_new_tokens=1024, temperature=0.7, top_p=0.6, repetition_penalty=1.05, do_sample=True):
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
    print(f"Generation Parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, repetition_penalty={repetition_penalty}, do_sample={do_sample}")
    
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
    translated_text = translate_long_text(extracted_text, target_lang, progress=progress, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample)
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


def translate_input_text(source_text, target_lang, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.6, repetition_penalty=1.05, do_sample=True):
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
    print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    print(f"Generation Parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, repetition_penalty={repetition_penalty}, do_sample={do_sample}")
    
    # Perform translation
    result = translate_long_text(source_text, target_lang, prompt=prompt, progress=progress, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample)
    
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
with gr.Blocks(title=lang['app_title']) as demo:
    gr.Markdown(f"# 🚀 {lang['app_title']}")
    if lang['app_description1']:
        gr.Markdown(lang['app_description1'])
    if lang['app_description2']:
        gr.Markdown(lang['app_description2'])
    
    with gr.Tabs():
        # Tab 1: Text Translation
        with gr.TabItem(lang['text_tab']):
            with gr.Row():
                with gr.Column():
                    # Default input text
                    default_input_text = "# Machine Learning Technical Overview\n\nThe neural network architecture consists of multiple layers including convolutional neural networks (CNNs) and recurrent neural networks (RNNs). The model uses backpropagation to optimize weights and biases, with stochastic gradient descent (SGD) as the optimizer.\n\nKey hyperparameters include learning rate, batch size, and epoch count. The model achieves 95% accuracy on the validation set, with a precision of 0.92 and recall of 0.94.\n\nThis implementation uses PyTorch for model development and TensorBoard for visualization. The deployment pipeline includes Docker containerization and Kubernetes orchestration for scaling."
                    
                    input_text = gr.Textbox(
                        label=lang['original_text'],
                        lines=10,
                        placeholder=lang['placeholder'],
                        value=default_input_text
                    )
                    input_char_count = gr.Textbox(
                        label="",
                        value=f"{len(default_input_text)} {lang['characters']}",
                        interactive=False,
                        container=False,
                        scale=1
                    )
                    target_lang_text = gr.Dropdown(
                        choices=language_list,
                        value="Chinese (中文)" if current_lang == 'zh' else "English",
                        label=lang['target_language']
                    )
                    
                    # Visible prompt input
                    gr.Markdown(lang['prompt_note'])
                    prompt_input = gr.Textbox(
                        label=lang['translation_prompt'],
                        lines=5,
                        placeholder="Enter translation prompt here. This will be sent to the model with your text.",
                        value=f"Translate the following segment into {'中文' if current_lang == 'zh' else 'English'}, without additional explanation.\n"
                    )
                
                with gr.Column():
                    output_text = gr.Textbox(
                        label=lang['translation_result'],
                        lines=10,
                        interactive=False
                    )
                    
                    # Generation parameters
                    with gr.Accordion(lang['gen_params'], open=False):
                        max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=4096,
                            step=128,
                            value=1024,
                            label=lang['max_new_tokens']
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            value=0.7,
                            label=lang['temperature']
                        )
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=0.6,
                            label=lang['top_p']
                        )
                        repetition_penalty = gr.Slider(
                            minimum=0.8,
                            maximum=1.5,
                            step=0.05,
                            value=1.05,
                            label=lang['repetition_penalty']
                        )
                        do_sample = gr.Checkbox(
                            value=True,
                            label=lang['do_sample']
                        )
                    
                    translate_btn = gr.Button(lang['translate_btn'], variant="primary")
                    output_char_count = gr.Textbox(
                        label="",
                        value=f"0 {lang['characters']}",
                        interactive=False,
                        container=False,
                        scale=1
                    )
            
            # Update character count when input text changes
            def update_char_count(text):
                return f"{len(text)} {lang['characters']}"
            
            input_text.change(
                fn=update_char_count,
                inputs=input_text,
                outputs=input_char_count
            )
            
            # Update prompt when target language changes
            target_lang_text.change(
                fn=update_prompt,
                inputs=target_lang_text,
                outputs=prompt_input
            )
            
            # Update character counts after translation
            def translate_and_count(source_text, target_lang, prompt, max_new_tokens, temperature, top_p, repetition_penalty, do_sample):
                result = translate_input_text(source_text, target_lang, prompt, max_new_tokens, temperature, top_p, repetition_penalty, do_sample)
                return [result, update_char_count(result)]
            
            translate_btn.click(
                fn=translate_and_count,
                inputs=[input_text, target_lang_text, prompt_input, max_new_tokens, temperature, top_p, repetition_penalty, do_sample],
                outputs=[output_text, output_char_count]
            )
        
        # Tab 2: PDF Translation
        with gr.TabItem(lang['pdf_tab']):
            gr.Markdown(lang['pdf_description'])
            
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label=lang['upload_pdf'],
                        file_types=[".pdf"]
                    )
                    target_lang_pdf = gr.Dropdown(
                        choices=language_list,
                        value="Chinese (中文)" if current_lang == 'zh' else "English",
                        label=lang['target_language']
                    )
                    
                    # Generation parameters
                    with gr.Accordion(lang['gen_params'], open=False):
                        max_new_tokens_pdf = gr.Slider(
                            minimum=128,
                            maximum=4096,
                            step=128,
                            value=1024,
                            label=lang['max_new_tokens']
                        )
                        temperature_pdf = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            value=0.7,
                            label=lang['temperature']
                        )
                        top_p_pdf = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=0.6,
                            label=lang['top_p']
                        )
                        repetition_penalty_pdf = gr.Slider(
                            minimum=0.8,
                            maximum=1.5,
                            step=0.05,
                            value=1.05,
                            label=lang['repetition_penalty']
                        )
                        do_sample_pdf = gr.Checkbox(
                            value=True,
                            label=lang['do_sample']
                        )
                    
                    translate_pdf_btn = gr.Button(lang['translate_pdf_btn'], variant="primary")
            
            with gr.Row():
                with gr.Column():
                    extracted_text = gr.Textbox(
                        label=lang['extracted_text'],
                        lines=15,
                        interactive=False
                    )
                    extracted_char_count = gr.Textbox(
                        label="",
                        value=f"0 {lang['characters']}",
                        interactive=False,
                        container=False,
                        scale=1
                    )
                
                with gr.Column():
                    translated_pdf_text = gr.Textbox(
                        label=lang['translation_result'],
                        lines=15,
                        interactive=False
                    )
                    translated_pdf_char_count = gr.Textbox(
                        label="",
                        value=f"0 {lang['characters']}",
                        interactive=False,
                        container=False,
                        scale=1
                    )
            
            # Update PDF text character counts
            def update_pdf_char_counts(extracted, translated):
                return [
                    f"{len(extracted)} {lang['characters']}",
                    f"{len(translated)} {lang['characters']}"
                ]
            
            # Process PDF and update character counts
            def process_pdf_and_count(pdf_file, target_lang, max_new_tokens, temperature, top_p, repetition_penalty, do_sample):
                extracted, translated = process_pdf_and_translate(pdf_file, target_lang, max_new_tokens, temperature, top_p, repetition_penalty, do_sample)
                counts = update_pdf_char_counts(extracted, translated)
                return [extracted, translated, counts[0], counts[1]]
            
            translate_pdf_btn.click(
                fn=process_pdf_and_count,
                inputs=[pdf_input, target_lang_pdf, max_new_tokens_pdf, temperature_pdf, top_p_pdf, repetition_penalty_pdf, do_sample_pdf],
                outputs=[extracted_text, translated_pdf_text, extracted_char_count, translated_pdf_char_count]
            )
        
        # Tab 3: Templates Help
        with gr.TabItem(lang['templates_help_tab']):
            gr.Markdown(f"# {lang['templates_help_title']}")
            
            # Terminology Intervention Template
            with gr.Accordion(lang['terminology_template_title'], open=False):
                gr.Markdown("### Usage")
                gr.Markdown("Use this template when you need to control specific terminology translations.")
                gr.Markdown("### Example")
                gr.Markdown("参考下面的翻译：")
                gr.Markdown("machine learning 翻译成 机器学习")
                gr.Markdown("neural network 翻译成 神经网络")
                gr.Markdown("")
                gr.Markdown("将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：")
                gr.Markdown("I'm studying machine learning and neural networks.")
                
            # Context Translation Template
            with gr.Accordion(lang['context_template_title'], open=False):
                gr.Markdown("### Usage")
                gr.Markdown("Use this template when you need to provide additional context for better translation.")
                gr.Markdown("### Example")
                gr.Markdown("This text is about artificial intelligence research.")
                gr.Markdown("参考上面的信息，把下面的文本翻译成中文，注意不需要翻译上文，也不要额外解释：")
                gr.Markdown("The model achieved 95% accuracy on the test set.")
                
            # Format-Preserving Translation Template
            with gr.Accordion(lang['format_template_title'], open=False):
                gr.Markdown("### Usage")
                gr.Markdown("Use this template when you need to preserve formatting tags in the translation.")
                gr.Markdown("### Example")
                gr.Markdown("将以下<source></source>之间的文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释，原文中的<sn></sn>标签表示标签内文本包含格式信息，需要在译文中相应的位置尽量保留该标签。输出格式为：<target>str</target>")
                gr.Markdown("")
                gr.Markdown("<source>")
                gr.Markdown("The <sn>machine learning</sn> model is trained on <sn>large datasets</sn>.")
                gr.Markdown("</source>")
                
            # Prompt Usage Tips
            with gr.Accordion(lang['prompt_tips_title'], open=False):
                gr.Markdown(f"### {lang['general_tips']}")
                gr.Markdown(lang['prompt_tip_1'])
                gr.Markdown(lang['prompt_tip_2'])
                gr.Markdown(lang['prompt_tip_3'])
                gr.Markdown(lang['prompt_tip_4'])

# Launch the application
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch()