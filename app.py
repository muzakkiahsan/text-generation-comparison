import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# === Model Loader ===

@st.cache_resource
def load_mistral_7b():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16) # Use bfloat16 for efficiency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@st.cache_resource
def load_fathom_14b():
    model_id = "FractalAIResearch/Fathom-R1-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16) # Use bfloat16 for efficiency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# === Text Generation Function ===
def generate_text(prompt: str, model_option: str, max_length: int = 100):
    start_time = time.time()
    
    if model_option == "facebook/opt-125m":
        tokenizer, model, device = load_opt_125m()
    else: # openai-community/gpt2
        tokenizer, model, device = load_gpt2()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Generate text. You can adjust parameters like max_length, num_return_sequences, temperature, top_k, top_p
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True, # Enable sampling for more varied outputs
            temperature=0.7, # Controls randomness
            top_k=50, # Limits the sampling pool
            top_p=0.95 # Nucleus sampling
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()
    generation_time = end_time - start_time
    
    return generated_text, generation_time

# === UI Streamlit ===
st.title("Text Generation Comparison App")
st.markdown("Compare text generation between **OPT-125M** and **GPT-2** models side by side")

st.divider()

# Parameters section
col1, col2 = st.columns(2)
with col1:
    max_length = st.slider("Max Length", min_value=50, max_value=200, value=100)
with col2:
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)

prompt_text = st.text_area("Masukkan Teks Awal (Prompt)", height=150, placeholder="Ketik prompt Anda di sini...")

if st.button("🚀 Compare Both Models", type="primary"):
    if prompt_text:
        st.divider()
        st.header("📊 Comparison Results")
        
        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        # Generate with OPT-125M
        with col1:
            st.subheader("🤖 OPT-125M")
            with st.spinner("Generating with OPT-125M..."):
                opt_result, opt_time = generate_text(prompt_text, "facebook/opt-125m", max_length)
            
            st.success(f"✅ Generated in {opt_time:.2f} seconds")
            st.text_area("Result:", value=opt_result, height=200, key="opt_result", disabled=True)
        
        # Generate with GPT-2
        with col2:
            st.subheader("🤖 GPT-2")
            with st.spinner("Generating with GPT-2..."):
                gpt2_result, gpt2_time = generate_text(prompt_text, "openai-community/gpt2", max_length)
            
            st.success(f"✅ Generated in {gpt2_time:.2f} seconds")
            st.text_area("Result:", value=gpt2_result, height=200, key="gpt2_result", disabled=True)
        
        # Performance comparison
        st.divider()
        st.subheader("⚡ Performance Comparison")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("OPT-125M Time", f"{opt_time:.2f}s")
        
        with perf_col2:
            st.metric("GPT-2 Time", f"{gpt2_time:.2f}s")
        
        with perf_col3:
            if opt_time < gpt2_time:
                faster_model = "OPT-125M"
                time_diff = gpt2_time - opt_time
            else:
                faster_model = "GPT-2"
                time_diff = opt_time - gpt2_time
            
            st.metric("Faster Model", faster_model, f"{time_diff:.2f}s faster")
        
        # Additional metrics
        st.subheader("📈 Text Statistics")
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.write("**OPT-125M Statistics:**")
            opt_word_count = len(opt_result.split())
            opt_char_count = len(opt_result)
            st.write(f"- Words: {opt_word_count}")
            st.write(f"- Characters: {opt_char_count}")
            st.write(f"- Words/second: {opt_word_count/opt_time:.1f}")
        
        with stat_col2:
            st.write("**GPT-2 Statistics:**")
            gpt2_word_count = len(gpt2_result.split())
            gpt2_char_count = len(gpt2_result)
            st.write(f"- Words: {gpt2_word_count}")
            st.write(f"- Characters: {gpt2_char_count}")
            st.write(f"- Words/second: {gpt2_word_count/gpt2_time:.1f}")
            
    else:
        st.warning("⚠️ Mohon masukkan prompt untuk menggenerasi teks.")

# Add individual model testing option
st.divider()
st.subheader("🔧 Test Individual Models")

col1, col2 = st.columns(2)

with col1:
    if st.button("Test OPT-125M Only"):
        if prompt_text:
            with st.spinner("Generating with OPT-125M..."):
                opt_result, opt_time = generate_text(prompt_text, "facebook/opt-125m", max_length)
            st.success(f"Generated in {opt_time:.2f} seconds")
            st.write(opt_result)

with col2:
    if st.button("Test GPT-2 Only"):
        if prompt_text:
            with st.spinner("Generating with GPT-2..."):
                gpt2_result, gpt2_time = generate_text(prompt_text, "openai-community/gpt2", max_length)
            st.success(f"Generated in {gpt2_time:.2f} seconds")
            st.write(gpt2_result)
