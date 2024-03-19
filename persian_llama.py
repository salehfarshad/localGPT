from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from huggingface_hub import hf_hub_download
# from constants import CONTEXT_WINDOW_SIZE, MAX_NEW_TOKENS, MODELS_PATH, N_BATCH, N_GPU_LAYERS, MODEL_ID, MODEL_BASENAME

n_gpu_layers = 20  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 1024

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# model_path = hf_hub_download(
#     repo_id=MODEL_ID,
#     filename=MODEL_BASENAME,
#     resume_download=True,
#     cache_dir=MODELS_PATH,
# )

# kwargs = {
#     "model_path": model_path,
#     "n_ctx": CONTEXT_WINDOW_SIZE,
#     "max_tokens": MAX_NEW_TOKENS,
#     "n_batch": N_BATCH,  # set this based on your GPU & CPU RAM
#     "n_gpu_layer":N_GPU_LAYERS,
# }
# llm_path = r'./models\\models--mostafaamiri--persian-llama-7b-GGUF-Q4\\snapshots\\982b03b059d05d08e94c1fff252e32b8e49f23a4\\persian_llama_7b.Q4_K_M.gguf'
llm_path = r'./models\\models--asedmammad--PersianMind-v1.0-GGUF\\snapshots\\95ca2c0e97446513bed7c804d13cd58107adb92a\\PersianMind-v1.0.q2_K.gguf'

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=llm_path,
    n_gpu_layers=n_gpu_layers, n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=n_ctx,
    temperature=0.2,
    repeat_penalty=1.2,
)

llm("""
"### Context": "در یک جنگل سرسبز، یک شیر نر قدرتمند زندگی می کرد که سلطان جنگل نام داشت. او از تمام حیوانات دیگر قوی تر بود و همه از او می ترسیدند. یک روز، یک موش صحرایی کوچک و شجاع به قلمرو سلطان شیر وارد شد. سلطان شیر عصبانی شد و موش صحرایی را تهدید کرد."
"### Question": "1. موش صحرایی چرا به قلمرو سلطان شیر رفت؟\n2. سلطان شیر چه واکنشی نسبت به موش صحرایی نشان داد؟"
""")
