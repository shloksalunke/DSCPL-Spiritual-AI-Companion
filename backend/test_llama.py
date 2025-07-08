from llama_cpp import Llama

llm = Llama(model_path="./models/llama-2-7b.Q4_K_M.gguf", n_ctx=2048)

prompt = "Q: I feel anxious and need prayer. Can you help me?\nA:"
output = llm(prompt, max_tokens=256, stop=["Q:", "\n"])

print("\n=== LLaMA Response ===")
print(output["choices"][0]["text"].strip())

