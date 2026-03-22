"""
model_loader.py - Loads the LoRA-fine-tuned model (Qwen2.5 or Mistral).
Only used when USE_MOCK_LLM=false.

Base model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
Adapter: trained with PEFT/LoRA via unsloth (r=16, alpha=16)

NOTE: Using transformers + PEFT instead of Unsloth for inference
to avoid gradient_checkpointing bug with Unsloth fast inference.
"""

import os
from typing import Any


def load_lora_model(model_path: str) -> Any:
    """
    Load the LoRA adapter using transformers + PEFT (compatible with Unsloth-trained models).

    Args:
        model_path: local path to adapter folder (adapter_config.json + adapter_model.safetensors)

    Returns:
        A callable LLM object with .invoke(prompt) -> str
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    base_model_id = os.environ.get("BASE_MODEL_ID", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

    print(f"[model_loader] Carregando base model: {base_model_id}")
    
    # Configuração 4-bit via BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"[model_loader] Carregando tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"[model_loader] Aplicando LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)

    print("[model_loader] Modelo pronto para inferência.")

    class _LoRALLM:
        def invoke(self, prompt: str) -> str:
            # Qwen2.5-Instruct usa ChatML — aplicar chat template
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente médico especializado em cardiologia. Se o caso não for cardiológico, informe que está fora do escopo.\n\n"
                        "Use somente as informações explicitamente fornecidas no contexto do paciente. "
                        "Nunca invente histórico, comorbidades, exames prévios ou fatores de risco não informados.\n\n"
                        "Se o histórico não estiver presente no contexto, escreva explicitamente: "
                        "'Histórico relevante não informado'.\n"
                        "Diferencie fatos informados de hipóteses clínicas. "
                        "Não transforme inferências em histórico confirmado.\n\n"
                        "Responda com:\n"
                        "• Resumo clínico\n"
                        "• Raciocínio clínico\n"
                        "• Hipótese diagnóstica principal\n"
                        "• Diagnósticos diferenciais\n"
                        "• Exames recomendados"
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            # apply_chat_template adiciona <|im_start|>/<|im_end|> automaticamente
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # fallback para modelos sem chat template
                text = f"### Instrução\n{prompt}\n\n### Resposta\n"

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # Qwen2.5 ChatML usa <|im_end|> como stop token além do EOS
            stop_token_ids = [tokenizer.eos_token_id]
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id and im_end_id != tokenizer.eos_token_id:
                stop_token_ids.append(im_end_id)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.15,
                    eos_token_id=stop_token_ids,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Retorna só o texto gerado (sem o prompt de entrada)
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            return tokenizer.decode(generated, skip_special_tokens=True)

        def __call__(self, prompt: str) -> str:
            return self.invoke(prompt)

    return _LoRALLM()
