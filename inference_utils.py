import torch
import functools

HF_SYSTEM_MARKER = "system"
HF_USER_MARKER = "user"
HF_ASSISTANT_MARKER = "assistant"

def handle_eval_mode(func):
    """Decorator that handles turning on eval mode for inference"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = None
        if args:    
            model = args[0]
        elif kwargs:
            model = kwargs["model"]
        if model:
            recover_train_mode = False
            if model.train:
                recover_train_mode = True
                model.eval()

        result = func(*args, **kwargs)

        if model and recover_train_mode:
            model.train()
        return result
    return wrapper

@handle_eval_mode
@torch.no_grad()
def chat(model, tokenizer, text, device, max_new_tokens=1024, temperature=0.2, top_p=0.9):
    """
    Chat with a model given a user text.
    """
    messages = [
        {"role": HF_SYSTEM_MARKER, "content": ""},
        {"role": HF_USER_MARKER, "content": f"You're a helpful assistant\n\n{text}"},
    ]
    # Format the text into qwen format with <im_start> tokens etc. 
    # Make sure to add generation prompt (assistant token) at the end of the messages.
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, 
                                   max_new_tokens=max_new_tokens, 
                                   do_sample=(temperature > 0), 
                                   temperature=temperature, 
                                   top_p=top_p, 
                                   pad_token_id=tokenizer.eos_token_id
                                   )
    # Grab the assistant answer
    # We skip the input tokens so we only see the newly generated summary
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    out_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return out_text

@handle_eval_mode
@torch.no_grad()
def summarize(model, tokenizer, text, device, max_new_tokens=100):
  """
  Summarize a given text.
  """
  messages = [
      {"role": HF_SYSTEM_MARKER, "content": ""},
      {"role": HF_USER_MARKER, "content": f"Summarize the following text:\n\n{text}"},
  ]
  # 3. Apply the Chat Template
  # This converts the list above into the raw string format Qwen expects 
  # (e.g., <|im_start|>system...)
  input_text = tokenizer.apply_chat_template(
      messages, 
      tokenize=False, 
      add_generation_prompt=True
  )
  model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
  generated_ids = model.generate(
      model_inputs.input_ids,
      max_new_tokens=max_new_tokens,
      do_sample=False      # False = deterministic (better for factual summaries)
  )
  # We skip the input tokens so we only see the newly generated summary
  generated_ids = [
      output_ids[len(input_ids):] 
      for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]
  summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return summary
