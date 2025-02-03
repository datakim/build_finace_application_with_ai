from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Suppose this BFSI org has disclaimers or specialized text in a small dataset
# Each "input" might be partial policy text, each "target" the updated disclaimers or brand wording.

model_name = "openlm-research/open_llama_7b"                 #A
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Parameter-efficient config for BFSI disclaimers
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],                      #B
    lora_dropout=0.05,
)

peft_model = get_peft_model(base_model, lora_config)         #C

# BFSI text pairs: (input_text, target_text)
bfsidata = [
    ("Loan policy snippet: 'Max LTV is 80%'", 
     "Insert disclaimers on brand style and local law..."),
    ("Compliance text: 'Keep AML logs 7 years'", 
     "Rewrite to add BFSI disclaimers..."),
]
for input_text, target_text in bfsidata:                     #D
    inputs = tokenizer(input_text, return_tensors="pt")
    labels = tokenizer(target_text, return_tensors="pt")["input_ids"]
    outputs = peft_model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()                                          #E
    # Typically you'd call an optimizer.step() here

peft_model.save_pretrained("./bfsifinetuned-lora")           #F
print("Finished BFSI partial fine-tuning.")
