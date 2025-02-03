import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")    #A

prompt_text = """
You are a BFSI compliance AI. 
Summarize key obligations from:
'Under Section 502(b), lenders must
disclose risk-based interest rates,
and keep loan records for 7 years...'
"""                                           #B

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",                     
  messages=[                                 
    {"role": "system",                       
     "content": "You are a BFSI compliance 
                 assistant."},               #C
    {"role": "user", "content": prompt_text}
  ],
  temperature=0.2
)

summary = response["choices"][0]["message"]     
          ["content"]                          #D
print("Compliance Summary:", summary)          #E

