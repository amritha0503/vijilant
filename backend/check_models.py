"""Quick script to check available Google AI models"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("no API key found")
    exit(1)

genai.configure(api_key=api_key)

print("Available models:")
print("=" * 60)
for model in genai.list_models():
    print(f"\nName: {model.name}")
    print(f"  Display Name: {model.display_name}")
    print(f"  Supported methods: {', '.join(model.supported_generation_methods)}")
