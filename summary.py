# summary.py
import re
import io
import json
import time
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import requests
import pdfkit
import tiktoken
from bs4 import BeautifulSoup
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
import fitz
import openai
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

if not firebase_admin._apps:
    try:
        firebase_cred_path = "firebase-adminsdk.json"
        if not os.path.exists(firebase_cred_path):
            raise FileNotFoundError("firebase-adminsdk.json not found in the project directory.")
        with open(firebase_cred_path, 'r') as f:
            creds = json.load(f)
        cred = credentials.Certificate(firebase_cred_path)
        bucket_name = f"{creds['project_id']}.appspot.com"
        logging.debug(f"Constructed bucket name: {bucket_name}")
        firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
        logging.info(f"Firebase initialized with storage bucket: {bucket_name}")
    except Exception as e:
        logging.error(f"Failed to initialize Firebase: {str(e)}")
        raise

db = firestore.client(database_id="statside-summary")

class TextProcessor:
    def __init__(self, model):
        self.model = None
        # Support for three models: GPT-4 (gpt4), GPT-4-mini (openai), and TogetherAI (togetherai)
        if model.lower() in ["openai", "gpt4"]:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key or self.openai_api_key.strip() == "":
                raise ValueError("OPENAI_API_KEY is missing or empty in .env file.")
            if model.lower() == "gpt4":
                self.model = "gpt-3.5-turbo"
            else:
                self.model = "gpt-4o-mini"
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        elif model.lower() == "togetherai":
            self.together_api_key = os.getenv("TOGETHERAI_API_KEY")
            if not self.together_api_key or self.together_api_key.strip() == "":
                raise ValueError("TOGETHERAI_API_KEY is missing or empty in .env file.")
            self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        else:
            raise ValueError(f"Unsupported model selection: {model}. Use 'openai', 'gpt4' or 'togetherai'.")

        logging.info(f"OpenAI API Key: {'Set' if hasattr(self, 'openai_api_key') and self.openai_api_key else 'Not Set'}")
        logging.info(f"TogetherAI API Key: {'Set' if hasattr(self, 'together_api_key') and self.together_api_key else 'Not Set'}")

    def get_base_name_from_link(self, link):
        base_name = re.sub(r"[^\w\-_\. ]", "_", link)
        logging.info(f"Using full URL as base_name: {base_name}")
        return base_name or "default_name"

    def is_google_cache_link(self, link):
        return "webcache.googleusercontent.com" in link

    def is_blank_text(self, text):
        clean_text = re.sub(r"\s+", "", text).strip()
        return len(clean_text) < 100

    def process_image_with_tesseract(self, image):
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            logging.error(f"Error processing image with Tesseract: {str(e)}")
            return ""

    def extract_text_from_pdf_native(self, pdf_bytes):
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text.strip()
        except Exception as e:
            logging.error(f"Error in native PDF extraction: {str(e)}")
            return ""

    def extract_text_from_pdf(self, pdf_content, link):
        base_name = self.get_base_name_from_link(link)
        pdf_bytes = pdf_content.read()
        native_text = self.extract_text_from_pdf_native(pdf_bytes)
        if native_text and not self.is_blank_text(native_text):
            logging.info("Native PDF text extraction succeeded.")
            return native_text
        images = convert_from_bytes(pdf_bytes)
        logging.info(f"OCR fallback: converting {len(images)} pages to images.")
        combined_text = ""
        def process_page(i, img):
            return self.process_image_with_tesseract(img)
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda x: process_page(x[0], x[1]), enumerate(images))
            for text in results:
                combined_text += text + "\n"
        if self.is_blank_text(combined_text):
            return ""
        return combined_text

    def extract_text_from_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ').strip()

    async def async_extract_text_from_url(self, url: str) -> dict:
        try:
            if self.is_google_cache_link(url):
                return {"text": "", "content_type": None, "error": "google_cache"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return {"text": "", "content_type": None, "error": f"HTTP error {response.status}"}
                    content_type = response.headers.get('Content-Type', '').lower()
                    logging.debug(f"URL: {url} returned Content-Type: {content_type}")
                    content = await response.read()
                    if ('application/pdf' in content_type or 'octet-stream' in content_type or url.lower().endswith('.pdf')):
                        text = self.extract_text_from_pdf(io.BytesIO(content), url)
                        if self.is_blank_text(text):
                            return {"text": "", "content_type": "pdf", "error": "blank_pdf"}
                        return {"text": text, "content_type": "pdf", "error": None}
                    elif ('html' in content_type or url.lower().endswith(('.htm', '.html'))):
                        text = self.extract_text_from_html(content)
                        if self.is_blank_text(text):
                            return {"text": "", "content_type": "html", "error": "blank_html"}
                        return {"text": text, "content_type": "html", "error": None}
                    elif 'text/plain' in content_type:
                        text = content.decode('utf-8', errors='ignore').strip()
                        if self.is_blank_text(text):
                            return {"text": "", "content_type": "text", "error": "blank_text"}
                        return {"text": text, "content_type": "text", "error": None}
                    else:
                        return {"text": "", "content_type": None, "error": "unsupported_type"}
        except Exception as e:
            logging.error(f"Error fetching URL {url}: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def process_uploaded_pdf(self, pdf_file, base_name="uploaded_pdf"):
        try:
            if hasattr(pdf_file, "name"):
                base_name = re.sub(r"[^\w\-_\. ]", "_", pdf_file.name)
            pdf_bytes = pdf_file.read()
            if not pdf_bytes:
                return {"text": "", "content_type": "pdf", "error": "Empty PDF file"}
            native_text = self.extract_text_from_pdf_native(pdf_bytes)
            if native_text and not self.is_blank_text(native_text):
                logging.info("Native PDF text extraction succeeded.")
                return {"text": native_text, "content_type": "pdf", "error": None}
            logging.info("Native PDF extraction failed. Falling back to OCR.")
            images = convert_from_bytes(pdf_bytes)
            logging.info(f"OCR fallback: converting {len(images)} page(s) to images.")
            combined_text = ""
            def process_page(i, img):
                return self.process_image_with_tesseract(img)
            with ThreadPoolExecutor() as executor:
                results = executor.map(lambda x: process_page(x[0], x[1]), enumerate(images))
                for text in results:
                    combined_text += text + "\n"
            if self.is_blank_text(combined_text):
                return {"text": "", "content_type": "pdf", "error": "blank_pdf"}
            return {"text": combined_text, "content_type": "pdf", "error": None}
        except Exception as e:
            logging.error(f"Error processing uploaded PDF: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def process_uploaded_html(self, html_file, base_name="uploaded_html"):
        try:
            if hasattr(html_file, "name"):
                base_name = re.sub(r"[^\w\-_\. ]", "_", html_file.name)
            html_bytes = html_file.read()
            if not html_bytes:
                return {"text": "", "content_type": "html", "error": "Empty HTML file"}
            text = self.extract_text_from_html(html_bytes)
            if self.is_blank_text(text):
                return {"text": "", "content_type": "html", "error": "blank_html"}
            return {"text": text, "content_type": "html", "error": None}
        except Exception as e:
            logging.error(f"Error processing uploaded HTML: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def preprocess_text(self, text):
        text = re.sub(r"[\r\n]{2,}", "\n", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def truncate_text(self, text, max_tokens=3000):
        encoding = tiktoken.get_encoding("gpt2")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return encoding.decode(tokens)

    def generate_summary_openai(self, text, custom_prompt):
        if not hasattr(self, 'openai_api_key') or not self.openai_api_key:
            raise ValueError("OpenAI API key is not set. Cannot generate summary with OpenAI model.")
        if not custom_prompt or custom_prompt.strip() == "":
            raise ValueError("Please enter a prompt or generate it through the sample prompt in the web app.")
        text = self.truncate_text(text, max_tokens=4000)
        prompt = custom_prompt + "\n\nText to summarize:\n" + text
        logging.info(f"Sending prompt to OpenAI: {prompt[:100]}...")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500,
            )
            summary = response.choices[0].message.content.strip()
            logging.info(f"Received summary: {summary[:100]}...")
            return {"summary": summary}
        except openai.AuthenticationError:
            raise ValueError("Invalid OpenAI API key provided.")
        except openai.OpenAIError as e:
            logging.error(f"OpenAI API error: {str(e)}")
            raise ValueError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error with OpenAI: {str(e)}")
            raise ValueError(f"Unexpected error generating summary with OpenAI: {str(e)}")

    def generate_summary_togetherai(self, text, custom_prompt):
        if not hasattr(self, 'together_api_key') or not self.together_api_key:
            raise ValueError("TogetherAI API key is not set. Cannot generate summary with TogetherAI model.")
        if not custom_prompt or custom_prompt.strip() == "":
            raise ValueError("Please enter a prompt or generate it through the sample prompt in the web app.")
        text = self.truncate_text(text, max_tokens=4000)
        prompt = custom_prompt + "\n\nText to summarize:\n" + text
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 1500,
            }
            response = requests.post("https://api.together.ai/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 401:
                raise ValueError("Invalid TogetherAI API key provided.")
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"].strip()
            logging.info(f"Received TogetherAI summary: {summary[:100]}...")
            return {"summary": summary}
        except requests.exceptions.HTTPError as e:
            logging.error(f"TogetherAI HTTP error: {str(e)}")
            raise ValueError(f"TogetherAI API error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error with TogetherAI: {str(e)}")
            raise ValueError(f"Unexpected error generating summary with TogetherAI: {str(e)}")

    def generate_summary(self, text, base_name, custom_prompt, user_id, display_name, file_url=None):
        if not custom_prompt or custom_prompt.strip() == "":
            raise ValueError("Please enter a prompt or generate it through the sample prompt in the web app.")
        
        summary_ref = db.collection("users").document(user_id).collection("summaries")
        # Check for an existing summary with the same file, prompt, and model.
        docs = summary_ref.where("base_name", "==", base_name).where("model", "==", self.model).stream()
        for doc in docs:
            existing_data = doc.to_dict()
            if existing_data.get("custom_prompt") == custom_prompt:
                summary_id = doc.id
                logging.info(f"Fetching existing summary for {summary_id} with same file, prompt, and model")
                return {"summary": existing_data["summary"], "input_data": existing_data.get("input_data", ""), "file_url": existing_data.get("file_url", ""), "summary_id": summary_id}
        
        summary_id = f"{base_name}_{int(time.time())}"
        logging.debug(f"Generating new summary for {base_name} with prompt: {custom_prompt[:50]}... and model: {self.model}")
        try:
            if "gpt" in self.model.lower():
                summary = self.generate_summary_openai(text, custom_prompt)
            else:
                summary = self.generate_summary_togetherai(text, custom_prompt)
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Unexpected error during summary generation: {str(e)}")
        if "Error" not in summary["summary"]:
            effective_date_pattern = r"(effective\s+(?:date\s*(?:is|of|:)?|on)|takes\s+effect\s+(?:on)?)\s*[:\s]*(?:(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?[,\s]+\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})"
            match = re.search(effective_date_pattern, text, re.IGNORECASE)
            if match:
                date_str = (match.group(0).split(":", 1)[-1].strip() if ":" in match.group(0)
                            else match.group(0).split("on", 1)[-1].strip() if "on" in match.group(0)
                            else match.group(0).split("effect", 1)[-1].strip())
                summary["summary"] += f"\nThis measure has an effective date of: {date_str}"
        input_data = base_name if base_name.startswith(("http://", "https://")) else base_name
        try:
            summary_data = {
                "summary": summary["summary"],
                "custom_prompt": custom_prompt,
                "model": self.model,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "base_name": base_name,
                "input_data": input_data
            }
            if file_url:
                summary_data["file_url"] = file_url
            summary_ref.document(summary_id).set(summary_data)
            logging.info(f"Saved new summary to Firestore with summary_id: {summary_id}, base_name: {base_name}, model: {self.model}")
        except Exception as e:
            logging.error(f"Failed to save summary to Firestore: {str(e)}")
            raise
        return {"summary": summary["summary"], "input_data": input_data, "file_url": file_url, "summary_id": summary_id}

async def process_input(input_data, model, custom_prompt, user_id, display_name, file_url=None, override_base_name=None):
    try:
        processor = TextProcessor(model=model)
        base_name = None
        if hasattr(input_data, "read"):
            base_name = input_data.name if hasattr(input_data, "name") else "uploaded_file"
            logging.info(f"Processing uploaded file: {base_name}")
            _, ext = os.path.splitext(base_name)
            ext = ext.lower()
            if ext == ".pdf":
                result = processor.process_uploaded_pdf(input_data, base_name=base_name)
            elif ext in [".htm", ".html"]:
                result = processor.process_uploaded_html(input_data, base_name=base_name)
            else:
                return {"error": "Unsupported file type. Please upload a PDF or HTML file.", "model": model}
            if result["error"]:
                return {"error": result["error"], "model": model}
            clean_text = processor.preprocess_text(result["text"])
            input_data_str = base_name
        elif isinstance(input_data, str) and input_data.startswith(("http://", "https://")):
            result = await processor.async_extract_text_from_url(input_data)
            if result["error"]:
                return {"error": result["error"], "model": model}
            clean_text = processor.preprocess_text(result["text"])
            base_name = override_base_name if override_base_name else processor.get_base_name_from_link(input_data)
            input_data_str = input_data
        else:
            return {"error": "Invalid input type. Expected URL, PDF, or HTML file.", "model": model}
        logging.debug(f"Calling generate_summary with base_name={base_name}, custom_prompt={custom_prompt[:50]}...")
        summary = processor.generate_summary(clean_text, base_name, custom_prompt, user_id, display_name, file_url=file_url)
        return {"model": model, "summary": summary["summary"], "input_data": input_data_str, "file_url": file_url, "summary_id": summary["summary_id"]}
    except ValueError as ve:
        logging.error(f"ValueError processing input: {str(ve)}")
        return {"error": f"ValueError: {str(ve)}", "model": model}
    except Exception as e:
        logging.error(f"Exception processing input: {str(e)}")
        return {"error": f"Exception: {str(e)}", "model": model}

if __name__ == "__main__":
    pass
