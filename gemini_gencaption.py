import google.generativeai as genai
from google.generativeai.types import RequestOptions
from google.api_core import retry
from google.api_core.exceptions import ResourceExhausted
import typing_extensions as typing
from typing import List, Optional
from pydantic import BaseModel
from PIL import Image
import time
import json
from tqdm import tqdm

# **Type of Item**: Identify the type of item (e.g., t-shirt, polo, dress, blouse, tank top).

class Caption(typing.TypedDict):
    Is_Item: bool
    Type_of_Item: Optional[str] = None
    Gender: Optional[str] = None
    Color: Optional[str] = None
    Pattern: Optional[str] = None
    Design_Elements: Optional[str] = None
    Sleeves: Optional[str] = None
    Collar: Optional[str] = None
    Buttons: Optional[str] = None
    Style_and_Fit: Optional[str] = None
    Material_and_Texture: Optional[str] = None

PROMPT = """Write a passage that describe in detail the dress | shirt | top&tee in the image, focusing on the following aspects:
Gender: Specify whether the item is designed for men, women, or is unisex.
Color: Describe the main colors and any secondary shades? 
Pattern: Describe any patterns (e.g., stripes, polka dots, graphics).
Design Elements: Note any logos, text, or illustrations present on the item.
Sleeves: Describe the style and length of the sleeves (e.g., short, long, cap sleeves) and any unique features (e.g., rolled cuffs, ruffles).
Collar: Detail the type of collar (e.g., crew neck, V-neck, polo collar) and any distinctive characteristics (e.g., ribbed, embellished).
Buttons: If applicable, describe the presence of buttons, their size, color, and placement.
Style and Fit: Comment on the overall style (e.g., casual, sporty) and fit (e.g., loose, fitted).
Material and Texture: Mention the material the item appears to be made from and its texture (e.g., cotton, polyester, soft, rough).

Minimum 100 words"
"""

API_KEY = ["AIzaSyA3VTlvsjnkrSrcrB31YyG3t5XsZ1eNX7U",
           "AIzaSyAKxFpXa63Vbx46bOs4mIzESRp-NvFLKvY",
           "AIzaSyA15g0YPRtxvZgsx-VomaiLWibkyFOochs",
           "AIzaSyB8jP8_xQykCzS2he-9hItTaga-xH95Kf4",
           "AIzaSyCRtrmAbehRz7_t-FAHMJSk0BpCyETUbWc",
           "AIzaSyDzJf-fpq44sWrlAnKwaJGlyl47_0rMjgw",]

genai.configure(api_key=API_KEY[0])

gemini = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.GenerationConfig(
                    # response_mime_type='application/json',
                    # response_schema=Caption,
                    max_output_tokens=512
                ))

if __name__ == "__main__":
    START_TIME = time.time()
    with open("fashionIQ_dataset/captions/cap.extend_clip.train.json") as f:
        dress_triplets = json.load(f)
    f_out = open("fashionIQ_dataset/captions/cap.gemini.caption.txt", "w", encoding='utf-8')
    key_number = 0 # bắt đầu với 'duongbuiq'
    start_time_api_key = {}
    for i in range(len(API_KEY)):
        start_time_api_key[i] = 0
    for i, item in enumerate(tqdm(dress_triplets)):
        try:
            candidate = Image.open(f"fashionIQ_dataset/images/{item['candidate']}.png")
            response = gemini.generate_content(contents=[candidate, PROMPT],
                                               request_options=RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)))
            description = response.text
            if "```json" == description[:7]:
                description = description[7:]
            if "```" == description[-3:]:
                description = description[:-3]
            print(description)
            write_content = {
                "id": i,
                "candiate": item['candidate'],
                "caption": description
            }
            f_out.write(json.dumps(write_content, indent=4, ensure_ascii=False) + '\n')
            # Sau 13 requests thì đổi API_KEY
            if (i+1) % 13 == 0:
                key_number = (key_number + 1) % len(API_KEY)
                genai.configure(api_key=API_KEY[key_number % len(API_KEY)])
                gemini = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    generation_config=genai.GenerationConfig(
                        response_mime_type='application/json',
                        response_schema=Caption,
                        max_output_tokens=77
                    ))
            if i > 5:
                break
        except ResourceExhausted:
            print("Exhausted key:", API_KEY[key_number])
            # key_number = (key_number + 1) % len(API_KEY)
            # end_time_api_key = time.time()
            # if start_time_api_key[key_number] != 0:
            #     time.sleep(max(0, 60 - (end_time_api_key - start_time_api_key[key_number])))
            # start_time_api_key[key_number] = end_time_api_key
            # genai.configure(api_key=API_KEY[key_number % len(API_KEY)])
            # gemini = genai.GenerativeModel(
            #     'gemini-1.5-flash',
            #     generation_config=genai.GenerationConfig(
            #         response_mime_type='application/json',
            #         response_schema=Caption,
            #         max_output_tokens=77
            #     ))
    f_out.close()
    END_TIME = time.time()
    print(f"Gen data in {END_TIME - START_TIME} seconds")