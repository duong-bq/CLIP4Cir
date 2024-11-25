import google.generativeai as genai
from google.generativeai.types import RequestOptions
from google.api_core import retry
from google.api_core.exceptions import ResourceExhausted
import typing_extensions as typing
from PIL import Image
import time
import json
from tqdm import tqdm

genai.configure(api_key="AIzaSyA3VTlvsjnkrSrcrB31YyG3t5XsZ1eNX7U")

class Captions(typing.TypedDict):
    caption1: str
    caption2: str

class Caption(typing.TypedDict):
    caption: str

PROMPT = '''
Write me a single sentence describing how to modify outfit 1 to outfit 2. No more than 20 words.
EXAMPLE:
{"caption": "has orange color instead of red and shorter sleeves"}
EXAMPLE:
{"caption": "one side dress, darker color and have red design"}
EXAMPLE:
{"caption": "do not have any text in back, only have write color"}
'''

API_KEY = ["AIzaSyA3VTlvsjnkrSrcrB31YyG3t5XsZ1eNX7U",
           "AIzaSyAKxFpXa63Vbx46bOs4mIzESRp-NvFLKvY",
           "AIzaSyA15g0YPRtxvZgsx-VomaiLWibkyFOochs",
           "AIzaSyB8jP8_xQykCzS2he-9hItTaga-xH95Kf4",
           "AIzaSyCRtrmAbehRz7_t-FAHMJSk0BpCyETUbWc",
           "AIzaSyDzJf-fpq44sWrlAnKwaJGlyl47_0rMjgw",]

gemini = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.GenerationConfig(
                    response_mime_type='application/json',
                    response_schema=Caption,
                    max_output_tokens=77
                ))

if __name__ == "__main__":
    START_TIME = time.time()
    with open("fashionIQ_dataset\captions\cap.extend_clip.train.json") as f:
        dress_triplets = json.load(f)
    f_out = open("fashionIQ_dataset\captions\cap.gemini.train3.txt", "w", encoding='utf-8')
    key_number = 0 # bắt đầu với 'duongbuiq'
    start_time_api_key = {}
    for i in range(len(API_KEY)):
        start_time_api_key[i] = 0
    for i, item in enumerate(tqdm(dress_triplets[3731:])):
        try:
            candidate = Image.open(f"fashionIQ_dataset\images\{item['candidate']}.png")
            target = Image.open(f"fashionIQ_dataset\images\{item['target']}.png")
            response = gemini.generate_content(contents=[candidate, target, PROMPT],
                                               request_options=RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)))
            write_content = {
                "id": i + 3731,
                "candiate": item['candidate'],
                "target": item['target'],
                "caption": [json.loads(response.text)['caption']]
            }
            f_out.write(json.dumps(write_content, indent=4, ensure_ascii=False) + '\n')
            # Sau 15 requests thì đổi API_KEY
            if (i+1) % 15 == 0:
                key_number = (key_number + 1) % len(API_KEY)
                end_time_api_key = time.time()
                time.sleep(max(0, 60 - (end_time_api_key - start_time_api_key[key_number])))
                start_time_api_key[key_number] = end_time_api_key
                genai.configure(api_key=API_KEY[key_number % len(API_KEY)])
                gemini = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    generation_config=genai.GenerationConfig(
                        response_mime_type='application/json',
                        response_schema=Caption,
                        max_output_tokens=77
                    ))
        except ResourceExhausted:
            print(API_KEY[key_number], "exhausted")
            key_number = (key_number + 1) % len(API_KEY)
            end_time_api_key = time.time()
            if start_time_api_key[key_number] != 0:
                time.sleep(max(0, 60 - (end_time_api_key - start_time_api_key[key_number])))
            start_time_api_key[key_number] = end_time_api_key
            genai.configure(api_key=API_KEY[key_number % len(API_KEY)])
            gemini = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.GenerationConfig(
                    response_mime_type='application/json',
                    response_schema=Caption,
                    max_output_tokens=77
                ))
    f_out.close()
    END_TIME = time.time()
    print(f"Gen data in {END_TIME - START_TIME} seconds")