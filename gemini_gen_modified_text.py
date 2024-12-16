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
import re
import multiprocessing


PROMPT = """
You are a customer in a clothing shop. The staff introduces you to a candidate fashion item like the first image. However, you are imagining a more beautiful outfit similar to the second photo. Please tell the staff the features of the item you want to buy. The features can be: color, pattern, length, sleeve, collar, pocket, button, zipper, material, style, fabric, or any other feature that you can see in the image.

Your sentence has to follow the 5 templates below:
1. A similar like this but have <second item feature>. E.g: A dress like this, but with longer sleeves and a shorter skirt.
2. Replace <candidate item feature> with <second item feature>. E.g: Replace the pattern of this shirt with stripes.
3. Add <second item feature>. E.g: Add buttons to this shirt.
4. Remove <candidate item feature>. E.g: Remove the pocket from this shirt.
5. Combine more than one template above. E.g: A polo like this, but with shorter sleeves. And v-neck collar instead of round collar.

Note:
- The staff does not know about the second item because that item is in your imagination, so you need to describe the features to her. Do not mention "the second item" in your sentence.
- You can describe only 1 or 2 features in a sentence. So please choose features that highlight the differences between the two items.
- Your sentence should be clear and concise. No more than 30 words.
- Return 5 sentences as 5 options for the staff to understand your request.
"""

PROMPT2 = """
You are a customer in a clothing shop. The staff introduces you to a candidate fashion item like the first image. However, you are imagining a more beautiful outfit similar to the second photo.

A modified text is a description about how the features of second item is different from the first item. The features can be: color, pattern, length, sleeve, collar, pocket, button, zipper, material, style, fabric, or any other feature that you can see in the image. With the modified text, the staff will help you find item look the the first item a bit but have feature like the modified text
Please write some modified texts that give the staff infomation to help you find the item you want to buy.

Your modified text should follow the 4 templates below:
1. Direct reference the second image feature. For example: "is solid white and buttons up with front pockets"
2. Comparison. For example: "has longer sleeves and is lighter in color"
3. Direct reference and Negation. For example: "is white colored with a graphic and no lace design"
4. Combination all template. For example: "is a solid color with a v-neck, have longer stripe, and no buttons"

Note:
- The staff does not know about the second item because that item is in your imagination, so you need to describe the features to her. Do not mention "the second item" in your sentence.
- You can refer from 2 to 4 features in a sentence. You can choose random features that highlight the differences between the two items.
- Cut out the extra openings in your modified text.
    + Example: "I am looking for a solid color with a v-neck, have longer stripe, and no buttons" -> "is a solid color with a v-neck, have longer stripe, and no buttons"
    + Example: "I want a dress like this, but with longer sleeves and a shorter skirt" -> "has longer sleeves and a shorter skirt"
    + Example: "It should be a a dark background with white hibiscus flowers and bamboo, not this beige and leaf pattern" -> "is a dark background with white hibiscus flowers and bamboo, not this beige and leaf pattern"
- Your sentence should be clear and concise. No more than 30 words.
- Return 4 modified text as 4 options for the staff to understand your request.
"""

API_KEY = "AIzaSyA3VTlvsjnkrSrcrB31YyG3t5XsZ1eNX7U"

genai.configure(api_key=API_KEY)

class ModifiedText(typing.TypedDict):
    option1: str
    option2: str
    option3: str
    option4: str

gemini = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.GenerationConfig(
                    response_mime_type='application/json',
                    response_schema=ModifiedText,
                    max_output_tokens=256
                ))

def gen_modified_text(process_number: int, from_index: int, to_index: int, data_path: str):
    with open(data_path) as f:
        dress_triplets = json.load(f)
    f_out = open(f"temp/modified.gemini.{process_number}.json", "w", encoding='utf-8')
    f_out.write("[\n")
    to_index = min(to_index, len(dress_triplets))
    for i, item in enumerate(tqdm(dress_triplets[from_index:to_index])):
        try:
            candidate = Image.open(f"fashionIQ_dataset/images/{item['candidate']}.png")
            target = Image.open(f"fashionIQ_dataset/images/{item['target']}.png")
            # target = Image.open(f".images/_{item['candidate']}.png")
            response = gemini.generate_content(contents=[candidate, target, PROMPT2],
                                               request_options=RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)))
            modified_text = response.text
            if "```json" == modified_text[:7]:
                modified_text = modified_text[7:]
            if "```" == modified_text[-3:]:
                modified_text = modified_text[:-3]
            # modified_text = re.sub(r'\s+', ' ', modified_text).strip()
            modified_text = json.loads(modified_text)
            modified_text = [value for key, value in modified_text.items()]

            write_content = {
                "id": i + from_index,
                "candidate": item['candidate'],
                "target": item['target'],
                "captions": modified_text
            }

            if i > to_index - from_index - 2:
                f_out.write(json.dumps(write_content, indent=4, ensure_ascii=False) + '\n')
                break
            else:
                f_out.write(json.dumps(write_content, indent=4, ensure_ascii=False) + ',\n')
        except ResourceExhausted:
            print("Exhausted key:", API_KEY)
    f_out.write("]\n")
    f_out.close()


def append_json_file(prefix_path: str, num_file: int, output_file_path: str):
    data = []
    for i in range(num_file):
        with open(f"{prefix_path}.{i}.json") as f:
            data.extend(json.load(f))
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    data_path = "fashionIQ_dataset/captions/cap.dress.train.json"
    with open(data_path) as f:
        dress_triplets = json.load(f)
        SAMPLE_SIZE = len(dress_triplets)
    NUM_PROCESSES = 48
    CHUNK = SAMPLE_SIZE // NUM_PROCESSES + 1
    processes = []
    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(target=gen_modified_text, args=(i, i * CHUNK, (i + 1) * CHUNK, data_path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Done!")

    append_json_file("temp/modified.gemini", NUM_PROCESSES, "fashionIQ_dataset/captions/cap.dress.train.gemini.json")


