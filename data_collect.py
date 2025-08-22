import os
import requests
from serpapi import GoogleSearch

API_KEY = ""



products = [
    "iPhone", "Android phone", "laptop computer", "wireless earbuds", "AirPods",
    "headphones", "TV remote control", "computer mouse", "wireless mouse",
    "mechanical keyboard", "security camera", "smart speaker", "wifi router",
    "smart light bulb", "power bank", "phone charger", "iPhone charger",
    "laptop charger", "USB-C cable", "Lightning cable", "micro USB cable",
    "wireless charger", "car charger", "kitchen blender", "microwave", "toaster",
    "electric kettle", "air fryer", "hand mixer", "stand mixer", "juicer",
    "food scale", "vacuum cleaner", "electric toothbrush", "hair dryer",
    "nail clipper", "smartwatch", "fitness tracker", "blood pressure monitor",
    "digital thermometer", "weighing scale", "DVD player", "soundbar",
    "bluetooth speaker", "CD player", "printer", "scanner", "calculator",
    "air freshener dispenser", "electric fan"
]

BASE_DIR = "images_for_finetuning"
os.makedirs(BASE_DIR, exist_ok=True)

for product in products:
    print(f"\n Searching images for: {product}")

    product_folder = os.path.join(BASE_DIR, product.replace(" ", "_"))
    os.makedirs(product_folder, exist_ok=True)

    params = {
        "engine": "google",
        "q": f"{product} site:amazon.com",  
        "tbm": "isch",                      
        "api_key": API_KEY,
        "num": 5                           
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "images_results" in results:
        for idx, image in enumerate(results["images_results"]):
            try:
                image_url = image["original"]
                response = requests.get(image_url, timeout=10)

                image_path = os.path.join(product_folder, f"{product.replace(' ', '_')}_{idx+1}.jpeg")
                with open(image_path, "wb") as f:
                    f.write(response.content)

                print(f" Saved: {image_path}")

            except Exception as e:
                print(f" Failed to download image {idx+1} for {product}: {e}")
    else:
        print(f" No images found for {product}")

print("\n All images saved successfully inside 'images_for_finetuning' directory!")
