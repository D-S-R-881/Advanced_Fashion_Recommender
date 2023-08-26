import pickle
import requests
import os

df = pickle.load(open('dataset.pkl','rb'))

# Create a directory to save the images
if not os.path.exists('images'):
    os.makedirs('images')

# Loop through similar indices and extract the first image link from each list
for i in range(7153):
    similar_images = df['image_url'][i]
    try:
        # Download the image
        response = requests.get(similar_images)
        if response.status_code == 200:
            # Generate a unique filename for each image
            image_filename = f'images/image_{i}.jpg'
            
            # Save the image locally
            with open(image_filename, 'wb') as f:
                f.write(response.content)
                
            print(f"Image saved: {image_filename}")
        else:
            print(f"Failed to download image from URL: {similar_images}")
    except (ValueError, SyntaxError):
        pass