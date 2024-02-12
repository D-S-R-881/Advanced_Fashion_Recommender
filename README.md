# Advanced Fashion Recommender
This is a fashion recommender system that uses natural language processing (NLP) and computer vision to provide personalized fashion recommendations to users.

<div align="center">
  <img src="https://github.com/D-S-R-881/Advanced_Fashion_Recommender/assets/78027597/34409112-b28b-4a1e-9075-9007e9a38e38" alt="Description of the image" width="800" height="320">
  <p><i>Front-End of the WebSite</i></p>
</div>

<b>Key features:</b>
* Interactive AI bot: The bot can process text and voice prompts from users to understand their style preferences. It uses OpenAI to extract keywords from text and voice prompts. And then uses TF-IDF vectorization To calculate cosine similarity between keywords.

<div align="center">
  <img src="https://github.com/D-S-R-881/Advanced_Fashion_Recommender/assets/78027597/df1b646c-6b26-4ef1-a948-54f35f4b420d" alt="Description of the image" width="700" height="320">
  <p><i>Fashion Bot</i></p>
</div>

<div align="center">
  <img src="https://github.com/D-S-R-881/Advanced_Fashion_Recommender/assets/78027597/85cf4c4c-d2d8-4425-9072-d6f8c10dfc2f" alt="Description of the image" width="800" height="320">
  <p><i>Reults Shown by Fashion Bot for a given Prompt</i></p>
</div>

* Fashion Lens: Users can upload images of products or outfits they like, and the system will recommend similar products. Uses ResNet50 to extract image features and then uses KNN algorithm to recommend similar products to the user

<div align="center">
  <img src="https://github.com/D-S-R-881/Advanced_Fashion_Recommender/assets/78027597/2eb738c5-052d-4ddc-9a57-aff18bdc6919" alt="Description of the image" width="500" height="600">
  <p><i>Results Given by Fashion Lens for a given image input</i></p>
</div>

* Trending Outfits: Outfits similar to the trending outfits based on current fashion trends are recommended in this section.

<div align="center">
  <img src="https://github.com/D-S-R-881/Advanced_Fashion_Recommender/assets/78027597/cd7adc81-4dc4-4606-b1bc-de9d9a2bcc59" alt="Description of the image" width="700" height="320">
  <p><i>Trending Outfits</i></p>
</div>

* HomePage: Outfits similar to user purchase and search history is recommended in HomePage.

<div align="center">
  <img src="https://github.com/D-S-R-881/Advanced_Fashion_Recommender/assets/78027597/96dd59d5-7c14-43b0-84bc-f95a17ea1b3c" alt="Description of the image" width="700" height="320">
  <p><i>Home Page</i></p>
</div>

This fashion recommender system demonstrates the potential of leveraging AI technologies like NLP and computer vision to enhance the user shopping experience. The project successfully delivered features like an interactive AI bot and a "Fashion Lens" for image-based recommendations, catering to diverse user preferences.
