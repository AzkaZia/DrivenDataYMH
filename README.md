This model was developed as part of a DrivenData competition using private data. The code is open-source, and results can be reproduced using simulated data.

The competition required submitting fully working code that could generate predictions. I had to learn Docker and Ubuntu — honestly, one of the more challenging things I've done in life because of the time crunch. Although I placed low on the leaderboard, this experience taught me so much and solidified my love for machine learning.

In earlier versions of the project, I experimented with; HuggingFace Transformers, Sentence Transformers, Word2Vec, and nltk tokenizers
I also used multiple modeling strategies including multi-output classifiers.

In the final version, I trained separate models to predict Injury Location and Weapon Type

Notes  
This repository contains the full code used to train and evaluate models during the DrivenData YMH competition.

To respect the competition rules and data privacy:

🔒 Trained model files (.h5, .pkl, .joblib, etc.) are not included

📁 Competition datasets are not shared

🧠 The notebook (finalsubmission.ipynb) includes all the steps needed to retrain models locally

If you'd like to try out the project, you can use your own data or modify the notebook to run on synthetic data.
