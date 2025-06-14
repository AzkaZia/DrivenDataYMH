import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from gensim.models import Word2Vec
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

nltk.data.path.append('assets')
stop_words = set(stopwords.words('english'))
nltk.data.path.append('assets/nltk_data')
nlp = spacy.load('assets/en_core_web_sm')

df_narratives = pd.read_csv('data/test_features.csv')
df_features = pd.read_csv('data/submission_format.csv')
merged_df = pd.merge(df_narratives, df_features, on='uid', how='inner')

merged_df.head()

binary_targets = [
    'DepressedMood', 'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt', 
    'SuicideAttemptHistory', 'SuicideThoughtHistory', 'SubstanceAbuseProblem', 
    'MentalHealthProblem', 'DiagnosisAnxiety', 'DiagnosisDepressionDysthymia', 
    'DiagnosisBipolar', 'DiagnosisAdhd', 'IntimatePartnerProblem', 
    'FamilyRelationship', 'Argument', 'SchoolProblem', 
    'RecentCriminalLegalProblem', 'SuicideNote', 
    'SuicideIntentDisclosed', 'DisclosedToIntimatePartner', 
    'DisclosedToOtherFamilyMember', 'DisclosedToFriend'
]

def clean_text(text):
    doc = nlp(text)
    cleaned_text = " ".join([token.text for token in doc if token.is_alpha])
    return cleaned_text

def tokenize_text(text):
    return nltk.word_tokenize(text)
    
def handle_negations(tokens):
    negation_words = ["not", "no", "never", "n't"]
    transformed_tokens = []
    negate = False
    for token in tokens:
        if token in negation_words:
            negate = True
        elif negate:
            transformed_tokens.append("not_" + token)
            negate = False
        else:
            transformed_tokens.append(token)
    return transformed_tokens

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

def lemmatize_tokens(tokens):
    return [nlp(token)[0].lemma_ for token in tokens]

for col in ['NarrativeLE', 'NarrativeCME']:
    merged_df[col] = merged_df[col].apply(clean_text)
    merged_df[col] = merged_df[col].apply(tokenize_text)
    merged_df[col] = merged_df[col].apply(handle_negations)
    merged_df[col] = merged_df[col].apply(remove_stopwords)
    merged_df[col] = merged_df[col].apply(lemmatize_tokens)

merged_df.head()
word2vec_model = Word2Vec.load('assets/word2vec_model.model')

def get_embedding(tokens):
    return np.mean([word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv] or [np.zeros(50)], axis=0)

merged_df['NarrativeLE_emb'] = merged_df['NarrativeLE'].apply(get_embedding)
merged_df['NarrativeCME_emb'] = merged_df['NarrativeCME'].apply(get_embedding)

le_embeddings = np.vstack(merged_df['NarrativeLE_emb'].values)
cme_embeddings = np.vstack(merged_df['NarrativeCME_emb'].values)

le_embeddings_df = pd.DataFrame(le_embeddings, columns=[f'narrative_le_{i}' for i in range(le_embeddings.shape[1])])
cme_embeddings_df = pd.DataFrame(cme_embeddings, columns=[f'narrative_cme_{i}' for i in range(cme_embeddings.shape[1])])

merged_df = pd.concat([merged_df, le_embeddings_df, cme_embeddings_df], axis=1)

combined_embeddings = np.hstack([le_embeddings, cme_embeddings])

scaler = StandardScaler()
combined_embeddings_scaled = scaler.fit_transform(combined_embeddings)

model_main = joblib.load('assets/multi_target_xgb.pkl')
model_injury = tf.keras.models.load_model('assets/best_injury_model.h5')
model_weapon = tf.keras.models.load_model('assets/best_weapon_model.h5')

binary_pred = model_main.predict(combined_embeddings_scaled)
binary_pred = (binary_pred > 0.5).astype(int)

injury_pred = np.argmax(model_injury.predict(combined_embeddings_scaled), axis=1) + 1
weapon_pred = np.argmax(model_weapon.predict(combined_embeddings_scaled), axis=1) + 1

Predicted_Class = [
    'DepressedMood', 'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt', 
    'SuicideAttemptHistory', 'SuicideThoughtHistory', 'SubstanceAbuseProblem', 
    'MentalHealthProblem', 'DiagnosisAnxiety', 'DiagnosisDepressionDysthymia', 
    'DiagnosisBipolar', 'DiagnosisAdhd', 'IntimatePartnerProblem', 
    'FamilyRelationship', 'Argument', 'SchoolProblem', 
    'RecentCriminalLegalProblem', 'SuicideNote', 
    'SuicideIntentDisclosed', 'DisclosedToIntimatePartner', 
    'DisclosedToOtherFamilyMember', 'DisclosedToFriend'
]

uid_df = merged_df[['uid']]
submission_df = pd.DataFrame(data=binary_pred, columns=Predicted_Class)
submission_df['InjuryLocationType'] = injury_pred
submission_df['WeaponType1'] = weapon_pred

submission_df.insert(0, 'uid', uid_df['uid'])

submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' has been created.")