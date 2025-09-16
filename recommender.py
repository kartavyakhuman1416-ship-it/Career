import pandas as pd
import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#files
DATA_PATH="dataset.csv"
EMBED_CACHE="embeddings_cache.pkl"
MODEL_NAME='all-MiniLM-L6-V2' # SMALL and fast
#load dataset
df=pd.read_csv(DATA_PATH)
model=SentenceTransformer(MODEL_NAME)
#embedding for data set
def compute_embedding(df,cache_path=EMBED_CACHE):
    if os.path.exists(cache_path):
        try:
            with open(cache_path,'rb') as f:
                cache=pickle.load(f)
            if cache.get('n_entries')==len(df) and cache.get('model')==MODEL_NAME:
                return cache['embeddings']
        except Exception:
            pass
    texts=df['Profile'].astype(str).tolist()
    embeddings=model.encode(texts,convert_to_numpy=True,show_progress_bar=True)
    cache={
        'model':MODEL_NAME,
        'n_entries': len(df),
        'embeddings':embeddings
    }
    with open(cache_path,'wb') as f:
        pickle.dump(cache,f)
    return embeddings
career_embeddings=compute_embedding(df)
def recommend_career(user_skills,strong_subjects,top_k=5):
    """Return top_k career recommendation for the given user input.
    user_skills: comma-seperated string
    strong_subjects: comma-seperated string
    """
    user_profile=f"Skills:{user_skills}. Subjects:{strong_subjects}."
    user_emb=model.encode([user_profile],convert_to_numpy=True)[0]
    sims = cosine_similarity([user_emb],career_embeddings)[0]
    top_idx=np.argsort(sims)[-top_k:][::-1]
    result =[]
    for i in top_idx:
        result.append({
            'Career': df.iloc[i]['Career'],
            'Category':df.iloc[i].get('Category',''),
            'Profile': df.iloc[i]['Profile'],
            'Match%':round(float(sims[i])*100,2)
         })
    return result
if __name__=='__main__':
    skills=input('Enter your skills(comma-seperated):') or 'Python,Communication,Marketing'
    subs=input('Enter interested subjects (comma-seperated):') or 'Math,IT,Physics,Biology,Chemistry'
    recs=recommend_career(skills,subs,top_k=5)
    print('\nTop recommendations:')
    for r in recs:
        print(r)
