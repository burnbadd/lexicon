# Step 2: Imports
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Step 3: Load model and tokenizer
model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
vocab = list(tokenizer.vocab.keys())

# Step 4: Filter valid candidate words
def clean_vocab(vocab):
    return [word for word in vocab 
            if word.isalpha() and len(word) > 2 and not word.startswith("##")]

cleaned_vocab = clean_vocab(vocab)[:10000]  # limit to top 10k words for speed

# Step 5: Define seed words
cooperative_seeds = ['collaborate', 'agree', 'support', 'negotiate', 'compromise', 'assist', 'teamwork', 'align']
uncooperative_seeds = ['oppose', 'criticize', 'refuse', 'attack', 'blame', 'resist', 'argue', 'reject']

# Step 6: Embed seeds and candidates
cooperative_vec = np.mean(model.encode(cooperative_seeds), axis=0)
uncooperative_vec = np.mean(model.encode(uncooperative_seeds), axis=0)
candidate_embeddings = model.encode(cleaned_vocab)

# Step 7: Compute cosine similarity
coop_sim = cosine_similarity([cooperative_vec], candidate_embeddings)[0]
uncoop_sim = cosine_similarity([uncooperative_vec], candidate_embeddings)[0]

# Step 8: Get top-N most similar words
top_n = 200
coop_top = [cleaned_vocab[i] for i in np.argsort(coop_sim)[-top_n:]]
uncoop_top = [cleaned_vocab[i] for i in np.argsort(uncoop_sim)[-top_n:]]

# Optional: Remove overlaps
coop_final = [w for w in coop_top if w not in uncoop_top]
uncoop_final = [w for w in uncoop_top if w not in coop_top]

# Step 9: Output results
print("Top Cooperative Words:\n", coop_final[:30])
print("\nTop Uncooperative Words:\n", uncoop_final[:30])

# Step 10: Output results to csv
import csv

with open('coop_final.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['cooperative_words'])
    for word in coop_final:
        writer.writerow([word])

with open('uncoop_final.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['uncooperative_words'])
    for word in uncoop_final:
        writer.writerow([word])