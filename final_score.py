import os
import re
import tempfile
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import spacy
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from spacy.matcher import PhraseMatcher
import torch
torch._classes = None

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

nlp = spacy.load("en_core_web_lg")

skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


# === 1. File Extraction Functions ===
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text.strip()

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF files are allowed.")

# === 2. Preprocessing Function ===
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === 3. Chunking Function ===
def chunk_text(text, chunk_size=3):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text.strip()) if s.strip()]
    return [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

# === 4. Semantic Similarity Score ===
def semantic_score(jd_text, resume_text):
    jd_chunks = chunk_text(jd_text)
    resume_chunks = chunk_text(resume_text)

    jd_embs = model.encode(jd_chunks, normalize_embeddings=True)
    resume_embs = model.encode(resume_chunks, normalize_embeddings=True)

    max_scores = []
    for jd_emb in jd_embs:
        cos_sim = util.cos_sim(jd_emb, resume_embs)[0]
        max_scores.append(cos_sim.max().item())

    return np.mean(max_scores)

# === 5. Keyword Matching Score (TF-IDF based) ===
def keyword_score(jd_text, resume_text):
    corpus = [jd_text, resume_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return sim_matrix[0][0]

def extract_skills_from_jd(text):
    annotations = skill_extractor.annotate(text)
    return sorted(set(skill['doc_node_value'] for skill in annotations['results']['full_matches']))

def extract_skills_from_resume(text):
    annotations = skill_extractor.annotate(text)
    return sorted(set(skill['doc_node_value'] for skill in annotations['results']['full_matches']))

# === 6. Skill Overlap ===
def compute_skill_overlap(jd_text, resume_text, keyword_list):
    matches = [kw for kw in keyword_list if kw.lower() in resume_text.lower()]
    return len(matches) / len(keyword_list) if keyword_list else 0.0

# === 7. Hybrid Scoring Function ===
def hybrid_score(jd_text, resume_text, model, keyword_list, semantic_weight=0.6):
    semantic_part = semantic_score(jd_text, resume_text)
    keyword_part = keyword_score(jd_text, resume_text)
    skill_overlap_part = compute_skill_overlap(jd_text, resume_text, keyword_list)

    if skill_overlap_part == 0:
        hybrid = 0
    else:
        hybrid = (semantic_part * semantic_part) + ((1 - semantic_part) * keyword_part) + (0.2 * skill_overlap_part)
    
    return hybrid, semantic_part, keyword_part, skill_overlap_part

def get_domain_keywords(domain):
    domain_keywords = {
        "data_engineer": ["data pipeline", "sql", "python", "big data", "etl", "spark", "hadoop", "data warehouse"],
        "hr": ["recruitment", "onboarding", "payroll", "employee benefits", "performance management", "hr software"],
        "software_engineer": ["coding", "python", "software design", "agile", "data structures", "algorithms", "frontend", "backend"],
        "marketing": ["digital marketing", "seo", "content strategy", "branding", "advertising", "social media", "market research"],
        "cloud_engineer": ["cloud computing", "aws", "azure", "gcp", "cloud architecture", "kubernetes", "docker", "infrastructure as code"]
    }
    return domain_keywords.get(domain.lower(), [])

def detect_domain(text):
    domains = ["data_engineer", "hr", "software_engineer", "marketing", "cloud_engineer"]
    domain_scores = {domain: 0 for domain in domains}
    
    # Assign a score based on keyword match for each domain
    for domain in domains:
        keywords = get_domain_keywords(domain)
        domain_scores[domain] = compute_skill_overlap(text, text, keywords)
    
    # Return the domain with the highest score
    best_domain = max(domain_scores, key=domain_scores.get)
    return best_domain

def highlight_missing_skills(jd_skills, resume_skills):
    # Normalize to lowercase for comparison
    jd_skills_lower = {skill.lower() for skill in jd_skills}
    resume_skills_lower = {skill.lower() for skill in resume_skills}

    print("\n=== JD Skills (Normalized) ===")
    print(jd_skills_lower)
    print("\n=== Resume Skills (Normalized) ===")
    print(resume_skills_lower)

    # Find skills present in JD but missing from Resume
    missing = jd_skills_lower - resume_skills_lower
    
# # # === 9. File Paths ===
# resume_path = "Resume2.pdf"
# jd_path = "Data Engineer.pdf"

# # # === 10. Load and Preprocess ===
# resume_text_raw = extract_text(resume_path)
# job_desc_text_raw = extract_text(jd_path)

# resume_text_clean = preprocess(resume_text_raw)
# job_desc_text_clean = preprocess(job_desc_text_raw)

# resume_skills = extract_skills_from_resume(resume_text_clean)
# jd_skills = extract_skills_from_jd(job_desc_text_clean)

# missing_skills = highlight_missing_skills(jd_skills, resume_skills)

# if missing_skills:
#     print("\nMissing Skills from Resume:")
#     for skill in missing_skills:
#         print(f"- {skill}")
# else:
#     print("\nNo missing skills found in the resume.")

# print("\n✅ Resume and JD text extracted and cleaned.")

# print("=== Clean Resume Text ===")
# print(resume_text_clean)

# print("\n=== Clean Job Description Text ===")
# print(job_desc_text_clean)

# # # === 11. Dynamic Skills Extraction from JD Text ===
# extracted_skills = extract_skills_from_jd(job_desc_text_clean)

# print("\n=== Extracted Skills from JD ===")
# print(extracted_skills)

# resume_domain = detect_domain(resume_text_clean)
# print(f"\nDetected Resume Domain: {resume_domain.capitalize()}")

# # # === 12. Compute Hybrid Score ===
# hybrid, sem, kw, skill_overlap = hybrid_score(job_desc_text_clean, resume_text_clean, model, extracted_skills)

# print("\n=== Hybrid Confidence Score Breakdown ===")
# print(f" Semantic Score:  {sem*100:.2f}%")
# print(f" Keyword Match:  {kw*100:.2f}%")
# print(f" Skill Overlap:  {skill_overlap*100:.2f}%")
# print(f"\n✅ Final Confidence Score: {hybrid*100:.2f}%")

# print("\n❌ Missing Skills from Resume:")
# if missing_skills:
#     for skill in missing_skills:
#         print(f"- {skill}")
# else:
#     print("🎉 Resume covers all required skills!")
