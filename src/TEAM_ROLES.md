# Amazon ML Challenge 2026 - Team Roles & AI "Vibe Coding" Strategy

Welcome to the team repository. We are building a memory-efficient, highly accurate Entity Resolution pipeline to hit Rank 1 in the Business Entity Resolution challenge.

Since we are all using AI coding assistants ("vibe coding"), we must **divide and conquer**. If everyone tries to build the whole pipeline, we will end up with 4 broken, incompatible pipelines.

Instead, one person (the Team Lead) owns the core pipeline, and the other 3 members build "plugins" (specific Python functions) that the Lead integrates.

---

## 👑 Role 1: Core Pipeline & ML Integrator (Team Lead)
**Who is doing this:** The repository owner (User + Antigravity AI)
**Your AI Prompting Focus:** Pipeline architecture, memory management, LightGBM/XGBoost training, and submission generation.

**What this role does:**
1. Maintains the `baseline_fast.py` and `train_model.py` scripts.
2. Manages the strict 16GB RAM constraint using streaming algorithms.
3. Takes the Python functions written by Roles 2, 3, and 4, and plugs them into the main pipeline.
4. Generates the final `matching_results.tsv` and submits it to the leaderboard.

---

## 🔍 Role 2: The "Search Engine" (Blocking) Optimizer
**Your Goal:** Improve our Candidate Recall from 92% to 98% without crashing the 16GB RAM limit.
**The Problem:** Our current search engine only finds candidates if they share an *exact* word (like "Corp" and "Corp"). If one says "Corporation", it might miss it entirely.
**What you deliver:** Python code that improves `find_candidates_from_index()`.

**How to "Vibe Code" this (Prompt your AI):**
> *"I am working on an Entity Resolution problem with 10 million records and a strict 16GB RAM limit. Our current blocking strategy builds an inverted index of exact name tokens. It achieves 92% recall. Write a Python script that implements 'character n-gram blocking' or 'Soundex/Metaphone phonetic blocking' that streams the data to keep memory under 5GB, with the goal of hitting 98% recall."*

---

## 🧬 Role 3: The Feature Engineer (String Matching Expert)
**Your Goal:** Give our ML model smarter ways to compare two messy text strings (especially French addresses and Hindi scripts).
**The Problem:** The ML model only knows what we tell it. If we don't tell it that "St." and "Street" are the same, it gets confused.
**What you deliver:** Python functions that take two strings (`name1`, `name2`, `addr1`, `addr2`) and return similarity scores (floats between 0.0 and 1.0).

**How to "Vibe Code" this (Prompt your AI):**
> *"I need advanced feature engineering functions in Python for Entity Resolution. The dataset contains US, Indian, and French addresses. Write a function using Regex to extract postal codes (US 5-digit, India 6-digit, France 5-digit) from messy address strings and compare them. Write another function that uses the `rapidfuzz` library to compare business names while ignoring common legal suffixes (like LLC, Pvt Ltd, SARL)."*

---

## 🕵️‍♂️ Role 4: The Edge-Case & Error Analyst
**Your Goal:** Find the model's blind spots and destroy them.
**The Problem:** The ML model is 98% accurate on the candidates it sees, but it still makes mistakes. We need to know *why* it makes mistakes.
**What you deliver:** Analysis reports and new Python rules to catch weird edge cases.

**How to "Vibe Code" this (Prompt your AI):**
> *(Feed your AI a CSV of our False Positives — things the model matched but shouldn't have)*
> *"Attached are the false positive matches from our LightGBM model. Analyze these pairs. What patterns is the model getting tricked by? Are they businesses in the same building? Are they franchises? Give me 3 new Python feature functions I can write to help the model distinguish between these specific types of errors."*

---

## 🚀 The Workflow
1. **Roles 2, 3, and 4** prompt their AI to write specific, isolated Python functions (e.g., `def extract_french_postal_code(address):`).
2. They test their function locally to make sure it doesn't have syntax errors.
3. They push their function to this GitHub repo or send the snippet directly to the Team Lead.
4. **The Team Lead** pastes the function into the core pipeline and retrains the model to see how much the Overall Accuracy (F0.5) increases.
5. Repeat until we hit Rank 1.
