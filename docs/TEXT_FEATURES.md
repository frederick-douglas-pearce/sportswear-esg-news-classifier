# Text Feature Extraction Methods

This document provides a high-level overview of the natural language processing (NLP) techniques used to convert raw text into numerical features that machine learning models can process. These methods are fundamental to text classification but may not be covered in standard ML courses.

## The Challenge: Text → Numbers

Machine learning models work with numerical data, but news articles are text. The feature extraction pipeline transforms text into meaningful numerical representations while preserving semantic information about brands, ESG topics, and context.

```
Raw Text ──► Preprocessing ──► Feature Extraction ──► Numerical Matrix ──► ML Model
              (cleaning)         (TF-IDF, NER,         (X matrix)          (classify)
                                  embeddings)
```

## TF-IDF (Term Frequency-Inverse Document Frequency)

**What it does:** Converts text into a sparse matrix where each column represents a word/phrase and each value indicates how important that term is to the document.

**How it works:**
- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: Penalizes common words that appear everywhere (e.g., "the", "and")
- **TF-IDF = TF × IDF**: Words that are frequent in a document but rare overall get higher scores

**Example:**
```
Document: "Nike releases sustainability report on carbon emissions"

TF-IDF scores (simplified):
  "sustainability": 0.45  (rare, domain-specific → high score)
  "nike":           0.38  (brand name, moderately rare)
  "releases":       0.12  (common action verb → lower score)
  "the":            0.00  (stop word, filtered out)
```

**In this project:** Used in the EP (ESG Pre-filter) classifier with n-grams (1-2 word phrases) and vocabulary limited to top 10,000 terms.

## LSA (Latent Semantic Analysis) / Truncated SVD

**What it does:** Reduces the high-dimensional TF-IDF matrix to a smaller set of "topics" that capture semantic relationships between words.

**Why it's needed:** A TF-IDF matrix with 10,000 words creates 10,000 features—too many for efficient training. LSA compresses this to ~100-200 components while preserving the most important patterns.

**How it works:**
- Uses Singular Value Decomposition (SVD) to find latent "concepts"
- Words that co-occur in similar contexts are grouped together
- Reduces dimensionality while capturing semantic similarity

**Example:**
```
Original TF-IDF: 10,000 features (one per word)
After LSA (200 components): 200 features (latent topics)

Component 47 might capture: "carbon" + "emissions" + "climate" + "footprint"
Component 103 might capture: "workers" + "factory" + "wages" + "labor"
```

**In this project:** The EP classifier uses LSA with 200 components to reduce TF-IDF features while retaining ESG-relevant semantic patterns.

## NER (Named Entity Recognition)

**What it does:** Identifies and classifies named entities in text—people, organizations, locations, etc.—using a pre-trained language model (spaCy).

**Why it's useful for this project:** Helps distinguish between brand name contexts:
- "Puma launches new shoe" → ORG (organization) = sportswear brand ✓
- "A puma was spotted in the mountains" → No ORG = animal, not the brand ✗

**Features extracted:**
- **Brand-as-ORG**: Is the brand name recognized as an organization entity?
- **Proximity features**: How close is the brand mention to other organization entities?
- **Context features**: What other entities appear near the brand name?

**Example:**
```
Text: "Nike CEO John Donahoe announced partnerships with Adidas and Puma"

NER output:
  Nike     → ORG (organization)
  John Donahoe → PERSON
  Adidas   → ORG
  Puma     → ORG

Features: brand_as_org=True, nearby_orgs=2, has_person=True
```

**In this project:** The FP (False Positive) classifier uses NER features to detect whether brand names appear in sportswear-related contexts.

## Sentence Embeddings (Sentence-Transformers)

**What it does:** Converts entire sentences or documents into dense numerical vectors (typically 384-768 dimensions) that capture semantic meaning.

**How it differs from TF-IDF:**
- TF-IDF: Sparse, high-dimensional, based on word counts
- Embeddings: Dense, lower-dimensional, based on meaning

**Key advantage:** Semantically similar texts have similar embeddings, even if they use different words.

**Example:**
```
Text A: "Nike reduces carbon emissions by 30%"
Text B: "Athletic footwear company cuts greenhouse gases"

TF-IDF similarity: LOW (few shared words)
Embedding similarity: HIGH (same meaning)
```

**Models used:**
- `all-MiniLM-L6-v2`: Fast, 384-dimensional embeddings (used in FP classifier)
- `text-embedding-3-small`: OpenAI model, 1536 dimensions (used for evidence matching)

**In this project:** The FP classifier concatenates sentence embeddings with NER features for richer text representation.

## Feature Combination Strategy

The classifiers combine multiple feature types for robust performance:

**FP Classifier (False Positive Detection):**
```
┌─────────────────────────────────────────────────────────────────┐
│ Input: Article title + content + brand name                     │
├─────────────────────────────────────────────────────────────────┤
│ Feature Groups:                                                  │
│ ├── Sentence Embeddings (384 dim) - semantic meaning            │
│ ├── NER Context (12 features) - entity recognition              │
│ ├── Proximity Features (6 features) - brand/entity relationships│
│ └── FP Indicators (13 features) - domain-specific patterns      │
│                                                                  │
│ Total: ~415 features → Random Forest → is_sportswear (0/1)     │
└─────────────────────────────────────────────────────────────────┘
```

**EP Classifier (ESG Pre-filter):**
```
┌─────────────────────────────────────────────────────────────────┐
│ Input: Article title + content + metadata                        │
├─────────────────────────────────────────────────────────────────┤
│ Feature Groups:                                                  │
│ ├── TF-IDF + LSA (200 dim) - document topics                    │
│ ├── ESG Vocabulary Counts (4 features) - category keywords      │
│ └── Metadata Features (varies) - source, category               │
│                                                                  │
│ Total: ~210 features → Logistic Regression → has_esg (0/1)     │
└─────────────────────────────────────────────────────────────────┘
```

## Why These Methods?

| Method | Strength | Trade-off |
|--------|----------|-----------|
| TF-IDF + LSA | Fast, interpretable, captures topic patterns | Loses word order, limited semantic understanding |
| NER | Identifies entities, provides context | Requires pre-trained model, language-specific |
| Sentence Embeddings | Rich semantic representation | Slower, less interpretable, larger model size |

The project uses **different methods for different tasks**:
- **EP classifier**: TF-IDF + LSA (fast, topic-focused for ESG detection)
- **FP classifier**: Embeddings + NER (semantic understanding for brand disambiguation)

This hybrid approach balances performance, interpretability, and computational cost.
