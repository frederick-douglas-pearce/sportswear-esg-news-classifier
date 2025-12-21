"""Feature transformer for FP classifier - sklearn-compatible for deployment."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .preprocessing import (
    SPORTSWEAR_VOCAB,
    clean_text,
    compute_sportswear_vocab_features,
)
from src.data_collection.config import BRANDS


class FPFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible feature transformer for FP classifier.

    Supports multiple feature engineering methods:
    - tfidf_word: TF-IDF with word n-grams (default)
    - tfidf_char: TF-IDF with character n-grams
    - tfidf_lsa: TF-IDF followed by LSA dimensionality reduction
    - doc2vec: Gensim Doc2Vec embeddings
    - sentence_transformer: Pre-trained sentence embeddings
    - hybrid: TF-IDF + domain vocabulary features

    This class is designed to be serialized with joblib for Docker deployment.
    """

    # Valid feature engineering methods
    VALID_METHODS = [
        'tfidf_word',
        'tfidf_char',
        'tfidf_lsa',
        'tfidf_context',  # TF-IDF on context window around brand mentions
        'tfidf_proximity',  # TF-IDF + keyword proximity features
        'tfidf_doc2vec',  # TF-IDF + Doc2Vec embeddings (lexical + semantic)
        'tfidf_ner',  # TF-IDF + NER entity type features
        'tfidf_pos',  # TF-IDF + POS pattern features
        'doc2vec',
        'sentence_transformer',
        'hybrid',
    ]

    # NER entity types that suggest false positives vs sportswear
    # False positive indicators (animals, locations, etc.)
    FP_ENTITY_TYPES = ['ANIMAL', 'GPE', 'LOC', 'FAC', 'NORP', 'EVENT']
    # Sportswear indicators (organizations, products, money/business)
    SW_ENTITY_TYPES = ['ORG', 'PRODUCT', 'MONEY', 'PERCENT', 'CARDINAL']

    # POS patterns that suggest sportswear context
    SPORTSWEAR_POS_PATTERNS = [
        ('VBG', 'NN'),   # "wearing shoes"
        ('JJ', 'NN'),    # "athletic gear"
        ('NN', 'NN'),    # "running shoes"
        ('VB', 'NN'),    # "buy sneakers"
    ]

    # Sportswear-related keywords for proximity features
    SPORTSWEAR_KEYWORDS = [
        # Apparel types
        'shoe', 'shoes', 'sneaker', 'sneakers', 'footwear', 'boot', 'boots',
        'apparel', 'clothing', 'wear', 'sportswear', 'activewear', 'athleisure',
        'jersey', 'jerseys', 'shorts', 'pants', 'jacket', 'jackets',
        # Sports/Activity
        'athletic', 'sports', 'running', 'basketball', 'soccer', 'football',
        'tennis', 'golf', 'training', 'workout', 'fitness', 'gym',
        # Brand/Retail
        'brand', 'retailer', 'store', 'stores', 'shop', 'collection',
        'launch', 'release', 'sponsor', 'sponsorship', 'endorsement',
        # Product terms
        'product', 'products', 'line', 'model', 'design', 'edition',
    ]

    def __init__(
        self,
        method: str = 'tfidf_word',
        # TF-IDF parameters
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        norm: str = 'l2',
        # Character n-gram parameters
        char_ngram_range: tuple = (3, 5),
        char_max_features: int = 5000,
        # LSA parameters
        lsa_n_components: int = 100,
        # Context window parameters (for tfidf_context)
        context_window_words: int = 20,
        # Doc2Vec parameters
        doc2vec_vector_size: int = 100,
        doc2vec_min_count: int = 2,
        doc2vec_epochs: int = 40,
        doc2vec_dm: int = 1,
        doc2vec_window: int = 4,
        # Sentence Transformer parameters
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        # Hybrid parameters
        include_vocab_features: bool = True,
        vocab_window_size: int = 15,
        # Proximity parameters
        proximity_window_size: int = 15,
        # General
        random_state: int = 42,
    ):
        """Initialize the feature transformer.

        Args:
            method: Feature engineering method to use
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for word TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            sublinear_tf: Apply sublinear TF scaling
            norm: Normalization method ('l1', 'l2', or None)
            char_ngram_range: N-gram range for character TF-IDF
            char_max_features: Max features for character TF-IDF
            lsa_n_components: Number of LSA components
            context_window_words: Words before/after brand mention for context
            doc2vec_vector_size: Doc2Vec embedding dimension
            doc2vec_min_count: Minimum word count for Doc2Vec
            doc2vec_epochs: Training epochs for Doc2Vec
            doc2vec_dm: Doc2Vec training mode (1=DM, 0=DBOW)
            doc2vec_window: Context window for Doc2Vec
            sentence_model_name: Name of sentence-transformers model
            include_vocab_features: Whether to include domain vocab features
            vocab_window_size: Context window for vocab features
            proximity_window_size: Words to check for keyword proximity
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.lsa_n_components = lsa_n_components
        self.context_window_words = context_window_words
        self.doc2vec_vector_size = doc2vec_vector_size
        self.doc2vec_min_count = doc2vec_min_count
        self.doc2vec_epochs = doc2vec_epochs
        self.doc2vec_dm = doc2vec_dm
        self.doc2vec_window = doc2vec_window
        self.sentence_model_name = sentence_model_name
        self.include_vocab_features = include_vocab_features
        self.vocab_window_size = vocab_window_size
        self.proximity_window_size = proximity_window_size
        self.random_state = random_state

        # Fitted components (set during fit)
        self._tfidf = None
        self._tfidf_char = None
        self._lsa = None
        self._doc2vec_model = None
        self._sentence_model = None
        self._spacy_model = None  # For NER/POS features
        self._vocab_scaler = None  # For scaling vocab features in hybrid mode
        self._proximity_scaler = None  # For scaling proximity features
        self._doc2vec_scaler = None  # For scaling Doc2Vec in combined mode
        self._ner_scaler = None  # For scaling NER features
        self._pos_scaler = None  # For scaling POS features
        self._is_fitted = False

    def _validate_method(self) -> None:
        """Validate that the method is supported."""
        if self.method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                f"Must be one of: {self.VALID_METHODS}"
            )

    def _create_tfidf_word(self) -> TfidfVectorizer:
        """Create word-based TF-IDF vectorizer."""
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            norm=self.norm,
            stop_words='english',
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
        )

    def _create_tfidf_char(self) -> TfidfVectorizer:
        """Create character-based TF-IDF vectorizer."""
        return TfidfVectorizer(
            analyzer='char',
            ngram_range=self.char_ngram_range,
            max_features=self.char_max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            norm=self.norm,
            lowercase=True,
        )

    def _preprocess_texts(self, X: Union[List[str], np.ndarray]) -> List[str]:
        """Preprocess texts by applying text cleaning.

        Args:
            X: Input texts (list or array of strings)

        Returns:
            List of cleaned text strings
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()

        # Handle pandas Series
        if hasattr(X, 'tolist'):
            X = X.tolist()

        return [clean_text(str(text)) if text else "" for text in X]

    def _extract_brands_from_text(self, text: str) -> List[str]:
        """Extract potential brand mentions from text for vocab features.

        Uses the full BRANDS list from config to ensure all tracked brands
        are detected for feature extraction.
        """
        text_lower = text.lower()
        found_brands = []
        for brand in BRANDS:
            if brand.lower() in text_lower:
                found_brands.append(brand)
        return found_brands

    def _extract_context_windows(self, text: str, window_size: int = 20) -> str:
        """Extract text within N words of each brand mention.

        Args:
            text: Full article text
            window_size: Number of words before/after brand to include

        Returns:
            Concatenated context windows around all brand mentions
        """
        brands = self._extract_brands_from_text(text)
        if not brands:
            # No brands found - return empty or full text?
            # Return empty to focus only on brand-adjacent context
            return ""

        words = text.split()
        if not words:
            return ""

        # Find positions of all brand mentions
        context_windows = []
        text_lower = text.lower()

        for brand in brands:
            brand_lower = brand.lower()
            # Find all occurrences of this brand
            start_pos = 0
            while True:
                pos = text_lower.find(brand_lower, start_pos)
                if pos == -1:
                    break

                # Convert character position to word index (approximate)
                # Count words before this position
                words_before = text[:pos].split()
                word_idx = len(words_before)

                # Extract window
                start_idx = max(0, word_idx - window_size)
                end_idx = min(len(words), word_idx + window_size + 1)
                window = words[start_idx:end_idx]
                context_windows.append(' '.join(window))

                start_pos = pos + len(brand_lower)

        # Combine all context windows
        return ' '.join(context_windows)

    def _extract_all_context_windows(self, texts: List[str]) -> List[str]:
        """Extract context windows for all texts.

        Args:
            texts: List of text strings

        Returns:
            List of context window strings
        """
        return [
            self._extract_context_windows(text, self.context_window_words)
            for text in texts
        ]

    def _compute_vocab_features(self, texts: List[str]) -> np.ndarray:
        """Compute domain vocabulary features for all texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, n_vocab_features)
        """
        features_list = []
        for text in texts:
            brands = self._extract_brands_from_text(text)
            features = compute_sportswear_vocab_features(
                text, brands, window_size=self.vocab_window_size
            )
            features_list.append(list(features.values()))

        return np.array(features_list)

    def _compute_proximity_features(self, texts: List[str]) -> np.ndarray:
        """Compute keyword proximity features for all texts.

        For each text, computes:
        - min_distance: Minimum word distance from any brand to any sportswear keyword
        - avg_distance: Average distance across all brand mentions
        - keyword_count_near: Count of sportswear keywords within proximity window
        - has_keyword_near: Binary flag if any keyword within proximity window

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 4)
        """
        features_list = []
        keywords_set = set(kw.lower() for kw in self.SPORTSWEAR_KEYWORDS)

        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            brands = self._extract_brands_from_text(text)

            if not brands or not words:
                # No brands found - use default values
                features_list.append([100.0, 100.0, 0, 0])
                continue

            # Find word indices of all brand mentions
            brand_word_indices = []
            for brand in brands:
                brand_lower = brand.lower()
                start_pos = 0
                while True:
                    pos = text_lower.find(brand_lower, start_pos)
                    if pos == -1:
                        break
                    # Approximate word index
                    word_idx = len(text_lower[:pos].split())
                    brand_word_indices.append(word_idx)
                    start_pos = pos + len(brand_lower)

            if not brand_word_indices:
                features_list.append([100.0, 100.0, 0, 0])
                continue

            # Find word indices of all sportswear keywords
            keyword_indices = []
            for i, word in enumerate(words):
                # Strip punctuation for matching
                word_clean = word.strip('.,!?;:\'"()[]{}')
                if word_clean in keywords_set:
                    keyword_indices.append(i)

            if not keyword_indices:
                # No sportswear keywords found
                features_list.append([100.0, 100.0, 0, 0])
                continue

            # Compute distances from each brand mention to nearest keyword
            min_distances = []
            keywords_near_count = 0
            for brand_idx in brand_word_indices:
                distances_to_keywords = [abs(brand_idx - kw_idx) for kw_idx in keyword_indices]
                min_dist = min(distances_to_keywords)
                min_distances.append(min_dist)

                # Count keywords within proximity window
                keywords_near_count += sum(
                    1 for kw_idx in keyword_indices
                    if abs(brand_idx - kw_idx) <= self.proximity_window_size
                )

            min_distance = min(min_distances)
            avg_distance = sum(min_distances) / len(min_distances)
            has_keyword_near = 1 if min_distance <= self.proximity_window_size else 0

            features_list.append([min_distance, avg_distance, keywords_near_count, has_keyword_near])

        return np.array(features_list)

    def _load_spacy_model(self) -> None:
        """Load spaCy model for NER/POS features."""
        if self._spacy_model is None:
            import spacy
            self._spacy_model = spacy.load('en_core_web_sm')

    def _compute_ner_features(self, texts: List[str]) -> np.ndarray:
        """Compute NER-based features for all texts.

        For each text, computes entity type counts near brand mentions:
        - fp_entity_count: Count of false-positive-indicating entities near brand
        - sw_entity_count: Count of sportswear-indicating entities near brand
        - fp_entity_ratio: Ratio of FP entities to total entities
        - has_animal_near: Binary flag if ANIMAL entity near brand
        - has_location_near: Binary flag if GPE/LOC entity near brand
        - has_org_near: Binary flag if ORG entity near brand

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 6)
        """
        self._load_spacy_model()
        features_list = []

        for text in texts:
            brands = self._extract_brands_from_text(text)
            if not brands or not text.strip():
                features_list.append([0, 0, 0.5, 0, 0, 0])
                continue

            # Process text with spaCy
            doc = self._spacy_model(text[:100000])  # Limit text length for performance

            # Find brand positions
            text_lower = text.lower()
            brand_char_positions = []
            for brand in brands:
                brand_lower = brand.lower()
                start_pos = 0
                while True:
                    pos = text_lower.find(brand_lower, start_pos)
                    if pos == -1:
                        break
                    brand_char_positions.append((pos, pos + len(brand_lower)))
                    start_pos = pos + len(brand_lower)

            if not brand_char_positions:
                features_list.append([0, 0, 0.5, 0, 0, 0])
                continue

            # Count entities near brand mentions
            fp_count = 0
            sw_count = 0
            has_animal = 0
            has_location = 0
            has_org = 0
            window_chars = self.proximity_window_size * 6  # Approx chars per word

            for ent in doc.ents:
                # Check if entity is near any brand mention
                for brand_start, brand_end in brand_char_positions:
                    if (abs(ent.start_char - brand_end) <= window_chars or
                        abs(ent.end_char - brand_start) <= window_chars):
                        # Entity is near brand
                        if ent.label_ in self.FP_ENTITY_TYPES:
                            fp_count += 1
                            if ent.label_ == 'ANIMAL':
                                has_animal = 1
                            elif ent.label_ in ('GPE', 'LOC'):
                                has_location = 1
                        elif ent.label_ in self.SW_ENTITY_TYPES:
                            sw_count += 1
                            if ent.label_ == 'ORG':
                                has_org = 1
                        break  # Count entity once

            total_count = fp_count + sw_count
            fp_ratio = fp_count / total_count if total_count > 0 else 0.5

            features_list.append([fp_count, sw_count, fp_ratio, has_animal, has_location, has_org])

        return np.array(features_list)

    def _compute_pos_features(self, texts: List[str]) -> np.ndarray:
        """Compute POS-based features for all texts.

        For each text, computes POS patterns near brand mentions:
        - noun_after_brand: Count of nouns immediately after brand
        - verb_before_brand: Count of verbs before brand
        - adj_before_brand: Count of adjectives before brand
        - action_verb_near: Count of action verbs (VBD, VBZ) near brand
        - sportswear_pattern_count: Count of sportswear-indicating POS patterns

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 5)
        """
        self._load_spacy_model()
        features_list = []

        for text in texts:
            brands = self._extract_brands_from_text(text)
            if not brands or not text.strip():
                features_list.append([0, 0, 0, 0, 0])
                continue

            # Process text with spaCy
            doc = self._spacy_model(text[:100000])

            # Find brand token positions
            text_lower = text.lower()
            brand_token_indices = []
            for i, token in enumerate(doc):
                token_text_lower = token.text.lower()
                for brand in brands:
                    if brand.lower() in token_text_lower or token_text_lower in brand.lower():
                        brand_token_indices.append(i)
                        break

            if not brand_token_indices:
                features_list.append([0, 0, 0, 0, 0])
                continue

            noun_after = 0
            verb_before = 0
            adj_before = 0
            action_verb_near = 0
            sportswear_patterns = 0

            for brand_idx in brand_token_indices:
                # Check token after brand
                if brand_idx + 1 < len(doc):
                    next_token = doc[brand_idx + 1]
                    if next_token.pos_ == 'NOUN':
                        noun_after += 1

                # Check token before brand
                if brand_idx > 0:
                    prev_token = doc[brand_idx - 1]
                    if prev_token.pos_ == 'VERB':
                        verb_before += 1
                    elif prev_token.pos_ == 'ADJ':
                        adj_before += 1

                # Check for action verbs nearby (past tense often indicates non-brand usage)
                window = 5
                start = max(0, brand_idx - window)
                end = min(len(doc), brand_idx + window + 1)
                for i in range(start, end):
                    if doc[i].tag_ in ('VBD', 'VBZ', 'VBP') and doc[i].pos_ == 'VERB':
                        action_verb_near += 1

                # Check for sportswear-like patterns nearby
                for i in range(start, end - 1):
                    tag_pair = (doc[i].tag_[:2], doc[i + 1].tag_[:2])
                    if tag_pair in [('VB', 'NN'), ('JJ', 'NN'), ('NN', 'NN')]:
                        sportswear_patterns += 1

            features_list.append([noun_after, verb_before, adj_before, action_verb_near, sportswear_patterns])

        return np.array(features_list)

    def fit(self, X: Union[List[str], np.ndarray], y: Optional[np.ndarray] = None):
        """Fit the feature transformer on training data.

        Args:
            X: Training texts
            y: Target labels (not used, for sklearn compatibility)

        Returns:
            self
        """
        self._validate_method()

        # Preprocess texts
        texts = self._preprocess_texts(X)

        if self.method == 'tfidf_word':
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)

        elif self.method == 'tfidf_char':
            self._tfidf_char = self._create_tfidf_char()
            self._tfidf_char.fit(texts)

        elif self.method == 'tfidf_lsa':
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)

            # Fit LSA on TF-IDF features
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)

        elif self.method == 'tfidf_context':
            # Extract context windows around brand mentions
            context_texts = self._extract_all_context_windows(texts)
            # Fit TF-IDF on context windows
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(context_texts)

        elif self.method == 'tfidf_proximity':
            # TF-IDF + proximity features
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
            # Fit scaler on proximity features
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)

        elif self.method == 'tfidf_doc2vec':
            # TF-IDF + Doc2Vec (lexical + semantic)
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
            # Fit Doc2Vec model
            self._fit_doc2vec(texts)
            # Fit scaler on Doc2Vec embeddings to match TF-IDF scale
            doc2vec_features = self._transform_doc2vec(texts)
            self._doc2vec_scaler = StandardScaler()
            self._doc2vec_scaler.fit(doc2vec_features)

        elif self.method == 'tfidf_ner':
            # TF-IDF + NER entity type features
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
            # Fit scaler on NER features
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)

        elif self.method == 'tfidf_pos':
            # TF-IDF + POS pattern features
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
            # Fit scaler on POS features
            pos_features = self._compute_pos_features(texts)
            self._pos_scaler = StandardScaler()
            self._pos_scaler.fit(pos_features)

        elif self.method == 'doc2vec':
            self._fit_doc2vec(texts)

        elif self.method == 'sentence_transformer':
            self._fit_sentence_transformer()

        elif self.method == 'hybrid':
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
            # Fit scaler on vocab features to match TF-IDF scale
            if self.include_vocab_features:
                vocab_features = self._compute_vocab_features(texts)
                self._vocab_scaler = StandardScaler()
                self._vocab_scaler.fit(vocab_features)

        self._is_fitted = True
        return self

    def _fit_doc2vec(self, texts: List[str]) -> None:
        """Fit Doc2Vec model on texts."""
        from gensim.models import Doc2Vec
        from gensim.models.doc2vec import TaggedDocument

        # Create tagged documents
        documents = [
            TaggedDocument(text.split(), [f'doc_{i}'])
            for i, text in enumerate(texts)
        ]

        # Initialize and train model
        self._doc2vec_model = Doc2Vec(
            vector_size=self.doc2vec_vector_size,
            min_count=self.doc2vec_min_count,
            epochs=self.doc2vec_epochs,
            dm=self.doc2vec_dm,
            window=self.doc2vec_window,
            workers=4,
            seed=self.random_state,
        )

        self._doc2vec_model.build_vocab(documents)
        self._doc2vec_model.train(
            documents,
            total_examples=len(documents),
            epochs=self._doc2vec_model.epochs
        )

    def _fit_sentence_transformer(self) -> None:
        """Load sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        self._sentence_model = SentenceTransformer(self.sentence_model_name)

    def transform(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """Transform texts into feature vectors.

        Args:
            X: Texts to transform

        Returns:
            Feature matrix (sparse or dense depending on method)
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted. Call fit() first.")

        # Preprocess texts
        texts = self._preprocess_texts(X)

        if self.method == 'tfidf_word':
            return self._tfidf.transform(texts)

        elif self.method == 'tfidf_char':
            return self._tfidf_char.transform(texts)

        elif self.method == 'tfidf_lsa':
            tfidf_features = self._tfidf.transform(texts)
            return self._lsa.transform(tfidf_features)

        elif self.method == 'tfidf_context':
            # Extract context windows and transform
            context_texts = self._extract_all_context_windows(texts)
            return self._tfidf.transform(context_texts)

        elif self.method == 'tfidf_proximity':
            # TF-IDF + scaled proximity features
            tfidf_features = self._tfidf.transform(texts)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            return sparse.hstack([tfidf_features, proximity_scaled]).tocsr()

        elif self.method == 'tfidf_doc2vec':
            # TF-IDF + scaled Doc2Vec embeddings
            tfidf_features = self._tfidf.transform(texts)
            doc2vec_features = self._transform_doc2vec(texts)
            doc2vec_scaled = self._doc2vec_scaler.transform(doc2vec_features)
            return sparse.hstack([tfidf_features, doc2vec_scaled]).tocsr()

        elif self.method == 'tfidf_ner':
            # TF-IDF + scaled NER entity type features
            tfidf_features = self._tfidf.transform(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            return sparse.hstack([tfidf_features, ner_scaled]).tocsr()

        elif self.method == 'tfidf_pos':
            # TF-IDF + scaled POS pattern features
            tfidf_features = self._tfidf.transform(texts)
            pos_features = self._compute_pos_features(texts)
            pos_scaled = self._pos_scaler.transform(pos_features)
            return sparse.hstack([tfidf_features, pos_scaled]).tocsr()

        elif self.method == 'doc2vec':
            return self._transform_doc2vec(texts)

        elif self.method == 'sentence_transformer':
            return self._transform_sentence_transformer(texts)

        elif self.method == 'hybrid':
            return self._transform_hybrid(texts)

        raise ValueError(f"Unknown method: {self.method}")

    def _transform_doc2vec(self, texts: List[str]) -> np.ndarray:
        """Transform texts using Doc2Vec."""
        embeddings = []
        for text in texts:
            words = text.split()
            vector = self._doc2vec_model.infer_vector(words)
            embeddings.append(vector)
        return np.array(embeddings)

    def _transform_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Transform texts using sentence transformer."""
        return self._sentence_model.encode(texts, show_progress_bar=False)

    def _transform_hybrid(self, texts: List[str]) -> sparse.csr_matrix:
        """Transform texts using TF-IDF + vocabulary features."""
        # TF-IDF features
        tfidf_features = self._tfidf.transform(texts)

        if self.include_vocab_features:
            # Vocabulary features - scale to match TF-IDF
            vocab_features = self._compute_vocab_features(texts)
            vocab_features_scaled = self._vocab_scaler.transform(vocab_features)

            # Combine: sparse TF-IDF + scaled dense vocab
            return sparse.hstack([tfidf_features, vocab_features_scaled]).tocsr()

        return tfidf_features

    def fit_transform(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Training texts
            y: Target labels (not used)

        Returns:
            Feature matrix
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> Optional[np.ndarray]:
        """Get feature names (where available).

        Returns:
            Array of feature names or None if not applicable
        """
        if self.method in ['tfidf_word', 'hybrid', 'tfidf_proximity', 'tfidf_doc2vec',
                          'tfidf_ner', 'tfidf_pos']:
            if self._tfidf is not None:
                names = list(self._tfidf.get_feature_names_out())
                if self.method == 'hybrid' and self.include_vocab_features:
                    # Add vocab feature names
                    vocab_names = [
                        f'vocab_{cat}_{suffix}'
                        for cat in SPORTSWEAR_VOCAB.keys()
                        for suffix in ['count', 'any', 'near_brand']
                    ] + ['vocab_total_matches', 'vocab_near_brand_total']
                    names.extend(vocab_names)
                elif self.method == 'tfidf_proximity':
                    # Add proximity feature names
                    proximity_names = [
                        'prox_min_distance', 'prox_avg_distance',
                        'prox_keyword_count_near', 'prox_has_keyword_near'
                    ]
                    names.extend(proximity_names)
                elif self.method == 'tfidf_doc2vec':
                    # Add Doc2Vec embedding dimension names
                    doc2vec_names = [
                        f'd2v_{i}' for i in range(self.doc2vec_vector_size)
                    ]
                    names.extend(doc2vec_names)
                elif self.method == 'tfidf_ner':
                    # Add NER feature names
                    ner_names = [
                        'ner_fp_entity_count', 'ner_sw_entity_count', 'ner_fp_ratio',
                        'ner_has_animal_near', 'ner_has_location_near', 'ner_has_org_near'
                    ]
                    names.extend(ner_names)
                elif self.method == 'tfidf_pos':
                    # Add POS feature names
                    pos_names = [
                        'pos_noun_after_brand', 'pos_verb_before_brand',
                        'pos_adj_before_brand', 'pos_action_verb_near',
                        'pos_sportswear_pattern_count'
                    ]
                    names.extend(pos_names)
                return np.array(names)

        elif self.method == 'tfidf_char':
            if self._tfidf_char is not None:
                return self._tfidf_char.get_feature_names_out()

        # LSA, Doc2Vec, and SentenceTransformer don't have interpretable names
        return None

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for reproducibility.

        Returns:
            Dictionary with all configuration parameters
        """
        return {
            'method': self.method,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'sublinear_tf': self.sublinear_tf,
            'norm': self.norm,
            'char_ngram_range': self.char_ngram_range,
            'char_max_features': self.char_max_features,
            'lsa_n_components': self.lsa_n_components,
            'context_window_words': self.context_window_words,
            'doc2vec_vector_size': self.doc2vec_vector_size,
            'doc2vec_min_count': self.doc2vec_min_count,
            'doc2vec_epochs': self.doc2vec_epochs,
            'doc2vec_dm': self.doc2vec_dm,
            'doc2vec_window': self.doc2vec_window,
            'sentence_model_name': self.sentence_model_name,
            'include_vocab_features': self.include_vocab_features,
            'vocab_window_size': self.vocab_window_size,
            'proximity_window_size': self.proximity_window_size,
            'random_state': self.random_state,
        }

    def save_config(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        config = self.get_config()

        # Convert tuples to lists for JSON serialization
        config['ngram_range'] = list(config['ngram_range'])
        config['char_ngram_range'] = list(config['char_ngram_range'])

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> 'FPFeatureTransformer':
        """Create transformer from configuration file.

        Args:
            path: Path to configuration JSON

        Returns:
            New FPFeatureTransformer instance (unfitted)
        """
        path = Path(path)
        with open(path) as f:
            config = json.load(f)

        # Convert lists back to tuples
        config['ngram_range'] = tuple(config['ngram_range'])
        config['char_ngram_range'] = tuple(config['char_ngram_range'])

        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"FPFeatureTransformer(method='{self.method}', "
            f"max_features={self.max_features}, fitted={self._is_fitted})"
        )
