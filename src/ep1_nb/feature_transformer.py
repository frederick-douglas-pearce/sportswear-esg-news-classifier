"""Feature transformer for ESG Pre-filter classifier - sklearn-compatible for deployment."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .preprocessing import clean_text


class EPFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible feature transformer for ESG Pre-filter classifier.

    Detects whether an article contains ESG (Environmental, Social, Governance)
    content. Unlike the FP classifier which focuses on brand disambiguation,
    this classifier focuses on detecting ESG-related topics in article content.

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
        'tfidf_lsa_ner',  # TF-IDF LSA + NER entity type features
        'tfidf_lsa_proximity',  # TF-IDF LSA + ESG keyword proximity
        'tfidf_lsa_ner_proximity',  # TF-IDF LSA + NER + proximity features
        'tfidf_lsa_product',  # TF-IDF LSA + product detection features
        'tfidf_context',  # TF-IDF on context window around ESG keywords
        'tfidf_proximity',  # TF-IDF + keyword proximity features
        'tfidf_doc2vec',  # TF-IDF + Doc2Vec embeddings (lexical + semantic)
        'tfidf_ner',  # TF-IDF + NER entity type features
        'doc2vec',
        'sentence_transformer',
        'sentence_transformer_ner',  # Sentence embeddings + NER entity type features
        'sentence_transformer_ner_vocab',  # Sentence + NER + domain vocabulary features
        'sentence_transformer_ner_proximity',  # Sentence + NER + proximity features
        'hybrid',
    ]

    # NER entity types relevant for ESG content detection
    # Organizations, money, and percentages often indicate ESG reporting
    ESG_ENTITY_TYPES = ['ORG', 'MONEY', 'PERCENT', 'DATE', 'CARDINAL']
    # Geographic entities may indicate supply chain or environmental impact
    GEO_ENTITY_TYPES = ['GPE', 'LOC', 'FAC']

    # Environmental keywords
    ENVIRONMENTAL_KEYWORDS = [
        # Climate & emissions
        'carbon', 'emissions', 'greenhouse', 'climate', 'ghg', 'co2',
        'net zero', 'net-zero', 'carbon neutral', 'carbon footprint',
        'scope 1', 'scope 2', 'scope 3', 'decarbonization', 'decarbonize',
        # Energy
        'renewable', 'solar', 'wind', 'clean energy', 'green energy',
        'energy efficiency', 'energy consumption', 'fossil fuel',
        # Waste & materials
        'recycling', 'recycled', 'recyclable', 'waste', 'landfill',
        'circular economy', 'biodegradable', 'compostable', 'plastic',
        'single-use', 'packaging', 'upcycled', 'upcycling',
        # Sustainable materials
        'sustainable', 'sustainability', 'organic', 'eco-friendly',
        'natural materials', 'regenerative', 'certified',
        # Water & pollution
        'water', 'wastewater', 'pollution', 'toxic', 'chemicals',
        'dyeing', 'effluent', 'microplastics', 'microfibers',
        # Biodiversity
        'biodiversity', 'deforestation', 'conservation', 'ecosystem',
        'habitat', 'endangered', 'wildlife', 'forest',
        # Environmental certifications
        'bluesign', 'oeko-tex', 'gots', 'grs', 'higg index',
        'science-based targets', 'sbti', 'cdp', 'leed',
    ]

    # Social keywords
    SOCIAL_KEYWORDS = [
        # Worker rights
        'workers', 'labor', 'labour', 'worker rights', 'labor rights',
        'fair wage', 'living wage', 'minimum wage', 'working conditions',
        'overtime', 'forced labor', 'child labor', 'sweatshop',
        'worker safety', 'occupational safety', 'workplace safety',
        # Supply chain
        'supply chain', 'supplier', 'suppliers', 'factory', 'factories',
        'manufacturing', 'sourcing', 'traceability', 'transparency',
        'tier 1', 'tier 2', 'subcontractor', 'audit', 'auditing',
        # Human rights
        'human rights', 'uyghur', 'xinjiang', 'cotton', 'forced',
        'modern slavery', 'trafficking', 'exploitation',
        # Diversity & inclusion
        'diversity', 'inclusion', 'dei', 'equity', 'equality',
        'gender', 'women', 'minority', 'representation', 'inclusive',
        'lgbtq', 'disability', 'accessibility', 'belonging',
        # Community
        'community', 'communities', 'philanthropy', 'donation',
        'charitable', 'foundation', 'volunteering', 'social impact',
        'giving back', 'non-profit', 'nonprofit',
        # Health & wellbeing
        'health', 'wellbeing', 'wellness', 'mental health', 'safety',
        'injury', 'accident', 'fatality',
        # Social certifications
        'fair trade', 'fairtrade', 'sa8000', 'bsci', 'wrap', 'sedex',
    ]

    # Governance keywords
    GOVERNANCE_KEYWORDS = [
        # Corporate governance
        'governance', 'board', 'director', 'directors', 'chairman',
        'executive', 'leadership', 'oversight', 'fiduciary',
        'shareholder', 'stakeholder', 'accountability',
        # Ethics & compliance
        'ethics', 'ethical', 'compliance', 'code of conduct',
        'anti-corruption', 'bribery', 'whistleblower', 'misconduct',
        'fraud', 'investigation', 'lawsuit', 'settlement',
        # ESG reporting & frameworks
        'esg', 'csr', 'corporate social responsibility',
        'sustainability report', 'annual report', 'disclosure',
        'gri', 'sasb', 'tcfd', 'un global compact', 'sdg',
        'materiality', 'stakeholder engagement',
        # Responsible business
        'responsible', 'responsibility', 'commitment', 'pledge',
        'target', 'goal', 'initiative', 'program', 'strategy',
        # Risk & oversight
        'risk', 'due diligence', 'assessment', 'monitoring',
        'verification', 'certification', 'standard', 'framework',
    ]

    # Digital transformation keywords (ESG-adjacent)
    DIGITAL_KEYWORDS = [
        'digital', 'technology', 'innovation', 'tech', 'ai',
        'artificial intelligence', 'machine learning', 'automation',
        'blockchain', 'traceability', 'data', 'analytics',
        'digital transformation', 'e-commerce', 'online',
    ]

    # Non-ESG keywords - signals article is NOT about ESG topics
    NON_ESG_KEYWORDS = [
        # Product/fashion focus
        'collection', 'fashion week', 'runway', 'style', 'trend',
        'outfit', 'look', 'wear', 'wardrobe', 'season',
        'spring', 'summer', 'fall', 'winter', 'capsule',
        # Sports/performance focus
        'game', 'match', 'tournament', 'championship', 'athlete',
        'performance', 'speed', 'endurance', 'training',
        'marathon', 'race', 'competition', 'win', 'victory',
        # Product reviews
        'review', 'rating', 'best', 'top', 'favorite',
        'recommend', 'buy', 'purchase', 'sale', 'discount',
        'price', 'deal', 'affordable', 'expensive', 'value',
        # Celebrity/endorsement
        'celebrity', 'star', 'influencer', 'ambassador', 'collab',
        'collaboration', 'partnership', 'signed', 'deal',
        # Financial news (non-ESG)
        'stock', 'shares', 'trading', 'earnings', 'revenue',
        'profit', 'quarterly', 'fiscal', 'guidance', 'outlook',
    ]

    # Product sale/release keywords - strong signal of non-ESG content
    PRODUCT_SALE_KEYWORDS = [
        # Sales & deals
        'on sale', 'sale', 'discount', 'off', 'deal', 'deals',
        'black friday', 'cyber monday', 'prime day', 'clearance',
        'promo', 'coupon', 'save', 'savings',
        # Product releases
        'release', 'releases', 'release date', 'release info',
        'launches', 'launch', 'drops', 'dropping', 'just dropped',
        'new release', 'limited edition', 'exclusive',
        # Shopping guidance
        'where to buy', 'how to buy', 'how to get', 'in stock',
        'available', 'sold out', 'restocked', 'shop now',
        # Product reviews
        'review', 'reviews', 'tested', 'rating',
        'best', 'top', 'favorite', 'recommend',
        # Sneaker/shoe specific
        'sneaker', 'sneakers', 'shoe', 'shoes', 'colorway', 'colorways',
        'running shoe', 'walking shoe', 'hiking boot',
        # Retail/pricing
        'retail', 'price', 'msrp', 'cost',
    ]

    # Sponsorship/endorsement keywords - strong signal of non-ESG content
    SPONSORSHIP_KEYWORDS = [
        # Athlete endorsements
        'signature shoe', 'signature line', 'signature collection',
        'endorsement', 'endorsement deal', 'endorses', 'endorsed by',
        'brand ambassador', 'ambassador', 'face of',
        'athlete', 'star athlete', 'sponsored athlete',
        # Sponsorship deals
        'sponsor', 'sponsors', 'sponsorship', 'sponsored',
        'multi-year deal', 'lifetime deal', 'signed with',
        'signs with', 'inks deal', 'partnership deal',
        # Team/uniform sponsorships
        'jersey', 'kit', 'uniform', 'uniforms',
        'team jersey', 'city edition', 'home kit', 'away kit',
        'official partner', 'official sponsor', 'official supplier',
        # Fashion collaborations
        'collab', 'collaboration', 'collaborates', 'x ', ' x ',
        'capsule collection', 'limited collection', 'special edition',
        'designer collaboration', 'fashion collab',
        # Sports events (sponsorship context)
        'world cup', 'olympics', 'nba', 'nfl', 'mlb', 'premier league',
        'champions league', 'euro ', 'copa ',
        # Celebrity/influencer
        'celebrity', 'star', 'influencer', 'celeb',
        'wears', 'wearing', 'spotted in', 'seen in',
    ]

    # Publishers focused on ESG/sustainability content
    ESG_PUBLISHERS = [
        # ESG/sustainability focused
        'sustainablebrands.com',
        'greenbiz.com',
        'esgtoday.com',
        'responsible-investor.com',
        'just-style.com',
        'sourcingjournal.com',
        'ecotextile.com',
        'fashionrevolution.org',
        # Business news with ESG coverage
        'reuters.com',
        'bloomberg.com',
        'ft.com',
        'wsj.com',
        'theguardian.com',
        'bbc.com',
    ]

    # Publishers less likely to cover ESG content
    NON_ESG_PUBLISHERS = [
        # Fashion/style focused
        'vogue.com',
        'elle.com',
        'harpersbazaar.com',
        'instyle.com',
        # Sneaker/streetwear culture
        'hypebeast.com',
        'highsnobiety.com',
        'sneakernews.com',
        'solecollector.com',
        # Sports focused
        'espn.com',
        'si.com',
        'bleacherreport.com',
    ]

    # Categories that may indicate ESG content
    ESG_CATEGORIES = ['business', 'environment', 'politics', 'science']

    # Categories less likely to have ESG content
    NON_ESG_CATEGORIES = ['sports', 'entertainment', 'lifestyle', 'fashion']

    # Source reputation scores (based on observed ESG rate from labeling data)
    # Higher score = more likely to publish ESG content
    SOURCE_REPUTATION_SCORES = {
        # ESG-focused sources (score: 0.8-1.0)
        'sustainablebrands.com': 0.95,
        'greenbiz.com': 0.95,
        'esgtoday.com': 0.95,
        'just-style.com': 0.85,
        'sourcingjournal.com': 0.85,
        'ecotextile.com': 0.90,
        # Business news with ESG coverage (score: 0.5-0.7)
        'reuters.com': 0.65,
        'bloomberg.com': 0.60,
        'ft.com': 0.55,
        'wsj.com': 0.55,
        'theguardian.com': 0.60,
        'bbc.com': 0.50,
        'businesswire.com': 0.45,
        'prnewswire.com': 0.45,
        # Fashion/style sources (lower ESG rate, score: 0.2-0.4)
        'vogue.com': 0.30,
        'elle.com': 0.25,
        'wwd.com': 0.40,
        'fashionista.com': 0.35,
        'footwearnews.com': 0.35,
        # Sneaker/streetwear culture (rarely ESG, score: 0.1-0.2)
        'hypebeast.com': 0.15,
        'highsnobiety.com': 0.15,
        'sneakernews.com': 0.10,
        'solecollector.com': 0.10,
        'nicekicks.com': 0.10,
        # Sports news (rarely ESG, score: 0.1-0.2)
        'espn.com': 0.15,
        'si.com': 0.15,
        'bleacherreport.com': 0.10,
        # Deal/coupon sites (almost never ESG, score: 0.0-0.1)
        'slickdeals.net': 0.05,
        'techradar.com': 0.10,
        'tomsguide.com': 0.10,
    }

    # Headline patterns that suggest non-ESG content (sales, deals, releases)
    HEADLINE_SALE_PATTERNS = [
        r'\bsale\b', r'\bdeal\b', r'\bdeals\b', r'\bdiscount\b',
        r'\b\d+%\s*off\b', r'\bsave\b', r'\bclearance\b',
        r'\bblack friday\b', r'\bcyber monday\b',
        r'\bprime day\b', r'\bbogo\b',
    ]

    HEADLINE_RELEASE_PATTERNS = [
        r'\brelease\b', r'\breleases\b', r'\bdrops\b', r'\bdropping\b',
        r'\blaunch\b', r'\blaunches\b', r'\bcoming\b', r'\bsneak peek\b',
        r'\bfirst look\b', r'\breview\b', r'\bunboxing\b',
    ]

    HEADLINE_ATHLETE_PATTERNS = [
        r'\bsigns\b', r'\bsigned\b', r'\bendorses\b', r'\bwears\b',
        r'\bsponsored\b', r'\bsponsor\b', r'\bambassador\b',
        r'\bworld cup\b', r'\bolympics\b', r'\bnba\b', r'\bnfl\b',
    ]

    # Class-level caches to avoid redundant computation across instances
    _embedding_cache: ClassVar[Dict[str, np.ndarray]] = {}
    _ner_cache: ClassVar[Dict[str, np.ndarray]] = {}
    _shared_sentence_models: ClassVar[Dict[str, Any]] = {}
    _shared_spacy_model: ClassVar[Any] = None

    @classmethod
    def _get_texts_hash(cls, texts: List[str]) -> str:
        """Generate a unique hash from a list of texts."""
        combined = "\x00".join(texts)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    @classmethod
    def _get_embedding_cache_key(cls, model_name: str, texts_hash: str) -> str:
        """Generate cache key for sentence embeddings."""
        return f"emb:{model_name}:{texts_hash}"

    @classmethod
    def _get_ner_cache_key(cls, texts_hash: str, window_size: int) -> str:
        """Generate cache key for NER features."""
        return f"ner:{window_size}:{texts_hash}"

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all class-level caches."""
        cls._embedding_cache.clear()
        cls._ner_cache.clear()
        cls._shared_sentence_models.clear()
        cls._shared_spacy_model = None

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Get statistics about current cache usage."""
        return {
            'embedding_entries': len(cls._embedding_cache),
            'ner_entries': len(cls._ner_cache),
            'loaded_sentence_models': len(cls._shared_sentence_models),
            'spacy_loaded': cls._shared_spacy_model is not None,
        }

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
        # Context window parameters
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
        # Metadata parameters
        include_metadata_in_text: bool = True,
        include_metadata_features: bool = True,
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
            context_window_words: Words before/after keyword for context
            doc2vec_vector_size: Doc2Vec embedding dimension
            doc2vec_min_count: Minimum word count for Doc2Vec
            doc2vec_epochs: Training epochs for Doc2Vec
            doc2vec_dm: Doc2Vec training mode (1=DM, 0=DBOW)
            doc2vec_window: Context window for Doc2Vec
            sentence_model_name: Name of sentence-transformers model
            include_vocab_features: Whether to include domain vocab features
            vocab_window_size: Context window for vocab features
            proximity_window_size: Words to check for keyword proximity
            include_metadata_in_text: Whether to prepend source/category to text
            include_metadata_features: Whether to add discrete metadata features
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
        self.include_metadata_in_text = include_metadata_in_text
        self.include_metadata_features = include_metadata_features
        self.random_state = random_state

        # Fitted components (set during fit)
        self._tfidf = None
        self._tfidf_char = None
        self._lsa = None
        self._doc2vec_model = None
        self._sentence_model = None
        self._spacy_model = None
        self._vocab_scaler = None
        self._proximity_scaler = None
        self._neg_context_scaler = None
        self._doc2vec_scaler = None
        self._ner_scaler = None
        self._product_scaler = None
        self._metadata_scaler = None
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
        """Preprocess texts by applying text cleaning."""
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if hasattr(X, 'tolist'):
            X = X.tolist()
        return [clean_text(str(text)) if text else "" for text in X]

    def _extract_esg_keywords_from_text(self, text: str) -> List[str]:
        """Extract ESG keyword mentions from text."""
        text_lower = text.lower()
        found_keywords = []
        all_esg_keywords = (
            self.ENVIRONMENTAL_KEYWORDS +
            self.SOCIAL_KEYWORDS +
            self.GOVERNANCE_KEYWORDS +
            self.DIGITAL_KEYWORDS
        )
        for keyword in all_esg_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        return found_keywords

    def _extract_context_windows(self, text: str, window_size: int = 20) -> str:
        """Extract text within N words of each ESG keyword mention."""
        keywords = self._extract_esg_keywords_from_text(text)
        if not keywords:
            return ""

        words = text.split()
        if not words:
            return ""

        context_windows = []
        text_lower = text.lower()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            start_pos = 0
            while True:
                pos = text_lower.find(keyword_lower, start_pos)
                if pos == -1:
                    break

                words_before = text[:pos].split()
                word_idx = len(words_before)

                start_idx = max(0, word_idx - window_size)
                end_idx = min(len(words), word_idx + window_size + 1)
                window = words[start_idx:end_idx]
                context_windows.append(' '.join(window))

                start_pos = pos + len(keyword_lower)

        return ' '.join(context_windows)

    def _extract_all_context_windows(self, texts: List[str]) -> List[str]:
        """Extract context windows for all texts."""
        return [
            self._extract_context_windows(text, self.context_window_words)
            for text in texts
        ]

    def _compute_vocab_features(self, texts: List[str]) -> np.ndarray:
        """Compute ESG vocabulary features for all texts.

        For each text, computes:
        - environmental_count: Count of environmental keywords
        - social_count: Count of social keywords
        - governance_count: Count of governance keywords
        - digital_count: Count of digital keywords
        - total_esg_count: Total ESG keywords found
        - has_environmental: Binary flag
        - has_social: Binary flag
        - has_governance: Binary flag
        - non_esg_count: Count of non-ESG keywords
        - esg_ratio: Ratio of ESG to non-ESG keywords
        """
        features_list = []

        for text in texts:
            text_lower = text.lower()

            # Count keywords by category
            env_count = sum(1 for kw in self.ENVIRONMENTAL_KEYWORDS if kw.lower() in text_lower)
            social_count = sum(1 for kw in self.SOCIAL_KEYWORDS if kw.lower() in text_lower)
            gov_count = sum(1 for kw in self.GOVERNANCE_KEYWORDS if kw.lower() in text_lower)
            digital_count = sum(1 for kw in self.DIGITAL_KEYWORDS if kw.lower() in text_lower)
            non_esg_count = sum(1 for kw in self.NON_ESG_KEYWORDS if kw.lower() in text_lower)

            total_esg = env_count + social_count + gov_count + digital_count
            has_env = 1 if env_count > 0 else 0
            has_social = 1 if social_count > 0 else 0
            has_gov = 1 if gov_count > 0 else 0
            has_digital = 1 if digital_count > 0 else 0

            # ESG ratio (avoid division by zero)
            total_all = total_esg + non_esg_count
            esg_ratio = total_esg / total_all if total_all > 0 else 0.5

            features_list.append([
                env_count, social_count, gov_count, digital_count,
                total_esg, has_env, has_social, has_gov, has_digital,
                non_esg_count, esg_ratio
            ])

        return np.array(features_list)

    def _compute_proximity_features(self, texts: List[str]) -> np.ndarray:
        """Compute ESG keyword density and proximity features.

        For each text, computes:
        - keyword_density: ESG keywords per 100 words
        - max_cluster_size: Largest cluster of ESG keywords within window
        - avg_gap: Average word gap between ESG keywords
        - has_multiple_categories: Whether multiple ESG categories present
        """
        features_list = []
        all_esg = set(
            kw.lower() for kw in
            self.ENVIRONMENTAL_KEYWORDS + self.SOCIAL_KEYWORDS +
            self.GOVERNANCE_KEYWORDS + self.DIGITAL_KEYWORDS
        )

        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()

            if not words:
                features_list.append([0.0, 0, 100.0, 0])
                continue

            # Find positions of ESG keywords
            keyword_positions = []
            for i, word in enumerate(words):
                word_clean = word.strip('.,!?;:\'"()[]{}')
                if word_clean in all_esg:
                    keyword_positions.append(i)

            if not keyword_positions:
                features_list.append([0.0, 0, 100.0, 0])
                continue

            # Keyword density (per 100 words)
            density = (len(keyword_positions) / len(words)) * 100

            # Find largest cluster within proximity window
            max_cluster = 1
            for i, pos in enumerate(keyword_positions):
                cluster_size = 1
                for j in range(i + 1, len(keyword_positions)):
                    if keyword_positions[j] - pos <= self.proximity_window_size:
                        cluster_size += 1
                    else:
                        break
                max_cluster = max(max_cluster, cluster_size)

            # Average gap between keywords
            if len(keyword_positions) > 1:
                gaps = [keyword_positions[i+1] - keyword_positions[i]
                        for i in range(len(keyword_positions) - 1)]
                avg_gap = sum(gaps) / len(gaps)
            else:
                avg_gap = len(words)  # Single keyword, gap is text length

            # Check multiple categories
            has_env = any(kw.lower() in text_lower for kw in self.ENVIRONMENTAL_KEYWORDS)
            has_soc = any(kw.lower() in text_lower for kw in self.SOCIAL_KEYWORDS)
            has_gov = any(kw.lower() in text_lower for kw in self.GOVERNANCE_KEYWORDS)
            n_categories = sum([has_env, has_soc, has_gov])
            has_multiple = 1 if n_categories >= 2 else 0

            features_list.append([density, max_cluster, avg_gap, has_multiple])

        return np.array(features_list)

    def _compute_negative_context_features(self, texts: List[str]) -> np.ndarray:
        """Compute non-ESG context features.

        For each text, computes:
        - non_esg_count: Count of non-ESG keywords
        - non_esg_density: Non-ESG keywords per 100 words
        - esg_to_non_ratio: Ratio of ESG to non-ESG keywords
        - is_product_focused: Binary flag if many product/fashion keywords
        """
        features_list = []

        non_esg_set = set(kw.lower() for kw in self.NON_ESG_KEYWORDS)
        esg_set = set(
            kw.lower() for kw in
            self.ENVIRONMENTAL_KEYWORDS + self.SOCIAL_KEYWORDS +
            self.GOVERNANCE_KEYWORDS + self.DIGITAL_KEYWORDS
        )

        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()

            if not words:
                features_list.append([0, 0.0, 0.5, 0])
                continue

            # Count non-ESG keywords
            non_esg_count = sum(1 for word in words if word.strip('.,!?;:\'"()[]{}') in non_esg_set)

            # Count ESG keywords
            esg_count = sum(1 for word in words if word.strip('.,!?;:\'"()[]{}') in esg_set)

            # Densities
            non_esg_density = (non_esg_count / len(words)) * 100

            # Ratio
            total = esg_count + non_esg_count
            ratio = esg_count / total if total > 0 else 0.5

            # Product focused (many fashion/product terms)
            product_keywords = ['collection', 'style', 'fashion', 'trend', 'season', 'review', 'buy', 'sale']
            product_count = sum(1 for kw in product_keywords if kw in text_lower)
            is_product_focused = 1 if product_count >= 3 else 0

            features_list.append([non_esg_count, non_esg_density, ratio, is_product_focused])

        return np.array(features_list)

    def _compute_product_features(self, texts: List[str]) -> np.ndarray:
        """Compute product sale/release and sponsorship detection features.

        These features help distinguish product/sponsorship articles from ESG news.

        For each text, computes:
        - product_keyword_count: Count of PRODUCT_SALE_KEYWORDS found
        - product_keyword_density: Product keywords per 100 words
        - is_sale_article: Binary flag for sale-focused content
        - is_release_article: Binary flag for product release content
        - is_review_article: Binary flag for product review content
        - is_shopping_guide: Binary flag for shopping guidance content
        - product_score: Combined score (higher = more likely product article)
        - esg_product_ratio: Ratio of ESG keywords to product keywords
        - sponsorship_keyword_count: Count of SPONSORSHIP_KEYWORDS found
        - is_sponsorship_article: Binary flag for sponsorship/endorsement content
        - is_athlete_article: Binary flag for athlete-focused content
        - is_collab_article: Binary flag for collaboration/fashion content
        - combined_non_esg_score: Total score combining product + sponsorship signals
        """
        features_list = []

        # Pre-compute keyword sets for efficiency
        sale_keywords = {'sale', 'on sale', 'discount', 'off', 'deal', 'deals',
                        'black friday', 'cyber monday', 'prime day', 'clearance',
                        'promo', 'coupon', 'save', 'savings'}
        release_keywords = {'release', 'releases', 'release date', 'release info',
                           'launches', 'launch', 'drops', 'dropping', 'just dropped',
                           'new release', 'limited edition', 'exclusive'}
        review_keywords = {'review', 'reviews', 'tested', 'rating',
                          'best', 'top', 'favorite', 'recommend'}
        shopping_keywords = {'where to buy', 'how to buy', 'how to get', 'in stock',
                            'available', 'sold out', 'restocked', 'shop now'}
        product_type_keywords = {'sneaker', 'sneakers', 'shoe', 'shoes',
                                'colorway', 'colorways', 'running shoe',
                                'walking shoe', 'hiking boot'}

        # Sponsorship keyword sets
        endorsement_keywords = {'signature shoe', 'signature line', 'signature collection',
                               'endorsement', 'endorsement deal', 'endorses', 'endorsed by',
                               'brand ambassador', 'ambassador', 'face of'}
        sponsorship_keywords = {'sponsor', 'sponsors', 'sponsorship', 'sponsored',
                               'multi-year deal', 'lifetime deal', 'signed with',
                               'signs with', 'inks deal', 'partnership deal',
                               'official partner', 'official sponsor', 'official supplier'}
        athlete_keywords = {'athlete', 'star athlete', 'sponsored athlete',
                           'jersey', 'kit', 'uniform', 'uniforms',
                           'team jersey', 'city edition', 'home kit', 'away kit',
                           'world cup', 'olympics', 'nba', 'nfl', 'mlb',
                           'premier league', 'champions league'}
        collab_keywords = {'collab', 'collaboration', 'collaborates',
                          'capsule collection', 'limited collection', 'special edition',
                          'designer collaboration', 'fashion collab',
                          'celebrity', 'star', 'influencer', 'celeb',
                          'wears', 'wearing', 'spotted in', 'seen in'}

        esg_set = set(
            kw.lower() for kw in
            self.ENVIRONMENTAL_KEYWORDS + self.SOCIAL_KEYWORDS +
            self.GOVERNANCE_KEYWORDS + self.DIGITAL_KEYWORDS
        )

        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()

            if not words:
                features_list.append([0, 0.0, 0, 0, 0, 0, 0.0, 0.5, 0, 0, 0, 0, 0.0])
                continue

            # Count product keywords (check for multi-word phrases first)
            product_count = 0
            for kw in self.PRODUCT_SALE_KEYWORDS:
                if kw.lower() in text_lower:
                    product_count += 1

            # Count sponsorship keywords
            sponsorship_count = 0
            for kw in self.SPONSORSHIP_KEYWORDS:
                if kw.lower() in text_lower:
                    sponsorship_count += 1

            # Count ESG keywords
            esg_count = sum(1 for kw in esg_set if kw in text_lower)

            # Density
            product_density = (product_count / len(words)) * 100

            # Product category flags
            is_sale = 1 if any(kw in text_lower for kw in sale_keywords) else 0
            is_release = 1 if any(kw in text_lower for kw in release_keywords) else 0
            is_review = 1 if any(kw in text_lower for kw in review_keywords) else 0
            is_shopping = 1 if any(kw in text_lower for kw in shopping_keywords) else 0
            has_product_type = 1 if any(kw in text_lower for kw in product_type_keywords) else 0

            # Sponsorship category flags
            is_endorsement = 1 if any(kw in text_lower for kw in endorsement_keywords) else 0
            is_sponsorship = 1 if any(kw in text_lower for kw in sponsorship_keywords) else 0
            is_athlete = 1 if any(kw in text_lower for kw in athlete_keywords) else 0
            is_collab = 1 if any(kw in text_lower for kw in collab_keywords) else 0

            # Combined product score (weighted sum)
            product_score = (
                is_sale * 2 +       # Strong indicator
                is_release * 1.5 +  # Moderate indicator
                is_review * 1.5 +   # Moderate indicator
                is_shopping * 2 +   # Strong indicator
                has_product_type * 1  # Weak indicator
            )

            # Combined sponsorship score
            sponsorship_score = (
                is_endorsement * 2 +   # Strong indicator
                is_sponsorship * 2 +   # Strong indicator
                is_athlete * 1.5 +     # Moderate indicator
                is_collab * 1.5        # Moderate indicator
            )

            # Combined non-ESG score
            combined_non_esg_score = product_score + sponsorship_score

            # ESG to (product + sponsorship) ratio
            total_non_esg = product_count + sponsorship_count
            total = esg_count + total_non_esg
            ratio = esg_count / total if total > 0 else 0.5

            # Aggregate sponsorship flag (any sponsorship-related content)
            is_sponsorship_article = 1 if (is_endorsement or is_sponsorship) else 0

            features_list.append([
                product_count, product_density,
                is_sale, is_release, is_review, is_shopping,
                product_score, ratio,
                sponsorship_count, is_sponsorship_article, is_athlete, is_collab,
                combined_non_esg_score
            ])

        return np.array(features_list)

    def _load_spacy_model(self) -> None:
        """Load spaCy model for NER features."""
        if self._spacy_model is None:
            if EPFeatureTransformer._shared_spacy_model is None:
                import spacy
                EPFeatureTransformer._shared_spacy_model = spacy.load('en_core_web_sm')
            self._spacy_model = EPFeatureTransformer._shared_spacy_model

    def _compute_ner_features(self, texts: List[str]) -> np.ndarray:
        """Compute NER-based features for ESG content detection.

        For each text, computes:
        - org_count: Count of organization entities (companies, NGOs)
        - money_count: Count of monetary values (investments, costs)
        - percent_count: Count of percentages (targets, metrics)
        - date_count: Count of dates (timelines, commitments)
        - geo_count: Count of geographic entities (supply chain locations)
        - has_esg_entities: Binary flag if has ORG + (MONEY or PERCENT)
        """
        texts_hash = self._get_texts_hash(texts)
        cache_key = self._get_ner_cache_key(texts_hash, self.proximity_window_size)

        if cache_key in EPFeatureTransformer._ner_cache:
            return EPFeatureTransformer._ner_cache[cache_key].copy()

        self._load_spacy_model()
        features_list = []

        for text in texts:
            if not text.strip():
                features_list.append([0, 0, 0, 0, 0, 0])
                continue

            doc = self._spacy_model(text[:100000])

            org_count = sum(1 for ent in doc.ents if ent.label_ == 'ORG')
            money_count = sum(1 for ent in doc.ents if ent.label_ == 'MONEY')
            percent_count = sum(1 for ent in doc.ents if ent.label_ == 'PERCENT')
            date_count = sum(1 for ent in doc.ents if ent.label_ == 'DATE')
            geo_count = sum(1 for ent in doc.ents if ent.label_ in ('GPE', 'LOC'))

            # ESG content often has organizations with metrics or money
            has_esg_entities = 1 if (org_count > 0 and (money_count > 0 or percent_count > 0)) else 0

            features_list.append([org_count, money_count, percent_count, date_count, geo_count, has_esg_entities])

        result = np.array(features_list)
        EPFeatureTransformer._ner_cache[cache_key] = result.copy()

        return result

    def _compute_metadata_features(
        self,
        source_names: List[Optional[str]],
        categories: List[Optional[List[str]]],
    ) -> np.ndarray:
        """Compute discrete features from article metadata.

        Features computed:
        - is_esg_publisher: 1 if source is in ESG_PUBLISHERS
        - is_non_esg_publisher: 1 if source is in NON_ESG_PUBLISHERS
        - has_business_category: 1 if 'business' in categories
        - has_environment_category: 1 if 'environment' in categories
        - has_sports_category: 1 if 'sports' in categories
        - has_lifestyle_category: 1 if 'lifestyle' in categories
        - n_esg_categories: count of ESG-related categories
        - n_non_esg_categories: count of non-ESG categories
        - source_reputation_score: numeric ESG likelihood score (0.0-1.0)
        """
        features_list = []

        for source, cats in zip(source_names, categories):
            source_lower = source.lower() if source else ""
            cats_lower = [c.lower() for c in (cats or [])]

            # Publisher features
            is_esg_pub = 1 if any(
                pub in source_lower for pub in self.ESG_PUBLISHERS
            ) else 0
            is_non_esg_pub = 1 if any(
                pub in source_lower for pub in self.NON_ESG_PUBLISHERS
            ) else 0

            # Source reputation score (0.0-1.0, default 0.35 for unknown sources)
            reputation_score = 0.35  # Default for unknown sources
            for domain, score in self.SOURCE_REPUTATION_SCORES.items():
                if domain in source_lower:
                    reputation_score = score
                    break

            # Category features
            has_business = 1 if 'business' in cats_lower else 0
            has_environment = 1 if 'environment' in cats_lower else 0
            has_sports = 1 if 'sports' in cats_lower else 0
            has_lifestyle = 1 if 'lifestyle' in cats_lower else 0

            # Aggregate category counts
            n_esg_cats = sum(1 for c in cats_lower if c in self.ESG_CATEGORIES)
            n_non_esg_cats = sum(1 for c in cats_lower if c in self.NON_ESG_CATEGORIES)

            features_list.append([
                is_esg_pub, is_non_esg_pub,
                has_business, has_environment, has_sports, has_lifestyle,
                n_esg_cats, n_non_esg_cats,
                reputation_score,
            ])

        return np.array(features_list)

    def _compute_headline_features(
        self,
        titles: List[Optional[str]],
    ) -> np.ndarray:
        """Compute features from article headlines/titles.

        Analyzes title patterns to detect non-ESG content indicators:
        - sale/deal patterns (discount, % off, clearance)
        - product release patterns (drops, launch, release)
        - athlete/sponsorship patterns (signs, endorses, sponsored)

        Features computed:
        - has_sale_pattern: 1 if title contains sale/deal patterns
        - has_release_pattern: 1 if title contains release patterns
        - has_athlete_pattern: 1 if title contains athlete/sponsorship patterns
        - n_non_esg_patterns: count of non-ESG patterns found
        - headline_esg_keywords: count of ESG keywords in title
        - headline_non_esg_ratio: ratio of non-ESG to total patterns
        """
        features_list = []

        for title in titles:
            title_lower = title.lower() if title else ""

            # Check for non-ESG patterns
            has_sale = 0
            has_release = 0
            has_athlete = 0

            if title_lower:
                has_sale = 1 if any(
                    re.search(pattern, title_lower)
                    for pattern in self.HEADLINE_SALE_PATTERNS
                ) else 0

                has_release = 1 if any(
                    re.search(pattern, title_lower)
                    for pattern in self.HEADLINE_RELEASE_PATTERNS
                ) else 0

                has_athlete = 1 if any(
                    re.search(pattern, title_lower)
                    for pattern in self.HEADLINE_ATHLETE_PATTERNS
                ) else 0

            # Count ESG keywords in title
            esg_keywords = (
                self.ENVIRONMENTAL_KEYWORDS +
                self.SOCIAL_KEYWORDS +
                self.GOVERNANCE_KEYWORDS
            )
            esg_count = sum(1 for kw in esg_keywords if kw.lower() in title_lower)

            # Total non-ESG patterns
            n_non_esg = has_sale + has_release + has_athlete

            # Calculate ratio
            total = esg_count + n_non_esg
            non_esg_ratio = n_non_esg / total if total > 0 else 0.5

            features_list.append([
                has_sale, has_release, has_athlete,
                n_non_esg, esg_count, non_esg_ratio
            ])

        return np.array(features_list)

    @staticmethod
    def format_metadata_prefix(
        source_name: Optional[str],
        categories: Optional[List[str]],
    ) -> str:
        """Format metadata as a natural text prefix for embedding."""
        parts = []
        if source_name:
            parts.append(source_name)
        if categories:
            parts.extend(categories)
        if parts:
            return " ".join(parts) + " "
        return ""

    def fit(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        source_names: Optional[List[Optional[str]]] = None,
        categories: Optional[List[Optional[List[str]]]] = None,
    ):
        """Fit the feature transformer on training data."""
        self._validate_method()
        texts = self._preprocess_texts(X)

        # Fit metadata scaler if enabled
        if self.include_metadata_features and source_names is not None:
            metadata_features = self._compute_metadata_features(
                source_names, categories or [None] * len(source_names)
            )
            self._metadata_scaler = StandardScaler()
            self._metadata_scaler.fit(metadata_features)

        if self.method == 'tfidf_word':
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)

        elif self.method == 'tfidf_char':
            self._tfidf_char = self._create_tfidf_char()
            self._tfidf_char.fit(texts)

        elif self.method == 'tfidf_lsa':
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)

        elif self.method == 'tfidf_lsa_ner':
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)

        elif self.method == 'tfidf_lsa_proximity':
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            self._neg_context_scaler = StandardScaler()
            self._neg_context_scaler.fit(neg_context_features)

        elif self.method == 'tfidf_lsa_ner_proximity':
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            self._neg_context_scaler = StandardScaler()
            self._neg_context_scaler.fit(neg_context_features)

        elif self.method == 'tfidf_lsa_product':
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)
            product_features = self._compute_product_features(texts)
            self._product_scaler = StandardScaler()
            self._product_scaler.fit(product_features)
            vocab_features = self._compute_vocab_features(texts)
            self._vocab_scaler = StandardScaler()
            self._vocab_scaler.fit(vocab_features)
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            self._neg_context_scaler = StandardScaler()
            self._neg_context_scaler.fit(neg_context_features)

        elif self.method == 'tfidf_context':
            context_texts = self._extract_all_context_windows(texts)
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(context_texts)

        elif self.method == 'tfidf_proximity':
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)

        elif self.method == 'tfidf_doc2vec':
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
            self._fit_doc2vec(texts)
            doc2vec_features = self._transform_doc2vec(texts)
            self._doc2vec_scaler = StandardScaler()
            self._doc2vec_scaler.fit(doc2vec_features)

        elif self.method == 'tfidf_ner':
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)

        elif self.method == 'doc2vec':
            self._fit_doc2vec(texts)

        elif self.method == 'sentence_transformer':
            self._fit_sentence_transformer()

        elif self.method == 'sentence_transformer_ner':
            self._fit_sentence_transformer()
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)

        elif self.method == 'sentence_transformer_ner_vocab':
            self._fit_sentence_transformer()
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            vocab_features = self._compute_vocab_features(texts)
            self._vocab_scaler = StandardScaler()
            self._vocab_scaler.fit(vocab_features)

        elif self.method == 'sentence_transformer_ner_proximity':
            self._fit_sentence_transformer()
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            self._neg_context_scaler = StandardScaler()
            self._neg_context_scaler.fit(neg_context_features)

        elif self.method == 'hybrid':
            self._tfidf = self._create_tfidf_word()
            self._tfidf.fit(texts)
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

        documents = [
            TaggedDocument(text.split(), [f'doc_{i}'])
            for i, text in enumerate(texts)
        ]

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
        model_name = self.sentence_model_name
        if model_name not in EPFeatureTransformer._shared_sentence_models:
            from sentence_transformers import SentenceTransformer
            EPFeatureTransformer._shared_sentence_models[model_name] = SentenceTransformer(model_name)
        self._sentence_model = EPFeatureTransformer._shared_sentence_models[model_name]

    def transform(
        self,
        X: Union[List[str], np.ndarray],
        source_names: Optional[List[Optional[str]]] = None,
        categories: Optional[List[Optional[List[str]]]] = None,
    ) -> np.ndarray:
        """Transform texts into feature vectors."""
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted. Call fit() first.")

        texts = self._preprocess_texts(X)

        # Compute metadata features if enabled
        metadata_scaled = None
        if self.include_metadata_features and self._metadata_scaler is not None:
            if source_names is not None:
                metadata_features = self._compute_metadata_features(
                    source_names, categories or [None] * len(source_names)
                )
            else:
                n_samples = len(texts)
                metadata_features = self._compute_metadata_features(
                    [None] * n_samples, [None] * n_samples
                )
            metadata_scaled = self._metadata_scaler.transform(metadata_features)

        if self.method == 'tfidf_word':
            features = self._tfidf.transform(texts)
            if metadata_scaled is not None:
                return sparse.hstack([features, metadata_scaled]).tocsr()
            return features

        elif self.method == 'tfidf_char':
            features = self._tfidf_char.transform(texts)
            if metadata_scaled is not None:
                return sparse.hstack([features, metadata_scaled]).tocsr()
            return features

        elif self.method == 'tfidf_lsa':
            tfidf_features = self._tfidf.transform(texts)
            features = self._lsa.transform(tfidf_features)
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'tfidf_lsa_ner':
            tfidf_features = self._tfidf.transform(texts)
            lsa_features = self._lsa.transform(tfidf_features)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            features = np.hstack([lsa_features, ner_scaled])
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'tfidf_lsa_proximity':
            tfidf_features = self._tfidf.transform(texts)
            lsa_features = self._lsa.transform(tfidf_features)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            neg_context_scaled = self._neg_context_scaler.transform(neg_context_features)
            features = np.hstack([lsa_features, proximity_scaled, neg_context_scaled])
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'tfidf_lsa_ner_proximity':
            tfidf_features = self._tfidf.transform(texts)
            lsa_features = self._lsa.transform(tfidf_features)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            neg_context_scaled = self._neg_context_scaler.transform(neg_context_features)
            features = np.hstack([lsa_features, ner_scaled, proximity_scaled, neg_context_scaled])
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'tfidf_lsa_product':
            tfidf_features = self._tfidf.transform(texts)
            lsa_features = self._lsa.transform(tfidf_features)
            product_features = self._compute_product_features(texts)
            product_scaled = self._product_scaler.transform(product_features)
            vocab_features = self._compute_vocab_features(texts)
            vocab_scaled = self._vocab_scaler.transform(vocab_features)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            neg_context_scaled = self._neg_context_scaler.transform(neg_context_features)
            features = np.hstack([
                lsa_features, product_scaled, vocab_scaled,
                proximity_scaled, neg_context_scaled
            ])
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'tfidf_context':
            context_texts = self._extract_all_context_windows(texts)
            features = self._tfidf.transform(context_texts)
            if metadata_scaled is not None:
                return sparse.hstack([features, metadata_scaled]).tocsr()
            return features

        elif self.method == 'tfidf_proximity':
            tfidf_features = self._tfidf.transform(texts)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            features = sparse.hstack([tfidf_features, proximity_scaled]).tocsr()
            if metadata_scaled is not None:
                return sparse.hstack([features, metadata_scaled]).tocsr()
            return features

        elif self.method == 'tfidf_doc2vec':
            tfidf_features = self._tfidf.transform(texts)
            doc2vec_features = self._transform_doc2vec(texts)
            doc2vec_scaled = self._doc2vec_scaler.transform(doc2vec_features)
            features = sparse.hstack([tfidf_features, doc2vec_scaled]).tocsr()
            if metadata_scaled is not None:
                return sparse.hstack([features, metadata_scaled]).tocsr()
            return features

        elif self.method == 'tfidf_ner':
            tfidf_features = self._tfidf.transform(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            features = sparse.hstack([tfidf_features, ner_scaled]).tocsr()
            if metadata_scaled is not None:
                return sparse.hstack([features, metadata_scaled]).tocsr()
            return features

        elif self.method == 'doc2vec':
            features = self._transform_doc2vec(texts)
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'sentence_transformer':
            features = self._transform_sentence_transformer(texts)
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'sentence_transformer_ner':
            sentence_features = self._transform_sentence_transformer(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            features = np.hstack([sentence_features, ner_scaled])
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'sentence_transformer_ner_vocab':
            sentence_features = self._transform_sentence_transformer(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            vocab_features = self._compute_vocab_features(texts)
            vocab_scaled = self._vocab_scaler.transform(vocab_features)
            features = np.hstack([sentence_features, ner_scaled, vocab_scaled])
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'sentence_transformer_ner_proximity':
            sentence_features = self._transform_sentence_transformer(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            neg_context_scaled = self._neg_context_scaler.transform(neg_context_features)
            features = np.hstack([sentence_features, ner_scaled, proximity_scaled, neg_context_scaled])
            if metadata_scaled is not None:
                return np.hstack([features, metadata_scaled])
            return features

        elif self.method == 'hybrid':
            features = self._transform_hybrid(texts)
            if metadata_scaled is not None:
                return sparse.hstack([features, metadata_scaled]).tocsr()
            return features

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
        texts_hash = self._get_texts_hash(texts)
        cache_key = self._get_embedding_cache_key(self.sentence_model_name, texts_hash)

        if cache_key in EPFeatureTransformer._embedding_cache:
            return EPFeatureTransformer._embedding_cache[cache_key].copy()

        embeddings = self._sentence_model.encode(texts, show_progress_bar=False)
        EPFeatureTransformer._embedding_cache[cache_key] = embeddings.copy()

        return embeddings

    def _transform_hybrid(self, texts: List[str]) -> sparse.csr_matrix:
        """Transform texts using TF-IDF + vocabulary features."""
        tfidf_features = self._tfidf.transform(texts)

        if self.include_vocab_features:
            vocab_features = self._compute_vocab_features(texts)
            vocab_features_scaled = self._vocab_scaler.transform(vocab_features)
            return sparse.hstack([tfidf_features, vocab_features_scaled]).tocsr()

        return tfidf_features

    def fit_transform(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        source_names: Optional[List[Optional[str]]] = None,
        categories: Optional[List[Optional[List[str]]]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, source_names, categories).transform(X, source_names, categories)

    # Metadata feature names
    METADATA_FEATURE_NAMES = [
        'meta_is_esg_publisher',
        'meta_is_non_esg_publisher',
        'meta_has_business_category',
        'meta_has_environment_category',
        'meta_has_sports_category',
        'meta_has_lifestyle_category',
        'meta_n_esg_categories',
        'meta_n_non_esg_categories',
        'meta_source_reputation_score',
    ]

    def get_feature_names_out(self) -> Optional[np.ndarray]:
        """Get feature names (where available)."""
        names = None

        if self.method in ['tfidf_word', 'hybrid', 'tfidf_proximity', 'tfidf_doc2vec', 'tfidf_ner']:
            if self._tfidf is not None:
                names = list(self._tfidf.get_feature_names_out())
                if self.method == 'hybrid' and self.include_vocab_features:
                    vocab_names = [
                        'vocab_env_count', 'vocab_social_count', 'vocab_gov_count',
                        'vocab_digital_count', 'vocab_total_esg', 'vocab_has_env',
                        'vocab_has_social', 'vocab_has_gov', 'vocab_has_digital',
                        'vocab_non_esg_count', 'vocab_esg_ratio'
                    ]
                    names.extend(vocab_names)
                elif self.method == 'tfidf_proximity':
                    proximity_names = [
                        'prox_keyword_density', 'prox_max_cluster',
                        'prox_avg_gap', 'prox_has_multiple_categories'
                    ]
                    names.extend(proximity_names)
                elif self.method == 'tfidf_doc2vec':
                    doc2vec_names = [f'd2v_{i}' for i in range(self.doc2vec_vector_size)]
                    names.extend(doc2vec_names)
                elif self.method == 'tfidf_ner':
                    ner_names = [
                        'ner_org_count', 'ner_money_count', 'ner_percent_count',
                        'ner_date_count', 'ner_geo_count', 'ner_has_esg_entities'
                    ]
                    names.extend(ner_names)

        elif self.method == 'tfidf_char':
            if self._tfidf_char is not None:
                names = list(self._tfidf_char.get_feature_names_out())

        if names is not None and self._metadata_scaler is not None:
            names.extend(self.METADATA_FEATURE_NAMES)
            return np.array(names)
        elif names is not None:
            return np.array(names)

        return None

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for reproducibility."""
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
            'include_metadata_in_text': self.include_metadata_in_text,
            'include_metadata_features': self.include_metadata_features,
            'random_state': self.random_state,
        }

    def save_config(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        config = self.get_config()
        config['ngram_range'] = list(config['ngram_range'])
        config['char_ngram_range'] = list(config['char_ngram_range'])
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> 'EPFeatureTransformer':
        """Create transformer from configuration file."""
        path = Path(path)
        with open(path) as f:
            config = json.load(f)
        config['ngram_range'] = tuple(config['ngram_range'])
        config['char_ngram_range'] = tuple(config['char_ngram_range'])
        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"EPFeatureTransformer(method='{self.method}', "
            f"max_features={self.max_features}, fitted={self._is_fitted})"
        )
