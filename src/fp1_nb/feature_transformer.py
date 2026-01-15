"""Feature transformer for FP classifier - sklearn-compatible for deployment."""

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
        'tfidf_lsa_ner',  # TF-IDF LSA + NER entity type features
        'tfidf_lsa_proximity',  # TF-IDF LSA + proximity features (positive + negative context)
        'tfidf_lsa_ner_proximity',  # TF-IDF LSA + NER + proximity features
        'tfidf_context',  # TF-IDF on context window around brand mentions
        'tfidf_proximity',  # TF-IDF + keyword proximity features
        'tfidf_doc2vec',  # TF-IDF + Doc2Vec embeddings (lexical + semantic)
        'tfidf_ner',  # TF-IDF + NER entity type features
        'tfidf_pos',  # TF-IDF + POS pattern features
        'doc2vec',
        'sentence_transformer',
        'sentence_transformer_ner',  # Sentence embeddings + NER entity type features
        'sentence_transformer_ner_brands',  # Sentence + NER + brand indicator/summary features
        'doc2vec_ner_brands',  # Doc2Vec + NER + brand indicator features
        'sentence_transformer_ner_vocab',  # Sentence + NER + domain vocabulary features
        'sentence_transformer_ner_proximity',  # Sentence + NER + proximity features (corporate/outdoor vocab)
        'sentence_transformer_ner_fp_indicators',  # Sentence + NER + FP indicator features (stock tickers, company suffixes, etc.)
        'tfidf_lsa_ner_proximity_brands',  # TF-IDF LSA + NER + proximity + brand indicator/summary features
        'hybrid',
    ]

    # NER entity types that suggest false positives vs sportswear
    # Original intuitive categories (best performing in experiments)
    # False positive indicators (locations, facilities, groups, events, persons)
    # PERSON added to catch false positives like "Manor Salomon" (soccer player)
    FP_ENTITY_TYPES = ['GPE', 'LOC', 'FAC', 'NORP', 'EVENT', 'PERSON']
    # Sportswear indicators (organizations, products, financial terms)
    SW_ENTITY_TYPES = ['ORG', 'PRODUCT', 'MONEY', 'PERCENT', 'CARDINAL']

    # Brands with person-name collisions (require PERSON NER detection)
    # These brand names are also common first/last names
    PERSON_NAME_BRANDS = [
        'Salomon',       # Salomon (person's last name, e.g., "Manor Salomon" soccer player)
        'Jordan',        # Jordan (common first name) - note: "Michael Jordan" IS sportswear-related
        'Brooks',        # Brooks (common last name)
    ]

    # Brands with geographic-name collisions (require GPE/LOC NER detection)
    # These brand names are also geographic regions
    GEOGRAPHIC_BRANDS = [
        'Patagonia',            # Patagonia region in South America
        'Columbia Sportswear',  # Columbia River, Columbia University, DC, etc.
    ]

    # Brands that need lowercase detection (lowercase = non-sportswear)
    LOWERCASE_BRANDS = [
        'Vans',  # "vans" lowercase = vehicles (police vans, delivery vans)
    ]

    # Brands with high false positive rates due to name collisions
    # These brands have names that commonly appear in non-sportswear contexts
    PROBLEMATIC_BRANDS = [
        'Vans',       # "vans" = vehicles (police vans, delivery vans, etc.)
        'Anta',       # "Anta" substring of "Santa", "Santana", etc.
        'Puma',       # Puma the animal, Ford Puma car, Puma helicopter
        'Patagonia',  # Patagonia region in South America
        'Columbia Sportswear',  # Columbia University, Columbia Pictures
        'Black Diamond',  # Black Diamond power company, mining company
        'New Balance',  # "new balance" in political/diplomatic contexts
        'Converse',   # "converse" as verb (to talk), Converse TX (city)
        'Timberland', # Timberland as wilderness/forest reference
    ]

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
        # Wearables/tech accessories (smartwatches, fitness trackers, etc.)
        'smartwatch', 'smart watch', 'wearable', 'wearables', 'fitness tracker',
        'smart ring', 'activity tracker', 'health tracking', 'heart rate',
        # Sports/Activity
        'athletic', 'sports', 'running', 'basketball', 'soccer', 'football',
        'tennis', 'golf', 'training', 'workout', 'fitness', 'gym',
        # Brand/Retail
        'brand', 'retailer', 'store', 'stores', 'shop', 'collection',
        'launch', 'release', 'sponsor', 'sponsorship', 'endorsement',
        # Product terms
        'product', 'products', 'line', 'model', 'design', 'edition',
        # Product announcements/launches
        'announces', 'announced', 'unveils', 'unveiled', 'introduces', 'introduced',
        'debuts', 'debuted', 'showcase', 'showcases', 'reveals', 'revealed',
    ]

    # Corporate/business vocabulary - signals legitimate business news about sportswear companies
    CORPORATE_KEYWORDS = [
        # Executive/leadership
        'ceo', 'cfo', 'coo', 'executive', 'chief', 'president', 'chairman',
        'management', 'leadership', 'founder', 'director',
        # Financial
        'revenue', 'earnings', 'profit', 'sales', 'quarterly', 'fiscal',
        'financing', 'loan', 'credit', 'investment', 'investors', 'capital',
        'billion', 'million', 'growth', 'margin', 'guidance', 'forecast',
        # Corporate actions
        'partnership', 'acquisition', 'merger', 'expansion', 'restructuring',
        'logistics', 'supply chain', 'distribution', 'warehouse', 'manufacturing',
        'headquarters', 'subsidiary', 'division',
        # Stock/investor
        'stock', 'shares', 'nasdaq', 'nyse', 'analyst', 'shareholder',
        # Business operations
        'company', 'firm', 'corporation', 'enterprise', 'business',
    ]

    # Outdoor/adventure gear vocabulary - for outdoor apparel brands
    OUTDOOR_KEYWORDS = [
        # Outdoor activities
        'outdoor', 'outdoors', 'hiking', 'climbing', 'mountaineering', 'backpacking',
        'camping', 'trekking', 'trail', 'adventure', 'expedition', 'alpine',
        # Outdoor gear types
        'backpack', 'backpacks', 'duffel', 'luggage', 'tent', 'sleeping bag',
        'daypack', 'rucksack', 'gear', 'equipment', 'kit',
        # Outdoor apparel
        'fleece', 'parka', 'shell', 'hardshell', 'softshell', 'insulation',
        'waterproof', 'breathable', 'windproof', 'thermal', 'base layer',
        'midlayer', 'outerwear', 'rainwear', 'puffer', 'down jacket',
        # Environment/terrain
        'mountain', 'mountains', 'summit', 'peak', 'wilderness', 'backcountry',
        'forest', 'nature', 'terrain',
        # Outdoor brands context
        'technical', 'performance', 'durable', 'lightweight', 'packable',
        # Review/product context
        'review', 'tested', 'testing', 'field test', 'hands-on',
    ]

    # Non-sportswear context vocabulary - signals false positive cases
    # These keywords near a brand mention suggest it's NOT about sportswear
    NON_SPORTSWEAR_KEYWORDS = [
        # Automotive context (Ford Puma, delivery vans, etc.)
        'ford', 'volkswagen', 'renault', 'leapmotor', 'car', 'cars', 'vehicle',
        'vehicles', 'automotive', 'automaker', 'driving', 'driver', 'drivers',
        'suv', 'sedan', 'hatchback', 'crossover', 'ev', 'electric vehicle',
        'mph', 'horsepower', 'engine', 'transmission', 'dealership',
        # Vans as vehicles (not Vans footwear)
        'delivery', 'cargo', 'commercial', 'fleet', 'transport', 'lorry',
        'truck', 'trucking', 'camper', 'motorhome', 'stolen',
        # Geographic regions (Patagonia region, not brand)
        'region', 'glacier', 'glaciers', 'fjord', 'fjords', 'sailing',
        'sailed', 'voyage', 'cruise', 'cruising', 'penguin', 'penguins',
        'guanaco', 'guanacos', 'chile', 'argentina', 'torres del paine',
        # Wildlife/animals (Puma the animal)
        'wild', 'wildlife', 'animal', 'animals', 'species', 'habitat',
        'predator', 'prey', 'hunting', 'hunted', 'mountain lion', 'cougar',
        'cat', 'feline', 'spotted', 'sighting', 'conservation', 'endangered',
        'zoo', 'sanctuary', 'ranger', 'rangers',
        # Power/utility companies (Black Diamond Power)
        'utility', 'utilities', 'power company', 'electric company',
        'psc', 'commission', 'ratepayer', 'ratepayers', 'electricity',
        'kilowatt', 'megawatt', 'grid', 'outage', 'blackout',
        # Investment/finance (Decathlon Capital)
        'capital', 'venture', 'vc', 'private equity', 'funding round',
        'series a', 'series b', 'seed funding', 'dilution', 'governance',
        # Universities/education (Columbia University)
        'university', 'college', 'campus', 'professor', 'faculty',
        'academic', 'research', 'institute', 'student', 'students',
        'undergraduate', 'graduate', 'phd', 'dissertation',
        # Political/diplomatic (New Balance of power)
        'diplomatic', 'diplomacy', 'geopolitical', 'accord', 'treaty',
        'alliance', 'alliances', 'military', 'defense', 'minister',
        'embassy', 'ambassador', 'sanctions',
        # Children's balance bikes (not New Balance)
        'balance bike', 'balance bikes', 'tricycle', 'kindergarten',
        'preschool', 'toddler', 'pedal', 'pedals',
        # TV/entertainment (Top Gear Patagonia special, TV series productions)
        'tv show', 'episode', 'special', 'presenter', 'presenters',
        'clarkson', 'hammond', 'jeremy', 'richard', 'james may',
        # Film/TV production (Patagonia TV series, etc.)
        'drama', 'thriller', 'series', 'film', 'movie', 'cinema',
        'producer', 'co-producer', 'production', 'filming', 'shooting',
        'director', 'screenplay', 'cast', 'starring', 'toplines',
        'netflix', 'hbo', 'amazon prime', 'streaming',
        # NASA/space events
        'nasa', 'space', 'astronaut', 'rocket', 'satellite',
    ]

    # Publishers strongly associated with sportswear content
    SPORTSWEAR_PUBLISHERS = [
        # Fashion/streetwear trade
        'wwd.com',              # Women's Wear Daily - fashion trade
        'highsnobiety.com',     # Streetwear/sneaker culture
        'hypebeast.com',        # Streetwear/fashion
        'sneakernews.com',      # Sneaker news
        'solecollector.com',    # Sneaker culture
        'nicekicks.com',        # Sneaker culture
        'footwearnews.com',     # Footwear industry
        # Outdoor/gear
        'gearjunkie.com',       # Outdoor gear reviews
        'outsideonline.com',    # Outdoor sports
        'backpacker.com',       # Backpacking/hiking gear
        'rei.com',              # REI outdoor retailer
        'switchbacktravel.com', # Outdoor gear reviews
        'outdoorgearlab.com',   # Outdoor gear testing
        'trailrunnermag.com',   # Trail running
        # Running/fitness
        'runningmagazine.ca',   # Running content
        'runnersworld.com',     # Running content
        # Sporting goods business
        'sgbonline.com',        # Sporting goods business
        'sportsonesource.com',  # Sports industry
        # Business/finance (publish sportswear company news)
        'finance.yahoo.com',    # Yahoo Finance
        'reuters.com',          # Reuters business
        'bloomberg.com',        # Bloomberg business
        'marketwatch.com',      # Market news
        'businesswire.com',     # Press releases
        'prnewswire.com',       # Press releases
        'globenewswire.com',    # Press releases
    ]

    # Publishers associated with false positives (wildlife, geography, etc.)
    FP_PUBLISHERS = [
        'nationalgeographic.com',  # Wildlife, geography
        'autoexpress.co.uk',       # Puma helicopters
        'flightglobal.com',        # Aircraft (Puma helicopters)
        'defensenews.com',         # Military (Puma vehicles)
        'janes.com',               # Defense/military
    ]

    # Categories that indicate sportswear content
    SPORTSWEAR_CATEGORIES = ['business', 'sports', 'lifestyle']

    # Categories that may indicate false positives
    FP_CATEGORIES = ['environment', 'science', 'world', 'crime']

    # Stock ticker patterns that indicate financial articles about different companies
    # These patterns help identify articles about companies with similar names
    # e.g., "NASDAQ:ANTA" (Antalpha Platform) vs "ANTA Sports"
    STOCK_TICKER_PATTERNS = [
        r'\bNASDAQ\s*:\s*ANTA\b',  # Antalpha Platform (not Anta Sports)
        r'\bNASDAQ\s*:\s*PBYI\b',  # Puma Biotechnology (not Puma sportswear)
        r'\bTSE\s*:\s*BDI\b',       # Black Diamond Group (mining)
        r'\bNASDAQ\s*:\s*BIRD\b',   # Could be confused but usually is Allbirds
        r'\bASX\s*:\s*PL3\b',       # Patagonia Lithium
        r'\bOTCMKTS\s*:',           # OTC markets often have FP companies
        r'\bOTCMKTS\s*:\s*PMMAF\b',  # Puma Exploration (mining)
    ]

    # Stock-only article patterns - articles purely about stock metrics without brand substance
    # These indicate financial analysis articles that aren't really about the sportswear brand
    STOCK_ONLY_PATTERNS = [
        r'\bshort\s+interest\s+(?:update|down|up|drops?|rises?|decreases?|increases?)\b',
        r'\btrading\s+(?:up|down)\s+\d+%\b',
        r'\bshares?\s+(?:gap\s+)?(?:up|down)\b',
        r'\bstock\s+(?:passes|crosses)\s+(?:above|below)\s+\d+\s*day\s+moving\s+average\b',
        r'\b50[\s-]?day\s+moving\s+average\b',
        r'\b200[\s-]?day\s+moving\s+average\b',
        r'\bcalculating\s+(?:the\s+)?intrinsic\s+value\b',
        r'\bhead[\s-]?to[\s-]?head\s+(?:review|analysis|comparison)\b',
    ]

    # Company suffixes that indicate non-sportswear entities
    # When brand name is followed by these, it's likely a different company
    COMPANY_SUFFIX_PATTERNS = [
        # Investment/finance
        (r'Salomon\s*&\s*Ludwin', 'Salomon'),  # Investment firm
        (r'Salomon\s+LLC', 'Salomon'),
        (r'Salomon\s+Capital', 'Salomon'),
        # Lumber/forestry/investment
        (r'Timberland\s+Lumber', 'Timberland'),
        (r'Timberland\s+Company', 'Timberland'),  # Not the boot brand
        (r'Timberland\s+Corp', 'Timberland'),
        (r'Timberlands?\s+(?:investment|assets?|ownership|portfolio)', 'Timberland'),  # Forestry investment
        (r'(?:own|manage|acquire)\s+timberlands?', 'Timberland'),  # Forestry operations
        (r'Weyerhaeuser.*[Tt]imberland', 'Timberland'),  # Weyerhaeuser forestry company
        # Timberland as place/institution (not boots brand)
        (r'Timberland\s+(?:Regional\s+)?Library', 'Timberland'),  # Timberland Regional Library
        (r'Timberland\s+(?:High\s+)?School', 'Timberland'),  # Schools named Timberland
        # Mining/resources
        (r'Black\s+Diamond\s+Group', 'Black Diamond'),
        (r'Black\s+Diamond\s+Power', 'Black Diamond'),
        (r'Black\s+Diamond\s+Therapeutics', 'Black Diamond'),
        (r'Patagonia\s+Lithium', 'Patagonia'),
        (r'Patagonia\s+Gold', 'Patagonia'),
        (r'Puma\s+Exploration', 'Puma'),
        (r'Puma\s+Biotechnology', 'Puma'),
        # Tech/platform
        (r'Antalpha\s+Platform', 'Anta'),  # NASDAQ:ANTA crypto company
        # Hotels/hospitality (not sportswear)
        (r'ANTA\s+Hotel', 'Anta'),  # ANTA Hotel chain (Radisson)
        (r'Anta\s+Hotel', 'Anta'),
        # Universities/institutions (Columbia Sportswear vs Columbia University)
        (r'Columbia\s+University', 'Columbia'),
        (r'Columbia\s+College', 'Columbia'),
        (r'Columbia\s+Business\s+School', 'Columbia'),
        # Political/diplomatic phrases (New Balance of power, not New Balance shoes)
        (r'[Nn]ew\s+[Bb]alance\s+of\s+(?:power|trade|forces?)', 'New Balance'),
        (r'(?:shift|change|alter)\s+(?:the\s+)?balance', 'New Balance'),
        # Generic suffixes that may indicate different companies
        (r'(\w+)\s+LLC\b', None),  # Matches any brand + LLC
        (r'(\w+)\s+Ltd\.?\b', None),
        (r'(\w+)\s+Inc\.?\b', None),
        (r'(\w+)\s+Corp\.?\b', None),
        (r'(\w+)\s+Capital\b', None),
        (r'(\w+)\s+Partners\b', None),
        (r'(\w+)\s+Holdings\b', None),
        (r'(\w+)\s+Therapeutics\b', None),
        (r'(\w+)\s+Biotechnology\b', None),
        (r'(\w+)\s+Biopharmaceutical\b', None),
        (r'(\w+)\s+Pharmaceuticals?\b', None),
        # Additional Timberland forestry patterns (from FP analysis)
        (r'JPMorgan\s+Campbell\s+Global.*[Tt]imberland', 'Timberland'),
        (r'Campbell\s+Global.*[Tt]imberland', 'Timberland'),
        (r'\d+[,\d]*\s+[Aa]cres\s+of\s+[Tt]imberland', 'Timberland'),
        (r'[Tt]imberland\s+(?:property|acquisition|platform|investing)', 'Timberland'),
    ]

    # Vehicle brand patterns that indicate automotive context
    # These patterns help identify articles about cars/vehicles, not sportswear
    VEHICLE_BRAND_PATTERNS = [
        # Ford Puma (car)
        r'\bFord\s+Puma\b',
        r'\bPuma\s+(?:EV|Gen-E|ST|Titanium|Trend|hybrid)\b',
        r'\bPuma\s+(?:hatchback|crossover|SUV)\b',
        # Other car context near Puma
        r'\b(?:Leapmotor|Renault)\s+\d+.*\bPuma\b',  # e.g., "fight the Renault 4 and Ford Puma"
        # Ford Vans
        r'\bFord\s+(?:Transit|E-Transit)\s+(?:van|vans|Courier)\b',
        # Generic vehicle patterns for Vans
        r'\b(?:delivery|cargo|commercial|police|prison|container|mobile|forensic)\s+vans?\b',
        r'\bvans?\s+(?:and\s+)?(?:trucks?|lorr(?:y|ies))\b',
        r'\b(?:seized|stolen|intercept(?:ed)?)\s+vans?\b',
        # Additional van types (from FP analysis)
        r'\b(?:vanity|transit|camper|sanitation|health|medical|snack|flatbed|council)\s+vans?\b',
        r'\bvans?\s+(?:were|was|being)\s+(?:stolen|broken|intercepted|used)\b',
        r'\bRivian\s+(?:electric\s+)?(?:delivery\s+)?vans?\b',  # Rivian electric delivery vans
        r'\belectric\s+(?:garbage|delivery)\s+(?:vans?|vehicles?)\b',
        # Puma helicopter (military)
        r'\bPuma\s+helicopter\b',
        r'\bSA\s*330\s*Puma\b',  # Puma helicopter model
        # Best Vans articles (automotive reviews)
        r'\bBest\s+(?:Vans|Minivans)\s+(?:for|of)\s+\d{4}\b',
        # Can-Am Spyder three-wheeled motorcycles (not Spyder ski apparel)
        r'\bCan[-\s]?Am\s+Spyder\b',
        r'\bSpyder\s+(?:F3|RT|Ryker)\b',  # Spyder model names
        r'\b(?:three[-\s]?wheel(?:ed)?|trike)\s+(?:motorcycle|vehicle)s?\b.*\bSpyder\b',
        r'\bSpyder\s+(?:motorcycle|roadster|trike)\b',
    ]

    # "On Running" phrase collision patterns
    # The brand "On Running" can be confused with common phrases like "focus on running"
    # These patterns indicate the text is NOT about On Running shoes
    ON_RUNNING_FP_PATTERNS = [
        # Sports context - "on running the ball", "focus on running game"
        r'\bon\s+running\s+the\s+ball\b',
        r'\bfocus(?:ed|ing)?\s+on\s+running\b',
        r'\bkeep\s+(?:a\s+)?(?:strong\s+)?focus\s+on\s+running\b',
        r'\bon\s+running\s+(?:the\s+)?(?:ground\s+)?game\b',
        r'\b(?:ground|rushing)\s+(?:game|attack).*\bon\s+running\b',
        # General verb usage - "keep on running", "go on running"
        r'\b(?:keep|kept|go|went|carry|carried)\s+on\s+running\b',
        # Political/business context - "on running for office", "on running the company"
        r'\bon\s+running\s+for\s+(?:office|president|governor|mayor)\b',
        r'\bon\s+running\s+(?:the\s+)?(?:company|business|organization)\b',
    ]

    # Animal context keywords for Puma (the animal)
    # These keywords near "Puma" suggest it's about the animal, not the brand
    ANIMAL_CONTEXT_KEYWORDS = [
        'mountain lion', 'cougar', 'big cat', 'feline', 'predator', 'prey',
        'wildlife', 'wild', 'spotted', 'sighting', 'tracking', 'hunt',
        'penguin', 'penguins', 'guanaco', 'guanacos',  # Puma prey in Patagonia
        'population', 'populations', 'species', 'habitat', 'conservation',
        'attack', 'attacked', 'kill', 'killed', 'eating', 'feeding',
        'territory', 'territories', 'range', 'cub', 'cubs', 'den',
    ]

    # Geographic context keywords for Patagonia (the region)
    # These keywords near "Patagonia" suggest it's about the region, not the brand
    GEOGRAPHIC_CONTEXT_KEYWORDS = [
        # Geographic features
        'region', 'province', 'territory', 'glacier', 'glaciers', 'fjord', 'fjords',
        'torres del paine', 'tierra del fuego', 'andes', 'steppe',
        # Countries/areas
        'chile', 'chilean', 'argentina', 'argentine', 'south america',
        # Activities in the region (not brand-sponsored)
        'expedition', 'voyage', 'sailing', 'sailed', 'cruise', 'cruising',
        'lodge', 'ranch', 'estancia',
        # Political/administrative
        'mayor', 'governor', 'municipal', 'province',
        # Mining (Patagonia Lithium, etc.)
        'lithium', 'mining', 'exploration', 'drill', 'drilling',
    ]

    # Person name patterns for brands that are also names
    # e.g., "Manor Salomon" (soccer player)
    PERSON_NAME_PATTERNS = [
        (r'\bManor\s+Salomon\b', 'Salomon'),  # Soccer player
        (r'\bSalomon\s+Kalou\b', 'Salomon'),  # Soccer player
        (r'\bSalomon\s+Rondon\b', 'Salomon'),  # Soccer player
        (r'\bMichael\s+(?:Lavie\s+)?Salomon\b', 'Salomon'),  # Business person
        # Dr./Professor prefix patterns (from FP analysis)
        (r'\bDr\.?\s+\w+[-\s]Salomon\b', 'Salomon'),  # Dr. Anne Vincent-Salomon
        (r'\bDr\.?\s+Salomon\b', 'Salomon'),  # Dr. Salomon (any doctor)
        (r'\bProfessor\s+\w*\s*Salomon\b', 'Salomon'),  # Professor Salomon
        (r'\bVincent[-\s]Salomon\b', 'Salomon'),  # Vincent-Salomon (hyphenated name)
    ]

    # Financial jargon patterns indicating stock-only articles
    # These help identify articles that are purely about financial metrics
    FINANCIAL_JARGON_PATTERNS = [
        r'\bshort\s+interest\b',
        r'\bmoving\s+average\b',
        r'\bSEC\s+filing\b',
        r'\bquarterly\s+report\b',
        r'\bearnings\s+call\b',
        r'\binstitutional\s+ownership\b',
        r'\bhedge\s+fund\b',
        r'\bshares\s+(?:traded|outstanding)\b',
        r'\bmarket\s+cap(?:italization)?\b',
        r'\bprice\s+target\b',
        r'\banalyst\s+(?:rating|recommendation)s?\b',
        r'\bGet\s+Free\s+Report\b',  # Common in financial article templates
        # Stock trading article patterns (added based on FP analysis)
        r'\bOTCMKTS\b',  # Over-the-counter stock ticker prefix
        r'\bstock\s+crosses\b',  # "stock crosses above/below"
        r'\btrading\s+(?:down|up|flat)\b',  # "trading down 3%"
        r'\bWhat\'?s\s+Next\??\b',  # Common financial article headline suffix
        r'\b(?:50|100|200)\s+day\s+moving\b',  # Specific moving averages
        # Additional financial patterns (from FP analysis)
        r'\bShould\s+You\s+(?:Buy|Sell)\b',  # Common financial headline suffix
        r'\bStock\s+(?:Passes|Crosses)\s+Above\b',  # Stock price crossing pattern
        r'\bShare\s+Price\s+(?:Passes|Crosses)\b',  # Share price crossing pattern
        r'\bHere\'?s\s+Why\b',  # Common financial headline suffix
        r'\bTSE\s*:\s*\w+\b',  # Toronto Stock Exchange tickers
        r'\bASX\s*:\s*\w+\b',  # Australian Stock Exchange tickers
        r'\bdebt-to-equity\s+ratio\b',
        r'\bquick\s+ratio\b',
        r'\bcurrent\s+ratio\b',
        r'\bFive\s+stocks\s+we\s+like\s+better\b',  # MarketBeat promotional content
        # Additional stock-only article patterns (from FP analysis)
        r'\bHere\'?s\s+What\s+Happened\b',  # Common stock article suffix
        r'\bShares\s+(?:Gap\s+)?(?:Up|Down)\b',  # "Shares Up 1.6%", "Shares Gap Down"
        r'\bStock\s+Price\s+(?:Up|Down)\b',  # "Stock Price Up 1.6%"
        r'\bDrawing\s+Market\s+Parallels\b',  # Financial analysis articles
    ]

    # Institution patterns that indicate non-sportswear entities
    # Universities, resorts, hospitals, housing societies
    INSTITUTION_PATTERNS = [
        # Universities/Colleges
        (r'\bColumbia\s+University\b', 'Columbia'),
        (r'\bColumbia\s+(?:College|Institute|School)\b', 'Columbia'),
        # Resorts/Hotels
        (r'\bTimberland\s+(?:Highlands?\s+)?Resort\b', 'Timberland'),
        (r'\bTimberland\s+(?:Lodge|Hotel|Inn)\b', 'Timberland'),
        # Housing
        (r'\b(?:Godrej\s+)?Prana\s+(?:Society|Housing|Apartments?)\b', 'Prana'),
        # Medical/Health
        (r'\bPrana\s+(?:Hyperbaric|Oxygen|Therapy|Clinic|Hospital)\b', 'Prana'),
        # Sports networks (not the gear brand)
        (r'\bBlack\s+Diamond\s+(?:Sports?\s+)?Network\b', 'Black Diamond'),
        # Everlast Gyms (fitness chain, not Everlast boxing equipment brand)
        (r'\bEverlast\s+Gym(?:s)?\b', 'Everlast'),
        (r'\bEverlast\s+(?:fitness|performance)\s+[Cc]entre?\b', 'Everlast'),
        (r'\bEverlast\s+(?:membership|facility|facilities)\b', 'Everlast'),
        # Entertainment/TV show tangential mentions (from FP analysis)
        # Brand appears in TV show context but article is not about the brand
        (r'\bStranger\s+Things\b.*\b(?:Nike|Adidas|Under\s+Armour|Reebok)\b', None),
        (r'\b(?:Season|Episode)\s+\d+\b.*\b(?:Nike|Adidas|Under\s+Armour|Reebok)\b', None),
        (r'\bcontinuity\s+error\b', None),
        (r'\banachronis(?:m|tic)\b', None),
    ]

    # Phrase patterns where brand names are used as common phrases
    # These patterns detect when the "brand" is actually a phrase
    PHRASE_NOT_BRAND_PATTERNS = [
        # New Balance as phrase
        (r'\bnew\s+balance\s+of\s+(?:power|trade|forces|payments?)\b', 'New Balance'),
        (r'\bstrike\s+a\s+new\s+balance\b', 'New Balance'),
        (r'\bfind(?:ing)?\s+(?:a\s+)?new\s+balance\b', 'New Balance'),
        # On Running as phrase
        (r'\b(?:keep|kept|keeps)\s+on\s+running\b', 'On Running'),
        (r'\bfocus(?:ed|ing)?\s+on\s+running\b', 'On Running'),
        (r'\bon\s+running\s+the\s+(?:ball|offense|game)\b', 'On Running'),
        (r'\bstrong\s+focus\s+on\s+running\b', 'On Running'),
        # On Running in property/geographic names (from FP analysis)
        (r'\bon\s+Running\s+Iron\s+Ranch\b', 'On Running'),
        (r'\bRunning\s+(?:Iron|Creek|River|Springs|Hills?|Ridge|Valley|Water)\s+(?:Ranch|Farm|Estate|Property)\b', 'On Running'),
        (r'\bon\s+Running\s+\w+\s+(?:Ranch|Farm|Estate|Property)\b', 'On Running'),
        # The North Face as mountaineering term (not the apparel brand)
        # "north face" refers to the north-facing side of a mountain
        (r'\b(?:the\s+)?north\s+face\s+of\s+(?:Mount|Mt\.?|the)\s+\w+\b', 'The North Face'),
        (r'\bski(?:ed|ing)?\s+(?:down\s+)?(?:the\s+)?north\s+face\b', 'The North Face'),
        (r'\bclimb(?:ed|ing)?\s+(?:the\s+)?north\s+face\b', 'The North Face'),
        (r'\b(?:Everest|Eiger|K2|Matterhorn)[\'s]*\s+north\s+face\b', 'The North Face'),
    ]

    # Product/Project disambiguation patterns
    # Brands that match food products, mining projects, venues, events, etc.
    PRODUCT_DISAMBIGUATION_PATTERNS = [
        # Food products
        (r'\bBlack\s+Diamond\s+Cheese\b', 'Black Diamond'),
        (r'\bBlack\s+Diamond\s+(?:cheddar|aged|slices?)\b', 'Black Diamond'),
        # Venue/Event names (from FP analysis)
        (r'\bBlack\s+Diamond\s+Ranch\b', 'Black Diamond'),  # Golf course in Florida
        (r'\bBlack\s+Diamond\s+Summit\b', 'Black Diamond'),  # Conference/event name
        (r'\bat\s+Black\s+Diamond\b', 'Black Diamond'),  # "tournament at Black Diamond"
        (r'\bBlack\s+Diamond\s+(?:Golf|Country)\s+Club\b', 'Black Diamond'),
        # Mining/Gold projects
        (r'\bConverse\s+(?:Gold\s+)?Project\b', 'Converse'),
        (r'\bConverse\s+(?:mine|mining|exploration)\b', 'Converse'),
        # Audio/Music equipment
        (r'\bPrana\s+(?:Pedal|Guitar|Audio)\b', 'Prana'),
        (r'\bAum\s+Guitar\s+Prana\b', 'Prana'),
        # Decathlon Capital (investment firm, not retailer)
        (r'\bDecathlon\s+Capital\s+Partners?\b', 'Decathlon'),
        # Computer chips (ASICs = Application-Specific Integrated Circuits)
        (r'\b(?:AI|custom|hardware)\s+ASICs?\b', 'ASICS'),
        (r'\bASICs?\s+(?:chip|processor|accelerator)\b', 'ASICS'),
    ]

    # Sponsored event patterns - indicate article IS about sportswear brand
    # These are NEGATIVE FP indicators (reduce FP score when detected)
    SPONSORED_EVENT_PATTERNS = [
        # Under Armour sponsored events
        (r'\bUnder\s+Armour\s+(?:Next|All[- ]?America)\b', 'Under Armour'),
        (r'\bUA\s+(?:Next|All[- ]?America)\b', 'Under Armour'),
        # Nike sponsored events
        (r'\bNike\s+(?:Bowl|Cup|Classic|Invitational|Championship)\b', 'Nike'),
        (r'\bNike\s+(?:Hoop\s+Summit|Elite)\b', 'Nike'),
        # Adidas sponsored events
        (r'\bAdidas\s+(?:Cup|Classic|Nations|Gauntlet)\b', 'Adidas'),
        # Puma sponsored events
        (r'\bPuma\s+(?:Cup|Classic|Championship)\b', 'Puma'),
        # Generic brand + event patterns
        (r'\b(?:Nike|Adidas|Puma|Under\s+Armour|Reebok)\s+(?:sponsored|presents?|hosts?)\b', None),
        # All-Star/All-America with brand context
        (r'\b(?:All[- ]?Star|All[- ]?America(?:n)?)\s+(?:Game|Match|Classic)\b', None),
    ]

    # Class-level caches to avoid redundant computation across instances
    # These are shared across all FPFeatureTransformer instances
    _embedding_cache: ClassVar[Dict[str, np.ndarray]] = {}
    _ner_cache: ClassVar[Dict[str, np.ndarray]] = {}
    _shared_sentence_models: ClassVar[Dict[str, Any]] = {}
    _shared_spacy_model: ClassVar[Any] = None

    # Precompiled regex patterns (compiled once at class level for performance)
    _compiled_patterns: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def _get_texts_hash(cls, texts: List[str]) -> str:
        """Generate a unique hash from a list of texts.

        Args:
            texts: List of text strings

        Returns:
            SHA256 hash of concatenated texts
        """
        # Use a separator unlikely to appear in texts
        combined = "\x00".join(texts)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    @classmethod
    def _get_embedding_cache_key(cls, model_name: str, texts_hash: str) -> str:
        """Generate cache key for sentence embeddings.

        Args:
            model_name: Name of the sentence transformer model
            texts_hash: Hash of the texts

        Returns:
            Cache key string
        """
        return f"emb:{model_name}:{texts_hash}"

    @classmethod
    def _get_ner_cache_key(cls, texts_hash: str, window_size: int) -> str:
        """Generate cache key for NER features.

        Args:
            texts_hash: Hash of the texts
            window_size: Proximity window size used for NER computation

        Returns:
            Cache key string
        """
        return f"ner:{window_size}:{texts_hash}"

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all class-level caches.

        Call this method to free memory when caches are no longer needed,
        e.g., after completing a feature engineering comparison.
        """
        cls._embedding_cache.clear()
        cls._ner_cache.clear()
        cls._shared_sentence_models.clear()
        cls._shared_spacy_model = None
        cls._compiled_patterns.clear()

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Get statistics about current cache usage.

        Returns:
            Dictionary with cache entry counts
        """
        return {
            'embedding_entries': len(cls._embedding_cache),
            'ner_entries': len(cls._ner_cache),
            'loaded_sentence_models': len(cls._shared_sentence_models),
            'spacy_loaded': cls._shared_spacy_model is not None,
            'patterns_compiled': len(cls._compiled_patterns) > 0,
        }

    @classmethod
    def _ensure_patterns_compiled(cls) -> None:
        """Compile regex patterns once at class level for performance.

        This method compiles all FP indicator regex patterns the first time it's
        called and caches them. Subsequent calls are no-ops. Provides ~15% speedup
        for _compute_fp_indicator_features() by avoiding repeated compilation.
        """
        if cls._compiled_patterns:
            return  # Already compiled

        cls._compiled_patterns = {
            'stock_ticker': [
                re.compile(p, re.IGNORECASE) for p in cls.STOCK_TICKER_PATTERNS
            ],
            'stock_only': [
                re.compile(p, re.IGNORECASE) for p in cls.STOCK_ONLY_PATTERNS
            ],
            'company_suffix': [
                (re.compile(p[0], re.IGNORECASE), p[1])
                for p in cls.COMPANY_SUFFIX_PATTERNS
            ],
            'vehicle': [
                re.compile(p, re.IGNORECASE) for p in cls.VEHICLE_BRAND_PATTERNS
            ],
            'on_running_fp': [
                re.compile(p, re.IGNORECASE) for p in cls.ON_RUNNING_FP_PATTERNS
            ],
            'person_name': [
                (re.compile(p[0], re.IGNORECASE), p[1])
                for p in cls.PERSON_NAME_PATTERNS
            ],
            'financial_jargon': [
                re.compile(p, re.IGNORECASE) for p in cls.FINANCIAL_JARGON_PATTERNS
            ],
            'institution': [
                (re.compile(p[0], re.IGNORECASE), p[1])
                for p in cls.INSTITUTION_PATTERNS
            ],
            'phrase_not_brand': [
                (re.compile(p[0], re.IGNORECASE), p[1])
                for p in cls.PHRASE_NOT_BRAND_PATTERNS
            ],
            'product_disambiguation': [
                (re.compile(p[0], re.IGNORECASE), p[1])
                for p in cls.PRODUCT_DISAMBIGUATION_PATTERNS
            ],
            'sponsored_event': [
                (re.compile(p[0], re.IGNORECASE), p[1])
                for p in cls.SPONSORED_EVENT_PATTERNS
            ],
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
        # Metadata parameters
        include_metadata_in_text: bool = True,
        include_metadata_features: bool = True,
        # Brand features
        include_brand_indicators: bool = False,  # Multi-hot encoding per brand (50 features)
        include_brand_summary: bool = False,  # Aggregate brand stats (3 features)
        # FP indicator features
        include_fp_indicators: bool = True,  # FP detection patterns (8 features)
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
            include_metadata_in_text: Whether to prepend source/category to text
            include_metadata_features: Whether to add discrete metadata features
            include_brand_indicators: Whether to add multi-hot brand encoding (50 features)
            include_brand_summary: Whether to add aggregate brand stats (3 features)
            include_fp_indicators: Whether to add FP indicator features (8 features)
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
        self.include_brand_indicators = include_brand_indicators
        self.include_brand_summary = include_brand_summary
        self.include_fp_indicators = include_fp_indicators
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
        self._neg_context_scaler = None  # For scaling negative context features
        self._doc2vec_scaler = None  # For scaling Doc2Vec in combined mode
        self._ner_scaler = None  # For scaling NER features
        self._pos_scaler = None  # For scaling POS features
        self._metadata_scaler = None  # For scaling metadata features
        self._brand_indicator_scaler = None  # For scaling multi-hot brand indicators
        self._brand_summary_scaler = None  # For scaling brand summary features
        self._brand_ner_scaler = None  # For scaling brand-specific NER features
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
        """Extract potential brand mentions from text using word boundary matching.

        Uses the full BRANDS list from config to ensure all tracked brands
        are detected for feature extraction.

        This prevents false positives like:
        - 'Anta' matching 'Santa' or 'Himanta'
        - 'ASICS' matching 'basic'
        """
        text_lower = text.lower()
        found_brands = []
        for brand in BRANDS:
            pattern = r"\b" + re.escape(brand.lower()) + r"\b"
            if re.search(pattern, text_lower):
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
        - min_distance: Minimum word distance from any brand to any relevant keyword
        - avg_distance: Average distance across all brand mentions
        - keyword_count_near: Count of relevant keywords within proximity window
        - has_keyword_near: Binary flag if any keyword within proximity window

        Uses combined vocabulary from SPORTSWEAR_KEYWORDS, CORPORATE_KEYWORDS,
        and OUTDOOR_KEYWORDS to capture sportswear product mentions, business
        news about sportswear companies, and outdoor gear content.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 4)
        """
        features_list = []
        # Combine all keyword lists for comprehensive coverage
        all_keywords = (
            self.SPORTSWEAR_KEYWORDS +
            self.CORPORATE_KEYWORDS +
            self.OUTDOOR_KEYWORDS
        )
        keywords_set = set(kw.lower() for kw in all_keywords)

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

    def _compute_negative_context_features(self, texts: List[str]) -> np.ndarray:
        """Compute negative context features to identify false positives.

        For each text, computes features based on NON_SPORTSWEAR_KEYWORDS:
        - neg_keyword_count: Count of non-sportswear keywords in full text
        - neg_keyword_near_brand: Count of non-sportswear keywords near brand
        - neg_ratio: Ratio of negative to positive keywords near brand
        - has_neg_context: Binary flag if negative keywords outnumber positive

        These features help identify articles where brand names appear in
        non-sportswear contexts (e.g., "Ford Puma" car, "Patagonia" region).

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 4)
        """
        features_list = []
        neg_keywords_set = set(kw.lower() for kw in self.NON_SPORTSWEAR_KEYWORDS)
        pos_keywords = (
            self.SPORTSWEAR_KEYWORDS +
            self.CORPORATE_KEYWORDS +
            self.OUTDOOR_KEYWORDS
        )
        pos_keywords_set = set(kw.lower() for kw in pos_keywords)

        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            brands = self._extract_brands_from_text(text)

            if not brands or not words:
                features_list.append([0, 0, 0.5, 0])
                continue

            # Count negative keywords in full text
            neg_count_full = sum(1 for word in words if word.strip('.,!?;:\'"()[]{}') in neg_keywords_set)

            # Find brand positions
            brand_word_indices = []
            for brand in brands:
                brand_lower = brand.lower()
                start_pos = 0
                while True:
                    pos = text_lower.find(brand_lower, start_pos)
                    if pos == -1:
                        break
                    word_idx = len(text_lower[:pos].split())
                    brand_word_indices.append(word_idx)
                    start_pos = pos + len(brand_lower)

            if not brand_word_indices:
                features_list.append([neg_count_full, 0, 0.5, 0])
                continue

            # Count keywords near brand mentions
            neg_near_brand = 0
            pos_near_brand = 0
            for brand_idx in brand_word_indices:
                window_start = max(0, brand_idx - self.proximity_window_size)
                window_end = min(len(words), brand_idx + self.proximity_window_size + 1)
                window_words = words[window_start:window_end]

                for word in window_words:
                    word_clean = word.strip('.,!?;:\'"()[]{}')
                    if word_clean in neg_keywords_set:
                        neg_near_brand += 1
                    if word_clean in pos_keywords_set:
                        pos_near_brand += 1

            # Compute ratio (avoid division by zero)
            total_near = neg_near_brand + pos_near_brand
            neg_ratio = neg_near_brand / total_near if total_near > 0 else 0.5

            # Binary flag: more negative than positive context
            has_neg_context = 1 if neg_near_brand > pos_near_brand else 0

            features_list.append([neg_count_full, neg_near_brand, neg_ratio, has_neg_context])

        return np.array(features_list)

    def _compute_brand_indicators(self, texts: List[str]) -> np.ndarray:
        """Compute multi-hot brand indicator features for all texts.

        Creates a binary indicator for each brand in the BRANDS list,
        allowing the model to learn brand-specific patterns.

        Features computed (per sample):
        - One indicator per brand in BRANDS list (50 binary features)

        This is particularly useful for tree-based models that can learn
        to apply different rules for problematic brands (Vans, Anta, Puma, etc.)
        that have high false positive rates due to name collisions.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, len(BRANDS))
        """
        features_list = []

        for text in texts:
            # Extract brands mentioned in this text
            mentioned_brands = set(self._extract_brands_from_text(text))

            # Create multi-hot encoding for each brand
            brand_indicators = [
                1 if brand in mentioned_brands else 0
                for brand in BRANDS
            ]

            features_list.append(brand_indicators)

        return np.array(features_list)

    def _compute_brand_summary(self, texts: List[str]) -> np.ndarray:
        """Compute aggregate brand summary features for all texts.

        Creates summary statistics about brand mentions:
        - n_brands: Count of brands mentioned
        - has_problematic_brand: Binary flag if any PROBLEMATIC_BRANDS mentioned
        - n_problematic_brands: Count of problematic brands mentioned

        This is a compact representation (3 features) that captures the
        key brand-related signals without adding 50 dimensions.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 3)
        """
        features_list = []

        for text in texts:
            # Extract brands mentioned in this text
            mentioned_brands = set(self._extract_brands_from_text(text))

            # Aggregate features
            n_brands = len(mentioned_brands)
            problematic_mentioned = mentioned_brands.intersection(set(self.PROBLEMATIC_BRANDS))
            has_problematic = 1 if problematic_mentioned else 0
            n_problematic = len(problematic_mentioned)

            features_list.append([n_brands, has_problematic, n_problematic])

        return np.array(features_list)

    def _compute_fp_indicator_features(self, texts: List[str]) -> np.ndarray:
        """Compute false positive indicator features for all texts.

        Detects patterns that strongly indicate false positives:
        - Stock tickers for different companies (NASDAQ:ANTA, TSE:BDI, etc.)
        - Company suffixes indicating different entities (LLC, Ltd, etc.)
        - Vehicle brand patterns (Ford Puma, delivery vans, etc.)
        - Animal context keywords near Puma
        - Geographic context keywords near Patagonia
        - Person name patterns (Manor Salomon, etc.)
        - Financial jargon (short interest, SEC filing, etc.)
        - Institution patterns (University, Resort, Hospital, etc.)
        - Phrase patterns where brand is used as phrase (new balance of power)
        - Product disambiguation (Black Diamond Cheese, Converse Gold Project)

        Also detects patterns that indicate TRUE sportswear content:
        - Sponsored events (Under Armour All-America, Nike Bowl, etc.)

        Features computed (per sample):
        - has_stock_ticker: Binary flag if stock ticker pattern found
        - has_company_suffix: Binary flag if company suffix pattern found
        - has_vehicle_pattern: Binary flag if vehicle pattern found
        - has_animal_context: Binary flag if animal keywords near Puma
        - has_geographic_context: Binary flag if geographic keywords near Patagonia
        - has_person_name: Binary flag if person name pattern found
        - has_financial_jargon: Binary flag if stock-only article jargon found
        - has_institution: Binary flag if institution pattern found
        - has_phrase_not_brand: Binary flag if brand used as common phrase
        - has_product_disambiguation: Binary flag if brand matches other product
        - has_sponsored_event: Binary flag if brand-sponsored event detected (NEGATIVE FP indicator)
        - fp_indicator_count: Total count of FP indicators (excludes sponsored_event)
        - fp_indicator_score: Weighted score of FP indicators (sponsored_event subtracts)

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 15)
        """
        features_list = []

        # Use precompiled regex patterns (compiled once at class level)
        self._ensure_patterns_compiled()
        stock_ticker_patterns = self._compiled_patterns['stock_ticker']
        stock_only_patterns = self._compiled_patterns['stock_only']
        company_suffix_patterns = self._compiled_patterns['company_suffix']
        vehicle_patterns = self._compiled_patterns['vehicle']
        on_running_fp_patterns = self._compiled_patterns['on_running_fp']
        person_patterns = self._compiled_patterns['person_name']
        financial_jargon_patterns = self._compiled_patterns['financial_jargon']
        institution_patterns = self._compiled_patterns['institution']
        phrase_patterns = self._compiled_patterns['phrase_not_brand']
        product_patterns = self._compiled_patterns['product_disambiguation']
        sponsored_event_patterns = self._compiled_patterns['sponsored_event']

        animal_keywords_set = set(kw.lower() for kw in self.ANIMAL_CONTEXT_KEYWORDS)
        geographic_keywords_set = set(kw.lower() for kw in self.GEOGRAPHIC_CONTEXT_KEYWORDS)

        for text in texts:
            text_lower = text.lower()
            brands = self._extract_brands_from_text(text)

            # Check stock ticker patterns
            has_stock_ticker = 0
            for pattern in stock_ticker_patterns:
                if pattern.search(text):
                    has_stock_ticker = 1
                    break

            # Check stock-only article patterns (pure financial metrics articles)
            has_stock_only = 0
            stock_only_match_count = 0
            for pattern in stock_only_patterns:
                if pattern.search(text):
                    stock_only_match_count += 1
                    # Require at least 2 matches for confidence
                    if stock_only_match_count >= 2:
                        has_stock_only = 1
                        break

            # Check company suffix patterns
            has_company_suffix = 0
            for pattern, brand in company_suffix_patterns:
                if pattern.search(text):
                    # If brand is specified, check if that brand is mentioned
                    if brand is None or brand in brands:
                        has_company_suffix = 1
                        break

            # Check vehicle patterns
            has_vehicle_pattern = 0
            for pattern in vehicle_patterns:
                if pattern.search(text):
                    has_vehicle_pattern = 1
                    break

            # Check "On Running" false positive patterns (only if On Running brand mentioned)
            has_on_running_fp = 0
            if 'On Running' in brands or 'On' in brands:
                for pattern in on_running_fp_patterns:
                    if pattern.search(text):
                        has_on_running_fp = 1
                        break

            # Check animal context (only if Puma is mentioned)
            has_animal_context = 0
            if 'Puma' in brands:
                words = text_lower.split()
                for word in words:
                    word_clean = word.strip('.,!?;:\'"()[]{}')
                    if word_clean in animal_keywords_set:
                        has_animal_context = 1
                        break
                # Also check multi-word phrases
                for phrase in ['mountain lion', 'big cat']:
                    if phrase in text_lower:
                        has_animal_context = 1
                        break

            # Check geographic context (only if Patagonia is mentioned)
            has_geographic_context = 0
            if 'Patagonia' in brands:
                words = text_lower.split()
                for word in words:
                    word_clean = word.strip('.,!?;:\'"()[]{}')
                    if word_clean in geographic_keywords_set:
                        has_geographic_context = 1
                        break
                # Also check multi-word phrases
                for phrase in ['torres del paine', 'tierra del fuego', 'south america']:
                    if phrase in text_lower:
                        has_geographic_context = 1
                        break

            # Check person name patterns
            has_person_name = 0
            for pattern, brand in person_patterns:
                if pattern.search(text):
                    if brand in brands:
                        has_person_name = 1
                        break

            # Check financial jargon (stock-only article indicators)
            has_financial_jargon = 0
            financial_match_count = 0
            for pattern in financial_jargon_patterns:
                if pattern.search(text):
                    financial_match_count += 1
                    # Require at least 2 matches for confidence
                    if financial_match_count >= 2:
                        has_financial_jargon = 1
                        break

            # Check institution patterns
            has_institution = 0
            for pattern, brand in institution_patterns:
                if pattern.search(text):
                    if brand in brands:
                        has_institution = 1
                        break

            # Check phrase-not-brand patterns
            has_phrase_not_brand = 0
            for pattern, brand in phrase_patterns:
                if pattern.search(text):
                    if brand in brands:
                        has_phrase_not_brand = 1
                        break

            # Check product disambiguation patterns
            has_product_disambiguation = 0
            for pattern, brand in product_patterns:
                if pattern.search(text):
                    if brand in brands:
                        has_product_disambiguation = 1
                        break

            # Check sponsored event patterns (NEGATIVE FP indicator - suggests sportswear)
            has_sponsored_event = 0
            for pattern, brand in sponsored_event_patterns:
                if pattern.search(text):
                    # If brand is specified, check if that brand is mentioned
                    # If brand is None, it's a generic pattern that applies to any sportswear brand
                    if brand is None or brand in brands:
                        has_sponsored_event = 1
                        break

            # Aggregate features (sponsored_event is NOT counted as FP indicator)
            fp_indicator_count = (
                has_stock_ticker + has_stock_only + has_company_suffix + has_vehicle_pattern +
                has_on_running_fp + has_animal_context + has_geographic_context + has_person_name +
                has_financial_jargon + has_institution + has_phrase_not_brand +
                has_product_disambiguation
            )

            # Weighted score (stock tickers and company suffixes are strongest signals)
            # sponsored_event SUBTRACTS from score (indicates sportswear, not FP)
            fp_indicator_score = (
                has_stock_ticker * 2.0 +
                has_stock_only * 2.0 +  # Stock-only articles are strong FP signals
                has_company_suffix * 2.0 +
                has_vehicle_pattern * 1.5 +
                has_on_running_fp * 2.0 +  # On Running phrase collision is strong signal
                has_animal_context * 1.0 +
                has_geographic_context * 1.0 +
                has_person_name * 1.5 +
                has_financial_jargon * 1.5 +
                has_institution * 2.0 +
                has_phrase_not_brand * 2.0 +
                has_product_disambiguation * 2.0 -
                has_sponsored_event * 2.0  # Subtract: indicates true sportswear
            )

            features_list.append([
                has_stock_ticker,
                has_stock_only,
                has_company_suffix,
                has_vehicle_pattern,
                has_on_running_fp,
                has_animal_context,
                has_geographic_context,
                has_person_name,
                has_financial_jargon,
                has_institution,
                has_phrase_not_brand,
                has_product_disambiguation,
                has_sponsored_event,
                fp_indicator_count,
                fp_indicator_score,
            ])

        return np.array(features_list)

    def _load_spacy_model(self) -> None:
        """Load spaCy model for NER/POS features.

        Uses a class-level shared model to avoid loading the model multiple
        times when comparing different feature engineering methods.
        """
        if self._spacy_model is None:
            # Check class-level shared model first
            if FPFeatureTransformer._shared_spacy_model is None:
                import spacy
                FPFeatureTransformer._shared_spacy_model = spacy.load('en_core_web_sm')
            self._spacy_model = FPFeatureTransformer._shared_spacy_model

    def _compute_ner_features(self, texts: List[str]) -> np.ndarray:
        """Compute NER-based features for all texts.

        For each text, computes entity type counts near brand mentions:
        - fp_entity_count: Count of false-positive-indicating entities near brand
        - sw_entity_count: Count of sportswear-indicating entities near brand
        - fp_entity_ratio: Ratio of FP entities to total entities
        - has_animal_near: Binary flag if ANIMAL entity near brand
        - has_location_near: Binary flag if GPE/LOC entity near brand
        - has_org_near: Binary flag if ORG entity near brand

        Uses class-level caching to avoid recomputing NER features for the
        same texts when comparing different feature engineering methods.

        Uses spaCy nlp.pipe() for batch processing to improve performance.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 6)
        """
        # Check cache first
        texts_hash = self._get_texts_hash(texts)
        cache_key = self._get_ner_cache_key(texts_hash, self.proximity_window_size)

        if cache_key in FPFeatureTransformer._ner_cache:
            # Return a copy to prevent accidental modification
            return FPFeatureTransformer._ner_cache[cache_key].copy()

        self._load_spacy_model()

        # Pre-extract brands for all texts
        all_brands = [self._extract_brands_from_text(text) for text in texts]

        # Truncate texts for spaCy processing (limit to 100k chars)
        truncated_texts = [text[:100000] for text in texts]

        # Batch process all texts with nlp.pipe() for ~30% speedup
        # n_process=1 for thread safety, batch_size=50 for memory efficiency
        docs = list(self._spacy_model.pipe(truncated_texts, batch_size=50))

        features_list = []
        window_chars = self.proximity_window_size * 6  # Approx chars per word

        for text, brands, doc in zip(texts, all_brands, docs):
            if not brands or not text.strip():
                features_list.append([0, 0, 0.5, 0, 0, 0])
                continue

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

        # Cache the result
        result = np.array(features_list)
        FPFeatureTransformer._ner_cache[cache_key] = result.copy()

        return result

    def _compute_brand_specific_ner_features(self, texts: List[str]) -> np.ndarray:
        """Compute brand-specific NER features to catch name collisions.

        This method detects specific false positive patterns for problematic brands:
        - PERSON entities near person-name brands (Salomon, Jordan, Brooks)
        - GPE/LOC entities near geographic brands (Patagonia, Columbia)
        - Lowercase occurrences of brands that are also common words (Vans)

        Features computed (8 features):
        - person_near_person_brand: 1 if PERSON entity near a person-name brand mention
        - location_near_geo_brand: 1 if GPE/LOC entity near a geographic brand mention
        - has_lowercase_vans: 1 if "vans" appears lowercase (likely vehicles, not footwear)
        - person_brand_mentioned: 1 if any person-name brand is mentioned
        - geo_brand_mentioned: 1 if any geographic brand is mentioned
        - lowercase_brand_mentioned: 1 if any lowercase-prone brand is mentioned
        - n_person_ner_near_brand: count of PERSON entities near brand mentions
        - brand_is_part_of_person_name: 1 if brand appears to be part of person's full name

        Uses spaCy nlp.pipe() for batch processing to improve performance.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 8)
        """
        self._load_spacy_model()
        features_list = []

        # Normalize brand lists for matching
        person_brands_lower = [b.lower() for b in self.PERSON_NAME_BRANDS]
        geo_brands_lower = [b.lower() for b in self.GEOGRAPHIC_BRANDS]
        # For geographic brands, also check partial match (e.g., "Columbia" matches "Columbia Sportswear")
        geo_brand_stems = ['patagonia', 'columbia']
        lowercase_brands_lower = [b.lower() for b in self.LOWERCASE_BRANDS]

        # Pre-extract brands for all texts
        all_brands = [self._extract_brands_from_text(text) for text in texts]

        # Truncate texts for spaCy processing (limit to 100k chars)
        truncated_texts = [text[:100000] for text in texts]

        # Batch process all texts with nlp.pipe() for improved performance
        docs = list(self._spacy_model.pipe(truncated_texts, batch_size=50))

        for text, brands, doc in zip(texts, all_brands, docs):
            text_lower = text.lower()

            # Initialize features with defaults
            person_near_person_brand = 0
            location_near_geo_brand = 0
            has_lowercase_vans = 0
            person_brand_mentioned = 0
            geo_brand_mentioned = 0
            lowercase_brand_mentioned = 0
            n_person_ner_near_brand = 0
            brand_is_part_of_person_name = 0

            if not brands or not text.strip():
                features_list.append([
                    person_near_person_brand, location_near_geo_brand, has_lowercase_vans,
                    person_brand_mentioned, geo_brand_mentioned, lowercase_brand_mentioned,
                    n_person_ner_near_brand, brand_is_part_of_person_name
                ])
                continue

            # Check which brand types are mentioned
            brands_lower = [b.lower() for b in brands]
            person_brand_mentioned = 1 if any(b in person_brands_lower for b in brands_lower) else 0
            geo_brand_mentioned = 1 if any(
                any(stem in b for stem in geo_brand_stems) for b in brands_lower
            ) else 0
            lowercase_brand_mentioned = 1 if any(b in lowercase_brands_lower for b in brands_lower) else 0

            # Check for lowercase "vans" (vehicles, not Vans footwear)
            # Look for "vans" that's NOT at start of sentence and NOT all caps
            if 'vans' in brands_lower:
                # Find all occurrences of "vans" in original text
                vans_pattern = r'(?<![A-Z])\bvans\b(?![A-Z])'  # lowercase "vans"
                if re.search(vans_pattern, text):
                    has_lowercase_vans = 1

            # Find brand character positions for proximity check
            window_chars = self.proximity_window_size * 6  # Approx chars per word
            brand_positions = {}  # brand_lower -> list of (start, end) positions

            for brand in brands:
                brand_lower = brand.lower()
                brand_positions[brand_lower] = []
                start_pos = 0
                while True:
                    pos = text_lower.find(brand_lower, start_pos)
                    if pos == -1:
                        break
                    brand_positions[brand_lower].append((pos, pos + len(brand_lower)))
                    start_pos = pos + len(brand_lower)

            # Check NER entities
            for ent in doc.ents:
                ent_start = ent.start_char
                ent_end = ent.end_char

                # Count PERSON entities near any brand mention
                if ent.label_ == 'PERSON':
                    for brand_lower, positions in brand_positions.items():
                        for brand_start, brand_end in positions:
                            if (abs(ent_start - brand_end) <= window_chars or
                                abs(ent_end - brand_start) <= window_chars):
                                n_person_ner_near_brand += 1

                                # Check if brand is part of the PERSON entity
                                # (e.g., "Manor Salomon" where "Salomon" is both brand and person's name)
                                ent_text_lower = ent.text.lower()
                                if brand_lower in ent_text_lower:
                                    brand_is_part_of_person_name = 1

                                # Check if this is a person-name brand
                                if brand_lower in person_brands_lower:
                                    person_near_person_brand = 1
                                break

                # Check GPE/LOC entities near geographic brands
                elif ent.label_ in ('GPE', 'LOC'):
                    for brand_lower, positions in brand_positions.items():
                        # Check if brand is a geographic brand
                        is_geo_brand = any(stem in brand_lower for stem in geo_brand_stems)
                        if is_geo_brand:
                            for brand_start, brand_end in positions:
                                if (abs(ent_start - brand_end) <= window_chars or
                                    abs(ent_end - brand_start) <= window_chars):
                                    location_near_geo_brand = 1

                                    # Check if brand is part of the location entity
                                    ent_text_lower = ent.text.lower()
                                    if any(stem in ent_text_lower for stem in geo_brand_stems):
                                        # The geographic entity itself contains the brand name
                                        # This is a strong signal of false positive
                                        location_near_geo_brand = 1
                                    break

            features_list.append([
                person_near_person_brand, location_near_geo_brand, has_lowercase_vans,
                person_brand_mentioned, geo_brand_mentioned, lowercase_brand_mentioned,
                n_person_ner_near_brand, brand_is_part_of_person_name
            ])

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

    def _compute_metadata_features(
        self,
        source_names: List[Optional[str]],
        categories: List[Optional[List[str]]],
    ) -> np.ndarray:
        """Compute discrete features from article metadata.

        Features computed:
        - is_sportswear_publisher: 1 if source is in SPORTSWEAR_PUBLISHERS
        - is_fp_publisher: 1 if source is in FP_PUBLISHERS
        - has_business_category: 1 if 'business' in categories
        - has_sports_category: 1 if 'sports' in categories
        - has_environment_category: 1 if 'environment' in categories
        - has_science_category: 1 if 'science' in categories
        - n_sportswear_categories: count of sportswear-related categories
        - n_fp_categories: count of FP-related categories

        Args:
            source_names: List of publisher names (can contain None)
            categories: List of category lists (can contain None)

        Returns:
            Feature matrix of shape (n_samples, 8)
        """
        features_list = []

        for source, cats in zip(source_names, categories):
            source_lower = source.lower() if source else ""
            cats_lower = [c.lower() for c in (cats or [])]

            # Publisher features
            is_sportswear_pub = 1 if any(
                pub in source_lower for pub in self.SPORTSWEAR_PUBLISHERS
            ) else 0
            is_fp_pub = 1 if any(
                pub in source_lower for pub in self.FP_PUBLISHERS
            ) else 0

            # Category features
            has_business = 1 if 'business' in cats_lower else 0
            has_sports = 1 if 'sports' in cats_lower else 0
            has_environment = 1 if 'environment' in cats_lower else 0
            has_science = 1 if 'science' in cats_lower else 0

            # Aggregate category counts
            n_sw_cats = sum(1 for c in cats_lower if c in self.SPORTSWEAR_CATEGORIES)
            n_fp_cats = sum(1 for c in cats_lower if c in self.FP_CATEGORIES)

            features_list.append([
                is_sportswear_pub,
                is_fp_pub,
                has_business,
                has_sports,
                has_environment,
                has_science,
                n_sw_cats,
                n_fp_cats,
            ])

        return np.array(features_list)

    @staticmethod
    def format_metadata_prefix(
        source_name: Optional[str],
        categories: Optional[List[str]],
    ) -> str:
        """Format metadata as a natural text prefix for embedding.

        Uses plain text without brackets/special formatting to work well
        with both TF-IDF (after punctuation removal) and sentence transformers.
        The domain name and categories are included as natural words that
        can be learned semantically.

        Args:
            source_name: Publisher name (e.g., "wwd.com")
            categories: List of categories (e.g., ["business", "sports"])

        Returns:
            Natural text prefix, e.g., "wwd.com business sports "
        """
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
        """Fit the feature transformer on training data.

        Args:
            X: Training texts
            y: Target labels (not used, for sklearn compatibility)
            source_names: Optional list of publisher names for discrete metadata features
            categories: Optional list of category lists for discrete metadata features

        Returns:
            self
        """
        self._validate_method()

        # For *_brands methods, enable brand features before any processing
        if self.method.endswith('_brands'):
            self.include_brand_indicators = True
            self.include_brand_summary = False  # Disabled: summary features don't add value

        # Preprocess texts
        texts = self._preprocess_texts(X)

        # Fit metadata scaler if metadata features enabled and metadata provided
        if self.include_metadata_features and source_names is not None:
            metadata_features = self._compute_metadata_features(source_names, categories or [None] * len(source_names))
            self._metadata_scaler = StandardScaler()
            self._metadata_scaler.fit(metadata_features)

        # Fit brand indicator scaler if enabled
        if self.include_brand_indicators:
            brand_indicators = self._compute_brand_indicators(texts)
            self._brand_indicator_scaler = StandardScaler()
            self._brand_indicator_scaler.fit(brand_indicators)

        # Fit brand summary scaler if enabled
        if self.include_brand_summary:
            brand_summary = self._compute_brand_summary(texts)
            self._brand_summary_scaler = StandardScaler()
            self._brand_summary_scaler.fit(brand_summary)

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

        elif self.method == 'tfidf_lsa_ner':
            # TF-IDF LSA + NER entity type features
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)

            # Fit LSA on TF-IDF features
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)

            # Fit scaler on NER features
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)

            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)

        elif self.method == 'tfidf_lsa_proximity':
            # TF-IDF LSA + proximity features (positive + negative context)
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)

            # Fit LSA on TF-IDF features
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)

            # Fit scaler on proximity features (uses combined vocabulary)
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)

            # Fit scaler on negative context features
            neg_context_features = self._compute_negative_context_features(texts)
            self._neg_context_scaler = StandardScaler()
            self._neg_context_scaler.fit(neg_context_features)

        elif self.method == 'tfidf_lsa_ner_proximity':
            # TF-IDF LSA + NER + proximity features + FP indicator features
            self._tfidf = self._create_tfidf_word()
            tfidf_features = self._tfidf.fit_transform(texts)

            # Fit LSA on TF-IDF features
            self._lsa = TruncatedSVD(
                n_components=self.lsa_n_components,
                random_state=self.random_state
            )
            self._lsa.fit(tfidf_features)

            # Fit scaler on NER features
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)

            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)

            # Fit scaler on proximity features (uses combined vocabulary)
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)

            # Fit scaler on negative context features
            neg_context_features = self._compute_negative_context_features(texts)
            self._neg_context_scaler = StandardScaler()
            self._neg_context_scaler.fit(neg_context_features)

            # Fit scaler on FP indicator features (8-dim) if enabled
            include_fp = getattr(self, 'include_fp_indicators', True)
            if include_fp:
                fp_indicator_features = self._compute_fp_indicator_features(texts)
                self._fp_indicator_scaler = StandardScaler()
                self._fp_indicator_scaler.fit(fp_indicator_features)

        elif self.method == 'tfidf_lsa_ner_proximity_brands':
            # TF-IDF LSA + NER + proximity + brand features (both indicators and summary)
            # + FP indicator features (stock tickers, company suffixes, vehicle patterns, etc.)
            # Brand scalers are fitted at start of fit() due to endswith('_brands') check
            # Same as tfidf_lsa_ner_proximity
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

            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)

            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)

            neg_context_features = self._compute_negative_context_features(texts)
            self._neg_context_scaler = StandardScaler()
            self._neg_context_scaler.fit(neg_context_features)

            # Fit scaler on FP indicator features (8-dim)
            fp_indicator_features = self._compute_fp_indicator_features(texts)
            self._fp_indicator_scaler = StandardScaler()
            self._fp_indicator_scaler.fit(fp_indicator_features)

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
            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)

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

        elif self.method == 'sentence_transformer_ner':
            # Sentence embeddings + NER entity type features
            self._fit_sentence_transformer()
            # Fit scaler on NER features
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)

        elif self.method == 'sentence_transformer_ner_brands':
            # Sentence embeddings + NER + brand features (both indicators and summary)
            # Brand scalers are fitted at start of fit() due to endswith('_brands') check
            self._fit_sentence_transformer()
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)

        elif self.method == 'doc2vec_ner_brands':
            # Doc2Vec embeddings + NER + brand features
            # Brand scalers are fitted at start of fit() due to endswith('_brands') check
            self._fit_doc2vec(texts)
            # Fit scaler for doc2vec embeddings
            doc2vec_features = self._transform_doc2vec(texts)
            self._doc2vec_scaler = StandardScaler()
            self._doc2vec_scaler.fit(doc2vec_features)
            # Fit scaler on NER features
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)

        elif self.method == 'sentence_transformer_ner_vocab':
            # Sentence embeddings + NER + domain vocabulary features
            self._fit_sentence_transformer()
            # Fit scaler on NER features
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)
            # Fit scaler on vocab features
            vocab_features = self._compute_vocab_features(texts)
            self._vocab_scaler = StandardScaler()
            self._vocab_scaler.fit(vocab_features)

        elif self.method == 'sentence_transformer_ner_proximity':
            # Sentence embeddings + NER + proximity features + negative context features
            self._fit_sentence_transformer()
            # Fit scaler on NER features
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)
            # Fit scaler on proximity features (uses combined vocabulary)
            proximity_features = self._compute_proximity_features(texts)
            self._proximity_scaler = StandardScaler()
            self._proximity_scaler.fit(proximity_features)
            # Fit scaler on negative context features
            neg_context_features = self._compute_negative_context_features(texts)
            self._neg_context_scaler = StandardScaler()
            self._neg_context_scaler.fit(neg_context_features)

        elif self.method == 'sentence_transformer_ner_fp_indicators':
            # Sentence embeddings + NER + FP indicator features (stock tickers, company suffixes, etc.)
            self._fit_sentence_transformer()
            # Fit scaler on NER features
            ner_features = self._compute_ner_features(texts)
            self._ner_scaler = StandardScaler()
            self._ner_scaler.fit(ner_features)
            # Fit scaler on brand-specific NER features
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            self._brand_ner_scaler = StandardScaler()
            self._brand_ner_scaler.fit(brand_ner_features)
            # Fit scaler on FP indicator features
            fp_indicator_features = self._compute_fp_indicator_features(texts)
            self._fp_indicator_scaler = StandardScaler()
            self._fp_indicator_scaler.fit(fp_indicator_features)

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
        """Load sentence transformer model.

        Uses class-level shared models to avoid loading the same model
        multiple times when comparing different feature engineering methods.
        """
        model_name = self.sentence_model_name

        # Check class-level shared models first
        if model_name not in FPFeatureTransformer._shared_sentence_models:
            from sentence_transformers import SentenceTransformer
            FPFeatureTransformer._shared_sentence_models[model_name] = SentenceTransformer(model_name)

        self._sentence_model = FPFeatureTransformer._shared_sentence_models[model_name]

    def transform(
        self,
        X: Union[List[str], np.ndarray],
        source_names: Optional[List[Optional[str]]] = None,
        categories: Optional[List[Optional[List[str]]]] = None,
    ) -> np.ndarray:
        """Transform texts into feature vectors.

        Args:
            X: Texts to transform
            source_names: Optional list of publisher names for discrete metadata features
            categories: Optional list of category lists for discrete metadata features

        Returns:
            Feature matrix (sparse or dense depending on method)
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted. Call fit() first.")

        # For *_brands methods, ensure brand features are enabled
        # (needed in case estimator was cloned by sklearn)
        if self.method.endswith('_brands'):
            self.include_brand_indicators = True
            self.include_brand_summary = False  # Disabled: summary features don't add value

        # Preprocess texts
        texts = self._preprocess_texts(X)

        # Compute metadata features if enabled and scaler was fitted
        # If metadata not provided but scaler exists, use default (empty) values
        metadata_scaled = None
        if self.include_metadata_features and self._metadata_scaler is not None:
            if source_names is not None:
                metadata_features = self._compute_metadata_features(source_names, categories or [None] * len(source_names))
            else:
                # Use default empty metadata when not provided (for deployment/inference)
                n_samples = len(texts)
                metadata_features = self._compute_metadata_features(
                    [None] * n_samples,
                    [None] * n_samples
                )
            metadata_scaled = self._metadata_scaler.transform(metadata_features)

        # Compute brand indicator features if enabled and scaler was fitted
        brand_indicators_scaled = None
        if self.include_brand_indicators and self._brand_indicator_scaler is not None:
            brand_indicators = self._compute_brand_indicators(texts)
            brand_indicators_scaled = self._brand_indicator_scaler.transform(brand_indicators)

        # Compute brand summary features if enabled and scaler was fitted
        brand_summary_scaled = None
        if self.include_brand_summary and self._brand_summary_scaler is not None:
            brand_summary = self._compute_brand_summary(texts)
            brand_summary_scaled = self._brand_summary_scaler.transform(brand_summary)

        # Helper to stack optional features (metadata + brand indicators + brand summary)
        def _stack_optional_features(features, is_sparse=False):
            """Stack metadata and brand features onto main features."""
            optional_features = []
            if metadata_scaled is not None:
                optional_features.append(metadata_scaled)
            if brand_indicators_scaled is not None:
                optional_features.append(brand_indicators_scaled)
            if brand_summary_scaled is not None:
                optional_features.append(brand_summary_scaled)

            if not optional_features:
                return features

            if is_sparse:
                return sparse.hstack([features] + optional_features).tocsr()
            else:
                return np.hstack([features] + optional_features)

        if self.method == 'tfidf_word':
            features = self._tfidf.transform(texts)
            return _stack_optional_features(features, is_sparse=True)

        elif self.method == 'tfidf_char':
            features = self._tfidf_char.transform(texts)
            return _stack_optional_features(features, is_sparse=True)

        elif self.method == 'tfidf_lsa':
            tfidf_features = self._tfidf.transform(texts)
            features = self._lsa.transform(tfidf_features)
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'tfidf_lsa_ner':
            # TF-IDF LSA + NER entity type features + brand-specific NER features
            # Note: _compute_ner_features uses class-level caching
            tfidf_features = self._tfidf.transform(texts)
            lsa_features = self._lsa.transform(tfidf_features)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            features = np.hstack([lsa_features, ner_scaled, brand_ner_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'tfidf_lsa_proximity':
            # TF-IDF LSA + proximity features (positive + negative context)
            tfidf_features = self._tfidf.transform(texts)
            lsa_features = self._lsa.transform(tfidf_features)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            neg_context_scaled = self._neg_context_scaler.transform(neg_context_features)
            features = np.hstack([lsa_features, proximity_scaled, neg_context_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'tfidf_lsa_ner_proximity':
            # TF-IDF LSA + NER + proximity features + brand-specific NER + optional FP indicators
            tfidf_features = self._tfidf.transform(texts)
            lsa_features = self._lsa.transform(tfidf_features)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            neg_context_scaled = self._neg_context_scaler.transform(neg_context_features)

            feature_arrays = [lsa_features, ner_scaled, brand_ner_scaled, proximity_scaled, neg_context_scaled]

            # Add FP indicator features if enabled (with backwards compatibility)
            include_fp = getattr(self, 'include_fp_indicators', True)
            if include_fp and self._fp_indicator_scaler is not None:
                fp_indicator_features = self._compute_fp_indicator_features(texts)
                fp_indicator_scaled = self._fp_indicator_scaler.transform(fp_indicator_features)
                feature_arrays.append(fp_indicator_scaled)

            features = np.hstack(feature_arrays)
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'tfidf_lsa_ner_proximity_brands':
            # TF-IDF LSA + NER + proximity + brand features (53-dim) + brand-specific NER + FP indicators
            # Brand features are added by _stack_optional_features since include_brand_* flags are True
            tfidf_features = self._tfidf.transform(texts)
            lsa_features = self._lsa.transform(tfidf_features)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            neg_context_scaled = self._neg_context_scaler.transform(neg_context_features)
            fp_indicator_features = self._compute_fp_indicator_features(texts)
            fp_indicator_scaled = self._fp_indicator_scaler.transform(fp_indicator_features)
            features = np.hstack([lsa_features, ner_scaled, brand_ner_scaled, proximity_scaled, neg_context_scaled, fp_indicator_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'tfidf_context':
            # Extract context windows and transform
            context_texts = self._extract_all_context_windows(texts)
            features = self._tfidf.transform(context_texts)
            return _stack_optional_features(features, is_sparse=True)

        elif self.method == 'tfidf_proximity':
            # TF-IDF + scaled proximity features
            tfidf_features = self._tfidf.transform(texts)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            features = sparse.hstack([tfidf_features, proximity_scaled]).tocsr()
            return _stack_optional_features(features, is_sparse=True)

        elif self.method == 'tfidf_doc2vec':
            # TF-IDF + scaled Doc2Vec embeddings
            tfidf_features = self._tfidf.transform(texts)
            doc2vec_features = self._transform_doc2vec(texts)
            doc2vec_scaled = self._doc2vec_scaler.transform(doc2vec_features)
            features = sparse.hstack([tfidf_features, doc2vec_scaled]).tocsr()
            return _stack_optional_features(features, is_sparse=True)

        elif self.method == 'tfidf_ner':
            # TF-IDF + scaled NER entity type features + brand-specific NER
            tfidf_features = self._tfidf.transform(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            features = sparse.hstack([tfidf_features, ner_scaled, brand_ner_scaled]).tocsr()
            return _stack_optional_features(features, is_sparse=True)

        elif self.method == 'tfidf_pos':
            # TF-IDF + scaled POS pattern features
            tfidf_features = self._tfidf.transform(texts)
            pos_features = self._compute_pos_features(texts)
            pos_scaled = self._pos_scaler.transform(pos_features)
            features = sparse.hstack([tfidf_features, pos_scaled]).tocsr()
            return _stack_optional_features(features, is_sparse=True)

        elif self.method == 'doc2vec':
            features = self._transform_doc2vec(texts)
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'sentence_transformer':
            features = self._transform_sentence_transformer(texts)
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'sentence_transformer_ner':
            # Sentence embeddings (384-dim) + scaled NER features (6-dim) + brand-specific NER (8-dim)
            sentence_features = self._transform_sentence_transformer(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            features = np.hstack([sentence_features, ner_scaled, brand_ner_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'sentence_transformer_ner_brands':
            # Sentence embeddings (384-dim) + scaled NER features (6-dim) + brand-specific NER (8-dim) + brand features (53-dim)
            # Brand features are added by _stack_optional_features since include_brand_* flags are True
            sentence_features = self._transform_sentence_transformer(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            features = np.hstack([sentence_features, ner_scaled, brand_ner_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'doc2vec_ner_brands':
            # Doc2Vec embeddings (vector_size-dim) + scaled NER features (6-dim) + brand-specific NER (8-dim) + brand features (50-dim)
            # Brand features are added by _stack_optional_features since include_brand_* flags are True
            doc2vec_features = self._transform_doc2vec(texts)
            doc2vec_scaled = self._doc2vec_scaler.transform(doc2vec_features)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            features = np.hstack([doc2vec_scaled, ner_scaled, brand_ner_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'sentence_transformer_ner_vocab':
            # Sentence embeddings (384-dim) + scaled NER (6-dim) + brand-specific NER (8-dim) + scaled vocab features
            sentence_features = self._transform_sentence_transformer(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            vocab_features = self._compute_vocab_features(texts)
            vocab_scaled = self._vocab_scaler.transform(vocab_features)
            features = np.hstack([sentence_features, ner_scaled, brand_ner_scaled, vocab_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'sentence_transformer_ner_proximity':
            # Sentence embeddings (384-dim) + scaled NER (6-dim) + brand-specific NER (8-dim) + scaled proximity (4-dim) + neg context (4-dim)
            sentence_features = self._transform_sentence_transformer(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            proximity_features = self._compute_proximity_features(texts)
            proximity_scaled = self._proximity_scaler.transform(proximity_features)
            neg_context_features = self._compute_negative_context_features(texts)
            neg_context_scaled = self._neg_context_scaler.transform(neg_context_features)
            features = np.hstack([sentence_features, ner_scaled, brand_ner_scaled, proximity_scaled, neg_context_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'sentence_transformer_ner_fp_indicators':
            # Sentence embeddings (384-dim) + scaled NER (6-dim) + brand-specific NER (8-dim) + FP indicators (8-dim)
            sentence_features = self._transform_sentence_transformer(texts)
            ner_features = self._compute_ner_features(texts)
            ner_scaled = self._ner_scaler.transform(ner_features)
            brand_ner_features = self._compute_brand_specific_ner_features(texts)
            brand_ner_scaled = self._brand_ner_scaler.transform(brand_ner_features)
            fp_indicator_features = self._compute_fp_indicator_features(texts)
            fp_indicator_scaled = self._fp_indicator_scaler.transform(fp_indicator_features)
            features = np.hstack([sentence_features, ner_scaled, brand_ner_scaled, fp_indicator_scaled])
            return _stack_optional_features(features, is_sparse=False)

        elif self.method == 'hybrid':
            features = self._transform_hybrid(texts)
            return _stack_optional_features(features, is_sparse=True)

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
        """Transform texts using sentence transformer.

        Uses class-level caching to avoid recomputing embeddings for the
        same texts when comparing different feature engineering methods.

        Args:
            texts: List of text strings to encode

        Returns:
            Numpy array of embeddings (n_samples, embedding_dim)
        """
        # Check cache first
        texts_hash = self._get_texts_hash(texts)
        cache_key = self._get_embedding_cache_key(self.sentence_model_name, texts_hash)

        if cache_key in FPFeatureTransformer._embedding_cache:
            # Return a copy to prevent accidental modification
            return FPFeatureTransformer._embedding_cache[cache_key].copy()

        # Compute embeddings
        embeddings = self._sentence_model.encode(texts, show_progress_bar=False)

        # Cache the result
        FPFeatureTransformer._embedding_cache[cache_key] = embeddings.copy()

        return embeddings

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
        y: Optional[np.ndarray] = None,
        source_names: Optional[List[Optional[str]]] = None,
        categories: Optional[List[Optional[List[str]]]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Training texts
            y: Target labels (not used)
            source_names: Optional list of publisher names for discrete metadata features
            categories: Optional list of category lists for discrete metadata features

        Returns:
            Feature matrix
        """
        return self.fit(X, y, source_names, categories).transform(X, source_names, categories)

    # Metadata feature names (8 features from _compute_metadata_features)
    METADATA_FEATURE_NAMES = [
        'meta_is_sportswear_publisher',
        'meta_is_fp_publisher',
        'meta_has_business_category',
        'meta_has_sports_category',
        'meta_has_environment_category',
        'meta_has_science_category',
        'meta_n_sportswear_categories',
        'meta_n_fp_categories',
    ]

    def get_feature_names_out(self) -> Optional[np.ndarray]:
        """Get feature names (where available).

        Returns:
            Array of feature names or None if not applicable
        """
        names = None

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

        elif self.method == 'tfidf_char':
            if self._tfidf_char is not None:
                names = list(self._tfidf_char.get_feature_names_out())

        # Add metadata feature names if metadata scaler is fitted
        if names is not None and self._metadata_scaler is not None:
            names.extend(self.METADATA_FEATURE_NAMES)
            return np.array(names)
        elif names is not None:
            return np.array(names)

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
            'include_metadata_in_text': self.include_metadata_in_text,
            'include_metadata_features': self.include_metadata_features,
            'include_brand_indicators': self.include_brand_indicators,
            'include_brand_summary': self.include_brand_summary,
            'include_fp_indicators': getattr(self, 'include_fp_indicators', True),
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
