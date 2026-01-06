# Project Structure

This document provides a complete file listing for the project.

> **Quick Start:** For a high-level overview, see the [main README](../README.md#project-structure).

```
sportswear-esg-news-classifier/
├── docker-compose.yml          # PostgreSQL + FP Classifier API containers
├── Dockerfile                  # Multi-stage build for classifier APIs
├── pyproject.toml              # Project dependencies and metadata (uv/pip)
├── .env.example                # Environment variable template
├── .env                        # Local environment variables (not committed)
├── logs/                       # Application logs
├── data/                       # Training data exports (JSONL)
│   ├── fp_training_data.jsonl        # FP classifier training data
│   └── ep_training_data.jsonl        # EP classifier training data
├── migrations/                 # Database migration scripts
│   └── 002_classifier_predictions.sql  # Classifier predictions table
├── scripts/
│   ├── collect_news.py         # CLI script for data collection
│   ├── label_articles.py       # CLI script for LLM-based labeling
│   ├── export_training_data.py # Export labeled data for ML training
│   ├── export_website_feed.py  # Export JSON/Atom feeds for website
│   ├── gdelt_backfill.py       # Historical backfill script (3 months)
│   ├── cleanup_non_english.py  # Remove non-English articles from database
│   ├── cleanup_false_positives.py # Identify/remove false positive brand matches
│   ├── train.py                # Unified training script for FP/EP classifiers
│   ├── predict.py              # Unified FastAPI service for all classifiers
│   ├── retrain.py              # Retrain models with version management
│   ├── register_model.py       # Register models in MLflow without retraining
│   ├── monitor_drift.py        # Monitor prediction drift for deployed models
│   ├── backup_db.sh            # Database backup script with rotation
│   ├── deploy_cloudrun.sh      # Google Cloud Run deployment script
│   ├── cron_collect.sh         # Cron wrapper for NewsData.io collection
│   ├── cron_scrape.sh          # Cron wrapper for GDELT collection
│   ├── cron_monitor.sh         # Cron wrapper for drift monitoring
│   └── setup_cron.sh           # User-friendly cron management
├── notebooks/
│   ├── fp1_EDA_FE.ipynb              # FP: EDA & Feature Engineering
│   ├── fp2_model_selection_tuning.ipynb  # FP: Model selection & tuning
│   ├── fp3_model_evaluation_deployment.ipynb  # FP: Test evaluation & deployment
│   ├── ep1_EDA_FE.ipynb              # EP: EDA & Feature Engineering
│   ├── ep2_model_selection_tuning.ipynb  # EP: Model selection & tuning
│   └── ep3_model_evaluation_deployment.ipynb  # EP: Test evaluation & deployment
├── models/                     # Saved ML models and artifacts
│   ├── registry.json                 # Model version registry
│   ├── fp_classifier_pipeline.joblib # FP production model
│   ├── fp_classifier_config.json     # FP model configuration
│   ├── ep_classifier_pipeline.joblib # EP production model
│   └── ep_classifier_config.json     # EP model configuration
├── docs/                       # Detailed documentation
│   ├── COLLECTION.md           # News collection pipeline details
│   ├── LABELING.md             # LLM labeling pipeline details
│   ├── DATABASE.md             # Schema, queries, backup procedures
│   ├── MLOPS.md                # MLflow, Evidently, monitoring
│   └── PROJECT_STRUCTURE.md    # This file
├── src/
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── config.py           # Settings, brands, keywords, API configuration
│   │   ├── api_client.py       # NewsData.io API wrapper with query generation
│   │   ├── gdelt_client.py     # GDELT DOC 2.0 API wrapper (free, 3 months history)
│   │   ├── scraper.py          # Full article text extraction with language detection
│   │   ├── database.py         # PostgreSQL operations with SQLAlchemy
│   │   ├── models.py           # SQLAlchemy models (Article, CollectionRun, labeling tables)
│   │   └── collector.py        # Orchestrates API collection + scraping phases
│   ├── labeling/
│   │   ├── __init__.py
│   │   ├── config.py           # Labeling settings, prompts, category definitions
│   │   ├── models.py           # Pydantic models for LLM responses
│   │   ├── chunker.py          # Paragraph-based article chunking
│   │   ├── embedder.py         # OpenAI embedding wrapper
│   │   ├── labeler.py          # Claude labeling logic
│   │   ├── classifier_client.py # HTTP client for FP/EP classifier APIs
│   │   ├── evidence_matcher.py # Match excerpts to chunks via similarity
│   │   ├── database.py         # Labeling-specific DB operations
│   │   └── pipeline.py         # Orchestrates full labeling flow with FP pre-filter
│   ├── fp1_nb/                 # FP classifier - EDA & feature engineering
│   │   ├── __init__.py
│   │   ├── data_utils.py       # Data loading, splitting, target analysis
│   │   ├── eda_utils.py        # Text analysis, brand distribution, word frequencies
│   │   ├── preprocessing.py    # Text cleaning, feature engineering
│   │   ├── feature_transformer.py  # Sentence transformer + NER features
│   │   ├── ner_analysis.py     # Named entity recognition utilities
│   │   └── modeling.py         # GridSearchCV, model evaluation, comparison
│   ├── fp2_nb/                 # FP classifier - model selection & tuning
│   │   ├── __init__.py
│   │   └── overfitting_analysis.py  # Train-val gap visualization
│   ├── fp3_nb/                 # FP classifier - evaluation & deployment
│   │   ├── __init__.py
│   │   ├── threshold_optimization.py  # Threshold tuning for target recall
│   │   ├── explainability.py   # SHAP, LIME, prototype explanations
│   │   └── deployment.py       # Pipeline export utilities
│   ├── ep1_nb/                 # EP classifier - EDA & feature engineering
│   │   ├── __init__.py
│   │   ├── data_utils.py       # Data loading, splitting, target analysis
│   │   ├── eda_utils.py        # Text analysis, brand distribution
│   │   ├── preprocessing.py    # Text cleaning, feature engineering
│   │   ├── feature_transformer.py  # EPFeatureTransformer with ESG vocabularies
│   │   └── modeling.py         # GridSearchCV, model evaluation
│   ├── ep2_nb/                 # EP classifier - model selection & tuning
│   │   ├── __init__.py
│   │   └── overfitting_analysis.py  # Train-val gap visualization
│   ├── ep3_nb/                 # EP classifier - evaluation & deployment
│   │   ├── __init__.py
│   │   ├── threshold_optimization.py  # Threshold tuning for target recall
│   │   └── deployment.py       # Pipeline export utilities
│   ├── deployment/             # Production deployment module
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration and risk level mapping
│   │   ├── data.py             # Data loading and splitting utilities
│   │   ├── preprocessing.py    # Text preprocessing for API
│   │   └── prediction.py       # FPClassifier wrapper class
│   └── mlops/                  # MLOps module for tracking & monitoring
│       ├── __init__.py
│       ├── config.py           # MLOps settings (MLflow, Evidently, alerts)
│       ├── tracking.py         # MLflow experiment tracking wrapper
│       ├── monitoring.py       # Evidently drift detection
│       ├── reference_data.py   # Reference dataset management
│       └── alerts.py           # Webhook notifications (Slack/Discord)
└── tests/                      # 548 tests
    ├── conftest.py             # Shared pytest fixtures
    ├── test_api_client.py      # NewsData.io client unit tests
    ├── test_gdelt_client.py    # GDELT client unit tests
    ├── test_scraper.py         # Scraper and language detection tests
    ├── test_collector.py       # Collector unit tests
    ├── test_database.py        # Database integration tests (requires PostgreSQL)
    ├── test_chunker.py         # Article chunker unit tests
    ├── test_labeler.py         # LLM labeling and response parsing tests
    ├── test_embedder.py        # OpenAI embedder unit tests
    ├── test_evidence_matcher.py # Evidence matching unit tests
    ├── test_labeling_pipeline.py # Labeling pipeline unit tests
    ├── test_fp_prefilter.py    # FP classifier pre-filter integration tests
    ├── test_fp1_nb_*.py        # FP notebook utility tests (data, modeling)
    ├── test_fp2_nb_*.py        # FP overfitting analysis tests
    ├── test_fp3_nb_*.py        # FP threshold and deployment tests
    ├── test_deployment.py      # Deployment module tests
    ├── test_explainability.py  # Model explainability tests (SHAP, LIME)
    ├── test_mlops_tracking.py  # MLflow tracking tests
    ├── test_mlops_monitoring.py # Drift monitoring tests
    ├── test_mlops_reference_data.py # Reference data management tests
    ├── test_retrain.py         # Retraining pipeline tests
    └── test_integration.py     # End-to-end pipeline tests
```
