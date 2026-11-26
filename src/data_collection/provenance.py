"""Where an article's FP training label came from, derived at query time (#154, D030).

Nothing here is stored: the source is a pure function of the article's status, its stored reason,
the latest FP prediction for it, and whether a human verified it. A stored column would have to be
re-synced on every relabel or correction.

Sources:
- ``llm``: the labeling LLM decided the label. For ``skipped`` rows this rests on the stored reason
  being one the pipeline writes after a judgment. For ``false_positive`` and ``labeled`` rows it is
  inferred from the presence of an FP prediction: a status moved by hand (``fix_label.py`` keeps
  the reason unless ``--manual`` stores a human marker) also reads as ``llm`` until #143's audit
  log records such moves.
- ``pre_classifier``: as ``llm``, but no FP prediction is recorded for the article: it was labeled
  before the FP classifier existed, or with the pre-filter off and no novelty score to record.
- ``fp_classifier``: a ``false_positive`` whose latest FP prediction is a skip (``skipped_llm``)
  and whose stored reason is that skip's own reason, so nothing has relabeled it since. Excluded
  from training by default.
- ``human``: a human ruled on it: an id in ``HUMAN_RULINGS`` at its ruled status, or a human marker
  as the reason (``FP_REASON_MANUAL``, ``MANUAL_SKIP_REASON``).
- ``unjudged``: a ``skipped`` row whose reply, or lack of one, does not establish whether the
  article has a sportswear brand (``UNJUDGED_SKIP_REASONS``). Excluded by default.
- ``unattributed``: a ``skipped`` row whose reason no pipeline path writes (no reason, or a reason
  left behind by a status move). Kept, and counted separately.
- ``rule``: a ``false_positive`` written by ``cleanup_false_positives.py``'s keyword sweep. Excluded
  by default.

An owner-attested cutoff ("articles before this date were reviewed") is deliberately not a source.
It is weaker than a per-article human ruling; the export offers it as a separate opt-in tier.

The FP export's query (``fp_export_query``) lives here, and so does the #192 backfill's cohort
(``select_backfill_cohort``, D039), which reads the same rows and sources, plus ``failed`` rows.
So do the relabel strata a spot-check measured too often wrong to train on (``EXCLUDED_STRATA``,
#289): an exclusion beside the source, not a source, since the LLM decided those rows.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, NamedTuple
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import load_only

from .models import (
    FP_ACTION_SKIPPED_LLM,
    FP_ACTIONS,
    FP_REASON_AUTO_DETECTED_PREFIX,
    FP_REASON_MANUAL,
    MANUAL_SKIP_REASON,
    JUDGED_SKIP_REASONS,
    SKIP_INSUFFICIENT_CONTENT,
    SKIP_NO_BRANDS,
    UNJUDGED_SKIP_REASONS,
    Article,
    BrandJudgment,
    BrandLabel,
    ClassifierPrediction,
    LabelingRun,
)

LLM = "llm"
PRE_CLASSIFIER = "pre_classifier"
FP_CLASSIFIER = "fp_classifier"
HUMAN = "human"
UNJUDGED = "unjudged"
UNATTRIBUTED = "unattributed"
RULE = "rule"

LABEL_SOURCES = frozenset({LLM, PRE_CLASSIFIER, FP_CLASSIFIER, HUMAN, UNJUDGED, UNATTRIBUTED, RULE})
# Sources the FP training export leaves out unless asked; the export reads this set.
EXCLUDED_BY_DEFAULT = frozenset({FP_CLASSIFIER, UNJUDGED, RULE})

# Labeling statuses the FP export reads, and the class each one is.
FP_POSITIVE_STATUSES = ("labeled", "skipped")
FP_NEGATIVE_STATUS = "false_positive"
# Not an FP training label: the backfill's arm X reads it (#86)
FAILED_STATUS = "failed"

# Human rulings with no audit log yet: article id -> the status the human ruled. A ruling counts
# only while the article still has that status, so a later relabel is not called human.
# Source: #182's comments of 2026-10-09 (corrections, skipped -> false_positive) and 2026-10-10
# (rows confirmed as correctly labeled), and #192's comment of 2026-10-28 (the backfill review; its
# 3 unlabelable rulings are left out, since no export reads that status), corrected by #234
# (2026-10-30), #249's review of the #245 relabel (2026-11-03), #248's template rows
# (2026-11-04), #232's spot-check of the relabeled cohorts (2026-11-18), #238's eval-case
# articles (2026-11-18), #275's eval-case articles (2026-11-20), and #248's review of its
# targeted relabel (2026-11-22), and #289's spot-check of that relabel (2026-11-23). #143 backfills
# these into its audit log and deletes this.
HUMAN_RULINGS: dict[str, str] = {
    # 2026-10-09: corrected to false_positive
    "aabc7f6e-8fc5-4c02-9880-b9e404bd840a": "false_positive",
    "f69140d1-cf99-48a2-8663-22dbae05af1a": "false_positive",
    "16497d46-e8cc-49b6-bd2d-2f0c297d5f5a": "false_positive",
    "d2b995a3-fa6d-4e45-9c42-2d35f6cd1c3f": "false_positive",
    "9ea08c90-be9a-48b1-a7e0-9c3405875fde": "false_positive",
    "852d660f-850c-45e5-9e0c-e8a8cf944208": "false_positive",
    "ad4950e8-8edc-4978-ba0e-ebf6448105a5": "false_positive",
    "477bff74-57b0-4f9c-b552-55a6bba2d611": "false_positive",
    "f5f9f1d3-ab3a-4e05-bafe-709650482b94": "false_positive",
    "0d56c213-d05a-4134-a8fa-08c6b4002076": "false_positive",
    "39a66305-56cf-4b24-9ba9-180d0498cb15": "false_positive",
    "4944b79e-e5af-43e6-b4b8-04624c58f75f": "false_positive",
    "a48bbba6-28aa-46cc-8d91-a54ef93df934": "false_positive",
    "382f2cd1-1b78-453c-8e7f-d91901e4a9f6": "false_positive",
    "3054e143-3cac-4864-854a-3a4b0723b7e6": "false_positive",
    "ae913840-b9ad-4403-a886-73a331f26abf": "false_positive",
    "2029c74e-2b8c-4f63-a200-8520d9ef8d94": "false_positive",
    # 2026-10-10: confirmed as labeled correctly
    "21825f8a-d923-4451-97e3-c305a0490a1e": "skipped",
    "eed1f353-8d0b-40c8-b9ce-cfacb6d865d0": "skipped",
    "0a21c3cd-b005-4401-9c44-21000e795e43": "skipped",
    "80a3cee0-a752-42ca-a593-a575846782e6": "skipped",
    "d8b66c00-ca1b-4e3f-9f6a-ed9509975566": "skipped",
    "55b54027-afc5-4566-ab7f-f6d393bc8343": "labeled",
    # 2026-10-27/28, #192 backfill review: corrected or confirmed as false_positive
    "0fe46113-cc55-457b-84d9-4b9aa043279c": "false_positive",
    "2f503c9b-356d-45e7-a77b-d1b2f33f8af8": "false_positive",
    "42c53c5c-1903-45ca-9349-73c9d252c213": "false_positive",
    "4b6a517e-8ea5-416a-9cee-c01553bdc621": "false_positive",
    "4eff3601-7090-4373-aa0e-45660875e428": "false_positive",
    "56129a38-cb46-484f-8ecd-544d4fe6182d": "false_positive",
    "60e1f528-2205-4922-8ca1-702f40751f9c": "false_positive",
    "6e18e893-6f0e-4d3e-a052-f271b220fdc7": "false_positive",
    "76204e0b-c8f1-46a2-b00d-921badaa3d76": "false_positive",
    "8185da14-35cd-45a4-8ec5-554d8be16524": "false_positive",
    "85aa4594-49eb-4323-b2fe-dcef4466c42b": "false_positive",
    "8e5eafab-599c-4955-a074-3cb56aef968a": "false_positive",
    "95da51df-784e-4da4-99fa-5870bf260f13": "false_positive",
    "ab9b199b-5bd7-4dc1-afd4-72826255d424": "false_positive",
    "b896fbc9-cb88-4346-8be4-80d6e35faf14": "false_positive",
    "baf68bd8-87d4-440e-89ee-596f00c2a475": "false_positive",
    "c0090f16-211c-46cf-b6e3-d71a2a5e0fad": "false_positive",
    "c4f1b564-34ac-4c85-96d5-8763fb142146": "false_positive",
    "dec87309-3c65-4723-ae6b-2a461570aee6": "false_positive",
    "e3cdcf6b-555c-4728-a09c-d7222bef76fe": "false_positive",
    "e611a949-b58e-4f65-a771-ec6c29a10e10": "false_positive",
    "e64d0acb-8746-478f-8b16-b24c7bfea396": "false_positive",
    "e781a7b3-4b00-4cef-981a-369817b48b58": "false_positive",
    "edfc9c0e-01c2-425c-8aef-f5fa4b06340e": "false_positive",
    "f36b2b69-7e4d-4173-a804-94a297954747": "false_positive",
    "fd48d9a7-4993-4651-b770-5d232e70d918": "false_positive",
    # 2026-10-27/28, #192 backfill review: judged skip (sportswear, no ESG category)
    "02e91e09-411e-4bbc-8400-aa06c8d722a3": "skipped",
    "034c0170-c0f0-40b2-990a-868769e75d21": "skipped",
    "0c7cec0f-e4e2-41f5-9192-535a85035bfc": "skipped",
    "10436e7a-67b8-46f9-846b-2869ad4bd191": "skipped",
    "159cd128-f30a-412f-b94f-cead2fd15739": "skipped",
    "160bdf90-3ed9-4f7e-912e-3c920ae88b9f": "skipped",
    "1769af09-0817-42b7-90ca-8b3654376711": "skipped",
    "1cb4d598-9c60-430c-b359-5492043c24a5": "skipped",
    "23d82ecf-d2f7-4ae4-9b8a-93abda969b11": "skipped",
    "27c31c75-d82e-4293-bb69-d10bde2d16d1": "skipped",
    "28c458b0-e6f3-4e69-9214-c723b2306d1e": "skipped",
    "367e1182-7d43-412e-ba93-a99067732e4c": "skipped",
    "3a637c91-2fd3-4f8e-86be-f1ea45780808": "skipped",
    "3d90579b-9121-4a7f-a968-e5ada8ff25a2": "skipped",
    "3f688bb6-1488-4f53-8a4f-93a13a119cf9": "skipped",
    "4c24359f-c478-4cbb-bf85-a22658414a35": "skipped",
    "4ea31398-2774-49f9-a38d-a97404ece306": "skipped",
    "61eab396-2602-42f7-ac02-c2afb02b52ef": "skipped",
    "6321d0a8-b808-4480-a0ee-7ff63313845a": "skipped",
    "66169a09-7c70-47b6-93d1-7081699e18e6": "skipped",
    "6907df87-1f23-47a8-8df1-9915f221253b": "skipped",
    "6ed659de-cb4c-4e18-acfc-5c2e30d1daa4": "skipped",
    "72e8eee7-e9d3-45fd-902c-37020708fd61": "skipped",
    "773a112d-bbe8-4c96-ba37-ea008950d2da": "skipped",
    "83701f55-0565-4874-b4e3-7b2c40c240f4": "skipped",
    "8d1baf3f-afc0-41b0-aa18-77bb9026371b": "skipped",
    "8d546993-05ff-4da1-a07c-3cdc071528fe": "skipped",
    "92459128-f6b5-44fe-9c0a-9a7f1c05970d": "skipped",
    "94a563b7-2c67-4c74-9884-66f7a4f35ee8": "skipped",
    "a4fd8d33-925f-4982-9bd8-21b3d537842d": "skipped",
    "af2d245b-c21c-4237-85cb-946d447fc91f": "skipped",
    "af3a3548-2e10-4786-a35e-1de14fd598d8": "skipped",
    "b87a0c65-dead-46fd-b821-0f7b5ac753d2": "skipped",
    "cb732069-976c-4067-95f3-2165fca597b7": "skipped",
    "d77a84e4-1010-47d6-a5f2-ee891d50f552": "skipped",
    "d8f3fe26-b6c4-430d-b77a-912794ff5b30": "skipped",
    "f1376b2c-6c13-4f1e-b988-0d8d3dc3b40d": "skipped",
    "f71d1159-ad99-4a97-b906-5e43b9aa3d71": "skipped",
    # 2026-10-27/28, #192 backfill review: labeled, with the labels reviewed
    "20dc82b4-23db-4514-a79c-b559f7a3cb40": "labeled",
    "26daa219-5160-430d-af73-c3f831c10393": "labeled",
    "459c2360-b7a4-41f3-b473-748577240ae6": "labeled",
    "544693d7-a9cf-4206-be65-707f1b64baf0": "labeled",
    "6fbe3e58-45b2-4152-9d7a-3e45f4191612": "labeled",
    "7445aca0-ec06-4d6d-9933-c935944e09be": "labeled",
    "7aabb199-fae3-4a94-abbd-e009761b8e93": "labeled",
    "7de2362b-3ba8-4dcd-8bdd-694ac3b16d68": "labeled",
    "925b8c15-2b18-4a3b-a0b2-ab9977402bab": "labeled",
    "9e3818c7-190e-46f9-bf85-94f8c1c951e6": "labeled",
    "a70e6272-10e9-4077-b5b5-7877440e3fae": "labeled",
    "c7c4d22d-cb03-4b73-a5ff-05d06d402917": "labeled",
    "cc511dea-ff2a-4c06-b4a8-9ebcc024dfd2": "labeled",
    "cc938f72-aa6e-4ef1-aa88-48d226fc8e4e": "labeled",
    "de4528d7-280c-4924-853d-95d341be2022": "labeled",
    "ec2dfeec-d41f-47f9-9e24-cf0069c9c18f": "labeled",
    # 2026-10-30, #234: rulings made consistent with each other
    # Puma's job cuts are Social -1, as Nike's are in a70e6272 (was an unjudged skip, "No ESG
    # content found", ruled financial 2026-10-22)
    "fbefe459-5ddf-4e05-b092-1bbc31b823cb": "labeled",
    # Allbirds articles whose subject includes the sale of the shoe business or the brand under
    # American Exchange: sportswear, no ESG category (were false_positive; stored with
    # MANUAL_SKIP_REASON); a passing mention of the sale stays false_positive
    "26b954bc-b255-49e8-a11f-a95c02eb22b2": "skipped",
    "2bfbbec7-7f4b-41ae-86a8-f71dc883e668": "skipped",
    "666c3315-b36e-47eb-a338-ef62813b17e1": "skipped",
    "6a0cbd28-e9c8-403b-8b0f-a5fe966a00f5": "skipped",
    "8f5ec01f-1e3c-486e-9f93-d98f8bdd36e6": "skipped",
    "d2873bbb-5c94-4343-ad8d-4feb83f36c59": "skipped",
    # 2026-11-03, #249: the owner's review of a random 30 of the #245 relabel's business rows that
    # went labeled -> skipped (seed 2451). Adidas's digital-marketing shift is Digital +1, a human
    # label; the other 29 are sportswear with no ESG category, stored with MANUAL_SKIP_REASON
    # (3f929135, Decathlon in India, borderline on its supply-chain angle)
    "57acb8be-eff4-456c-83f4-9883b3baf79a": "labeled",
    "05b0ed9e-00e7-40b4-8c14-748c7ceb84c5": "skipped",
    "06bf2048-41bd-42af-9c67-69967d9cfa59": "skipped",
    "10a991a5-8dff-4366-91ad-28c072b1c6bd": "skipped",
    "114d005e-8ee7-4f5a-8aa4-5630bce18904": "skipped",
    "178babec-795e-4299-a4ce-906a885b4d0d": "skipped",
    "1a2f4608-737a-4e32-9845-e904f55ac5d6": "skipped",
    "1cff9829-9d63-4196-83cd-de02e6461bf9": "skipped",
    "2f543b14-9d17-4f69-8fb6-c9ddea8ae259": "skipped",
    "375a058a-1ade-4f2b-9fb3-c20cde102058": "skipped",
    "3f929135-2ef9-45f5-bd87-0a34417b188f": "skipped",
    "41d5c4f6-ef5d-4ba3-9276-54a45d03b815": "skipped",
    "5c460161-9b2d-4df3-ac80-5c8636c7241b": "skipped",
    "770ccab6-6faf-4f62-a9aa-c6e02e22a378": "skipped",
    "79f08609-ef69-4844-be98-5cd0ba4679f8": "skipped",
    "8a04bf31-e8df-4b43-8823-b07b59154b50": "skipped",
    "8b0719e4-faec-49f2-ae25-8d6d3de74706": "skipped",
    "9acbf098-8757-4759-851f-3850bfa08784": "skipped",
    "a68a94a8-9cf4-498f-806f-3be53bc5bc58": "skipped",
    "a6f829bc-93c4-4cca-aead-14fb6cbe9db4": "skipped",
    "ad284937-9624-471b-b06b-4f0d8b20c3ce": "skipped",
    "b5f67cf2-bd06-48c9-8b0a-19f18a4c1428": "skipped",
    "b6f65f81-d68b-402c-a23e-a262c89b2dea": "skipped",
    "b88b183f-d09b-4b9f-8da1-da1e5e7975e7": "skipped",
    "bc569afa-a791-45b2-9dbb-9d0cfdd24707": "skipped",
    "d2bc8107-6b34-4a7c-8000-de7d9b91fe42": "skipped",
    "e4ea1abd-7581-42e7-827d-dc8e4d7c2a98": "skipped",
    "e9275b13-3e4e-469c-bddd-2bf3cd51cfd1": "skipped",
    "f95d95a9-e882-42e2-b141-62aba5e2a5dd": "skipped",
    "ff3d9904-3260-49bb-b231-0a260c27b6c0": "skipped",
    # 2026-11-04, #248: template or contentless stock pages that v1.13.0 judged `financial` while
    # its own reasoning called them templates or empty; mention_only under D040 ruling 6 (were skipped by the
    # #245 relabel; stored with FP_REASON_MANUAL). v1.14.0 takes them as cases before the retrain (#248)
    "02067ad1-d4dd-4c94-9af6-0d23d2d2122f": "false_positive",
    "201cc55a-2e10-404e-896b-8c9041f02203": "false_positive",
    "45e8c505-57ba-436c-ae91-b7ec261b695d": "false_positive",
    "711b5caa-e4c1-4128-b7f1-bfaa7442eee0": "false_positive",
    "7bf35563-a928-48ad-8c10-b5e8dd556eba": "false_positive",
    "81ab6c27-f216-431b-bda7-621bb3bac187": "false_positive",
    "9cd2ca15-a112-4fe1-8cfa-3ee8b51b0333": "false_positive",
    # 2026-11-18, #232: the owner's review of a seeded random 50 + 50 of the #245 relabel's
    # false_positive and judged-skip rows that the FP export reads (record on #232). Corrected
    # false_positive -> skipped: sale pieces that describe the product, and brand marketing,
    # sponsorship or survey content (stored with MANUAL_SKIP_REASON)
    "3bb19314-ab3a-4a52-977f-09eb2c5df027": "skipped",
    "4068c564-5de9-4174-9f5e-45c7b9ea691c": "skipped",
    "45d3a491-db08-4d41-9826-c59c17759fd8": "skipped",
    "6c12f566-f74d-45f3-b83d-254190b88b7a": "skipped",
    "83c84d1f-1039-43b6-a12e-5bb904ae99dc": "skipped",
    "c9c97e73-1c8b-4303-8640-ece1c5893d07": "skipped",
    "ca703893-209b-49bf-89f9-738ddee5ccdf": "skipped",
    # 2026-11-18, #232: confirmed at their stored status (stored with the status's human marker);
    # the one unlabelable ruling is left out, since no export reads that status
    "015e428f-2da3-4eb3-8887-3e53d1145745": "false_positive",
    "0676a92b-f947-428e-a6c3-31e51cb64ffc": "false_positive",
    "0d42d019-3335-45ac-8811-40ffbd48b7fb": "false_positive",
    "113483ca-da72-483a-8702-80cdd2f2d2e6": "false_positive",
    "18fa1c18-a306-4759-9bd4-6851351aed7b": "false_positive",
    "24881992-ca0f-4325-810c-bef054ee4697": "false_positive",
    "25d2a7a1-1ecb-4002-ae8d-7d6481289594": "false_positive",
    "27640210-8aa2-4072-aeb5-0d4b7cdc5d33": "false_positive",
    "276f3d4f-4f99-41f4-b17d-04b958b23c64": "false_positive",
    "344ab574-d72c-4199-8c53-2acd0f992a74": "false_positive",
    "35db2a8c-a6fb-4219-b6db-4f09d8dbad71": "false_positive",
    "3b648ae2-d3b3-4794-aa9d-d208ade65ac1": "false_positive",
    "3e06958f-08f9-4624-9e5e-8543b31de1db": "false_positive",
    "457e531a-9468-4dca-886a-2ce97ddcb8ad": "false_positive",
    "4ced25e3-a9d1-4747-a7f2-08c949f5c0a5": "false_positive",
    "4f09bf65-7c4f-45a6-85ea-a6fa42bc10a7": "false_positive",
    "552a4cf6-bb63-4d67-8fc2-b052285f44f6": "false_positive",
    "659c1a9e-311a-4ba2-aa36-51442233bd5f": "false_positive",
    "6912a731-f0e7-41fa-b305-52daf2439a6d": "false_positive",
    "6d7f98ac-fbad-4f15-a8d7-5d67a2e60896": "false_positive",
    "72c2b9a4-1d1d-4dd2-9c06-76dd8598b545": "false_positive",
    "73025688-59ee-4c22-af49-ec03f95739b4": "false_positive",
    "7c3737a5-54e4-49a2-8765-66c62e4c4781": "false_positive",
    "7e64bcd2-462a-4fb5-9ac8-426d049ef1bb": "false_positive",
    "88dd2403-bc27-495d-af1d-c63ae8bf2f83": "false_positive",
    "89c1bea8-a919-4c23-b72b-47cf676a85d7": "false_positive",
    "93dbe427-ffea-4fb9-a1ac-43ed15466779": "false_positive",
    "99142f57-6642-4e08-983e-3f28b8168c8d": "false_positive",
    "a087f586-5d8e-44bc-b56a-a133ed620c24": "false_positive",
    "a1c5cb7d-3d33-42c2-9b60-5dfe20e396ad": "false_positive",
    "a5929710-a078-4608-bf33-758d45d698dd": "false_positive",
    "aa37cad0-25ac-49a1-9aa4-db23b0187e28": "false_positive",
    "acd876eb-f0d5-43cf-a88d-b78eb0fefb7e": "false_positive",
    "b6cb6b76-f9c4-4de7-ba0a-5e6a8d771cf3": "false_positive",
    "bcf60434-4d1c-4966-a1b6-700ce0f492c9": "false_positive",
    "cd7c7316-5a34-4e8b-a764-527fe09a8c6f": "false_positive",
    "ce372079-8e29-423f-a254-937cf7dfb47a": "false_positive",
    "d98e038c-a55a-414d-8901-c5a3b6496f3d": "false_positive",
    "db70d846-048d-49ef-8839-3ac35946cd6c": "false_positive",
    "dd31a1b2-7151-4843-be2a-e3f5c7d5e921": "false_positive",
    "de4c91b9-68e9-47d9-af00-727e0b730fa2": "false_positive",
    "ebe30521-4542-4b78-a80b-c1ace5de974d": "false_positive",
    "0f08ab65-2a29-479d-9509-d521f07f82a2": "skipped",
    "16633ce9-af72-4d69-bb16-f940fc30bd6c": "skipped",
    "17246bda-9850-453e-b9b1-d04a0a0050cc": "skipped",
    "1a38ceac-d182-4655-8907-ab6ca149522f": "skipped",
    "1c1cae02-642e-4a88-a168-baa837cc7288": "skipped",
    "20c93b07-d693-4da5-84ec-7e1d77e7159a": "skipped",
    "21c6c57b-962a-448c-9f8e-f46bcc6776a7": "skipped",
    "24e0774b-a9f1-4f72-9c69-3eed51dd452c": "skipped",
    "29544f50-9ccd-4be8-9d46-8844867036da": "skipped",
    "3129b121-7952-411f-bf37-0a79955ad989": "skipped",
    "32cd3ea4-1e3e-46b0-8464-6b47a911bd3e": "skipped",
    "365fd5ee-5ba6-4bd9-a985-c767a47956c8": "skipped",
    "3e451b63-63da-4f5b-992f-3236ecc91cfb": "skipped",
    "478481bd-5ffb-471c-ab25-ab909c65cfbf": "skipped",
    "485f99c7-76b5-4d94-93a0-e3628b0cb268": "skipped",
    "49bc3f9d-bcec-42c0-a808-0e2fd87d842f": "skipped",
    "55a5ae80-464f-431e-8b21-894dd3e4c8f9": "skipped",
    "5ca11583-c61e-4ea0-9410-17905d884097": "skipped",
    "5fef9823-3c74-489d-a02b-2a91fc1d2ef9": "skipped",
    "6039171d-2c82-4291-b6dc-759d5162012b": "skipped",
    "61177c16-3824-44d9-b2c9-aec88a24f937": "skipped",
    "66d5c909-e24e-4daa-9b50-dfe5c4419dd5": "skipped",
    "677e2628-ee84-4c76-b61c-b6adf6c3c1e8": "skipped",
    "686eacf2-1268-480c-96a8-034b10cbe1ec": "skipped",
    "6d0702f4-a0f7-44f9-be39-652fa7b3e35f": "skipped",
    "6d85addd-23da-488f-8972-6b0f319c9be0": "skipped",
    "72a405e3-9b43-46b4-892d-5c93a24ac982": "skipped",
    "7301d1df-b484-4eb1-9306-95c5553c52e0": "skipped",
    "74e9fcb4-1c99-4d3f-b657-e2bec66bf6d2": "skipped",
    "7fe795b6-07cb-40f8-b48d-1ce01524dc42": "skipped",
    "8295787f-44cc-47e0-97d2-648d58775dcb": "skipped",
    "86bcd12a-7d48-4ae4-8eb9-282e507cbd2d": "skipped",
    "86fd6176-ca5f-4302-ab27-f41a42750548": "skipped",
    "8ee2aaaf-256e-4c08-9586-2ba82ebc8d84": "skipped",
    "951a5b01-41db-4f9e-9efe-18f313ba5989": "skipped",
    "a23208d5-dd91-4945-a424-6bcc61a71339": "skipped",
    "b1435c5f-e2e0-42e5-a171-4776e70da913": "skipped",
    "b926dec8-a769-4a5a-98d2-b7f87341cad9": "skipped",
    "bdde9956-17c7-4937-adcc-3c3072c8dd77": "skipped",
    "bfffbbb7-b548-48d9-91a8-595826f4ee4c": "skipped",
    "c0cfe209-ce4e-4479-b9f5-848052f0fd94": "skipped",
    "cafccc89-a47b-4e51-ac43-ef94513e96ae": "skipped",
    "cff775db-dffa-41f1-b7e6-4d8402891ae5": "skipped",
    "d93d717d-be1a-4ee9-bd53-a59c18474316": "skipped",
    "daa65205-d8f3-4368-9f2b-cea104e4fd5f": "skipped",
    "e5f2d166-b3cd-48ba-ac94-2c14af9d1216": "skipped",
    "f0906277-41d1-48f2-bfe6-6c79d02dc70a": "skipped",
    "f2756068-5c76-4c62-8685-862cf869e710": "skipped",
    "f6e9211d-9248-4cfe-9e35-bcc99f12d1a1": "skipped",
    "fcabee62-40e7-450f-87d3-ae3a81cb48f4": "skipped",
    # 2026-11-18, #238: the owner's rulings on eval-case articles (record on #238). Sale pieces
    # that describe the product (442282ca, bcc52f23, 6ec5fd75, eb5aa27c), a sale piece with business
    # content (02ffd32e) and market wraps with analysis of the brand (a7eeaa13, b8891fbc) are
    # sportswear with no ESG category, stored with MANUAL_SKIP_REASON; Nike's planned layoffs are
    # Social -1, a human label (19832789). 2026-11-20, #248: the owner also ruled a7eeaa13's
    # Adidas (UBS's downgrade with its stated reason) sportswear, financial; the status is unchanged
    "442282ca-90ae-4811-a7f5-1026bc8fcf75": "skipped",
    "bcc52f23-6048-40e3-ab75-ed701468a5a2": "skipped",
    "6ec5fd75-5e7a-4e06-b53f-58b4326cddba": "skipped",
    "eb5aa27c-2fc8-4db4-a5b7-a202b116c4ea": "skipped",
    "02ffd32e-cca0-476d-88f8-1486c790f421": "skipped",
    "a7eeaa13-de2d-4ed6-83ce-64f74a4bbda0": "skipped",
    "b8891fbc-7d8a-40b2-9b9b-527d9d373858": "skipped",
    "19832789-ca96-4c74-ba7a-7df0453c4874": "labeled",
    # 2026-11-20, #275: eval-case articles whose case notes recorded an owner ruling (#118's human
    # corrections and #215's Product and financial rulings, 2026-10-22) that had not reached this
    # list, and three the owner ruled on #275 (22a41460, 3c8017cb, cfe9e317). Each is at its stored
    # status, which its case expectation maps to (D045). Sale and deal listings that give the brand
    # only retail details, and a brand named only in a list of tariff-refund beneficiaries, are
    # false_positive; UGG's grant and donation are Social +1, a human-reviewed label (cdb26552); the
    # rest are sportswear with no ESG category
    "b7eea904-4747-4974-a958-abd78ab0df32": "false_positive",
    "e7968869-86d2-4692-b06c-071d7a9b1de4": "false_positive",
    "63f9fd9b-a7c8-4e49-8baf-52291f925d44": "false_positive",
    "6d6195b9-5b71-478f-bc38-63e5743381b6": "false_positive",
    "de535470-9a93-413a-a69b-e8ca7b6d0028": "false_positive",
    "e434b12d-eb59-4406-bd26-f3bad800aaf1": "false_positive",
    "cdb26552-8643-4007-a403-4f0451447e2f": "labeled",
    "2d4ff51c-d1e9-49e9-bb48-48561bbdd06c": "skipped",
    "12396c87-b2a4-4563-ac1b-d2923202afe2": "skipped",
    "2df21233-ef21-4ec7-87b7-44334f90ac19": "skipped",
    "1ba1c2a2-ac8a-4077-ac44-11feae1a08ec": "skipped",
    "22a41460-1976-41e4-9a36-e0fe0943441d": "skipped",
    "23c30b60-9671-4613-a93a-893d14830a14": "skipped",
    "a058da6e-8656-4dc7-901a-fc1d56d4b745": "skipped",
    "ea428de7-23a8-4004-80a7-5323c26e2f1a": "skipped",
    "9e973f0c-458b-4d3d-bf2a-55f3e3439149": "skipped",
    "eded2885-014c-45d1-8164-4bd6605a0618": "skipped",
    "3b4de2e6-78d4-47b9-bea1-fe298967a6d0": "skipped",
    "7dccbf22-de0e-4e60-acb5-f562b551fafc": "skipped",
    "cfe9e317-aed2-4c28-9c7b-ca9450d14e59": "skipped",
    "89330b08-e205-48b3-a233-ecce8b03b2d3": "skipped",
    "3c8017cb-e7f6-47dc-83ea-4d59468118ef": "skipped",
    "3fee9d78-c7a8-4936-8ff0-2b14edadbcce": "skipped",
    "a39e8011-0d4b-4c77-b7c9-c4b8857d7f72": "skipped",
    # 2026-11-22, #248: the owner's review of the 15 rows #248's targeted relabel moved to labeled
    # (record on #248). Template stock pages whose ESG item is only in an appended news summary
    # (2d031846) and a supplier's plant closure that gives the brand nothing of its own (5dff6b2e)
    # are mention_only; a one-item headline list (726d045a), the brand as a theft victim (0b2efd6c)
    # and an opinion piece on its marketing (def4658d) are sportswear with no ESG category, skipped
    # rather than false_positive so a contentless list stays out of the FP negatives; critics'
    # attack on its DEI goals is Social 0 (49b43ed7). The other 9 keep their LLM labels, unruled
    "2d031846-82ae-4351-8aab-b15a0ecbf86a": "false_positive",
    "5dff6b2e-d749-4433-b843-62933b946c26": "false_positive",
    "726d045a-d575-4bdc-a421-dcae313c5e61": "skipped",
    "0b2efd6c-f6fd-4b3b-98fb-1981027c8e7a": "skipped",
    "def4658d-405c-4431-a03c-f0f9a18dca36": "skipped",
    "49b43ed7-442e-47fd-9290-5b0253a11632": "labeled",
    # 2026-11-22 to 2026-11-23, #289: the owner's spot-check of #248's targeted relabel (record on
    # #289), confirmations included. A stratified seeded sample (arm x moved/kept), then the SK
    # review band's follow-up: every SK row not sampled whose text is under 1,500 characters.
    # Corrections: short financial pieces with no judgment about the brand are false_positive
    # (mention_only); a results story with depth is financial (20262ea9); a product named in
    # passing, and a festival named after its sponsor, are mention_only; special-edition products
    # and sale pieces that describe how products are built are product. Policy points: #290
    # sample, skipped -> false_positive (all 17)
    "08698912-e3cc-401a-a838-1d7db38de1ce": "false_positive",
    "20262ea9-058f-497e-a606-56b8e9d7451c": "skipped",
    "2341c37f-365c-4d2e-9a69-1fde09f1df9c": "false_positive",
    "369c8c12-f07a-469e-87ab-c59975aefc17": "false_positive",
    "57f9d796-7e62-4e65-b266-832a27adf6ea": "false_positive",
    "5813b017-726c-43fb-bf77-92a995f03158": "false_positive",
    "5f9a1ae9-d043-4618-8b26-de825834365e": "false_positive",
    "97cdff77-7e96-408c-8724-7568a0183781": "false_positive",
    "9eaec946-f934-4a80-8ed4-97871e6e9717": "false_positive",
    "9ef5f6f5-87a0-4049-813d-6e7d77372628": "false_positive",
    "c00511dc-40fa-4f7e-aff3-8529ac5c5d30": "false_positive",
    "c54f4d02-e86b-4d4d-b453-1a07a69ec42e": "false_positive",
    "cd6d7ec6-eba6-447e-b199-d28fd2e6e51c": "false_positive",
    "d9ba94c6-f01d-49ff-a759-c85ab398a641": "false_positive",
    "ed519615-99b3-46cf-a2d3-e7bb54a8ad4b": "false_positive",
    "f6018639-4758-4cce-99c9-34304797a8a6": "false_positive",
    "f6e0a6d9-ff3a-4cb2-a824-d066bccbbe83": "false_positive",
    # sample, false_positive -> skipped
    "0dd8635f-1e61-49c6-9d10-5dc462855137": "skipped",
    "1941aa19-c505-4733-8b6d-76fee0c568cc": "skipped",
    "1fe66929-a227-4ce3-a7fb-78df8661c7c9": "skipped",
    "20cbcd27-e7d9-48f8-83be-9b76fa34b8a9": "false_positive",
    "2507def1-c0c8-4f20-b423-603e7160583e": "skipped",
    "270851f7-3ede-4f8a-a7de-9d7389c4630c": "skipped",
    "3ab1d0de-64f2-485e-8540-251007e4ab24": "skipped",
    "451b4bc1-4d88-4a0d-b59e-f714596b381c": "skipped",
    "4cb73587-857e-4927-8c8e-0451466101ee": "skipped",
    "59dc0121-fd1f-401d-b2ca-518ba83928fb": "skipped",
    "5a3855c2-8d7a-4d08-b5d6-b16ad7a3ec55": "skipped",
    "6cd61de9-fbc3-453e-a5eb-c324ad95eb2a": "false_positive",
    "7e9e2f7d-1dae-412b-91bc-52687071803e": "skipped",
    "8020c343-40c6-4804-9649-7b1fe8d8c3b3": "skipped",
    "8d653551-d92a-4dcc-81df-f7f77f520a9a": "skipped",
    "9ed226ad-5116-4b61-b764-bdf83124eb4c": "skipped",
    "a4f1323c-d0ed-468b-9718-947a4c4f4bfd": "false_positive",
    "b08812e5-1f65-4f68-970e-c99bafa3b190": "skipped",
    "b55d807c-250c-4482-83c7-204d56f32845": "skipped",
    "b9ebdd6e-0141-4716-b917-17ee397f5a8b": "skipped",
    "bd95b776-7f38-41a6-bf30-4da4c976bcb5": "skipped",
    "becdd916-8e1d-48bc-a27b-e1d77a89989c": "skipped",
    "c701282e-d62f-4a4a-9e6b-55453acea2c6": "skipped",
    "cc2393f9-5ce4-4449-b452-c84120d64bd2": "false_positive",
    "cf7846a7-3390-47dd-867c-265ad863ac9c": "skipped",
    # sample, kept false_positive
    "087bcb43-f889-449a-86b6-c788e117ba5a": "false_positive",
    "0956d655-01e5-42e4-8cea-8805f9711831": "skipped",
    "0ad16d3e-baca-4a52-b3d2-e0d9733a0bbc": "false_positive",
    "1410e872-4595-4f21-8310-3e6745a4c7a7": "false_positive",
    "14aaafc0-ffda-44e6-86c9-c14deec4294f": "false_positive",
    "255bd0e9-6e03-433f-ab6e-5a026c2cf6f1": "false_positive",
    "2862facc-3671-41cf-9c7d-0ea4fc808115": "false_positive",
    "295b00af-f2fd-43ad-a37d-a5dd57bf299e": "false_positive",
    "2f4e1707-0273-4669-b6a3-757eb5b549ea": "false_positive",
    "307ef2b5-9e15-47d2-934b-84d94001d900": "false_positive",
    "39c85891-7d2d-4a5a-8ab5-ae5e3c3858ab": "false_positive",
    "4b685070-e2d3-4952-945e-0732c677f839": "false_positive",
    "5b14ba34-99bc-49a1-920e-b10e61fb2837": "false_positive",
    "8d8c57ae-ae90-43f2-99e9-29291708e70f": "false_positive",
    "979988e9-ee1a-4557-a829-5be7c5db1e62": "false_positive",
    "9e8e9f67-e5ee-4348-860f-43d885df502a": "false_positive",
    "a1d1d234-f3c5-4340-b2cb-26bf5bb1eacd": "false_positive",
    "a985ff0b-4e75-4554-a51c-ad6bb8a1f1fc": "false_positive",
    "b2b4e56f-26e1-487b-aa4b-388024759deb": "false_positive",
    "bbd189a2-aced-41b0-93d8-3406c03daa81": "false_positive",
    "df2f7eaa-6f44-48c9-a1de-b3528462681d": "false_positive",
    "e6379267-351a-47ca-a3e1-908ec6c28c6a": "false_positive",
    "f4edfcf8-9c4a-499c-a9ef-66b46c9bae6d": "false_positive",
    "f7972deb-3d30-400b-8aa1-dd198ccd606f": "false_positive",
    "feb4eb70-2a33-49f7-98e6-bf5fa609fa34": "false_positive",
    # sample, kept skipped
    "0275d5cf-a163-4522-b8b9-ffbdd8644709": "skipped",
    "06235693-8e15-4b84-a3b0-3f1695fed414": "skipped",
    "085681fd-c878-4053-bed2-930fc9db742f": "skipped",
    "1190fb25-a4e9-4828-8cd9-407b5f1559a4": "skipped",
    "11db93f0-fe99-4b14-88f4-c2018522da47": "skipped",
    "23d33195-add8-420a-9635-904ce6671482": "skipped",
    "255f1c93-e806-4762-959f-c1a438a0df42": "skipped",
    "28500898-6155-4e7e-bd15-41e84431a4ca": "skipped",
    "2d25ae36-9217-4672-80a7-ea883d82d1fc": "skipped",
    "352e37ee-3136-4111-9a0b-c330914876d4": "skipped",
    "4eec0073-1bb0-4907-a2ae-287908984977": "skipped",
    "517afde7-f828-46c6-b368-b74b2e9bf693": "skipped",
    "5b798ea4-0a67-41b4-b7ac-97fab7838217": "skipped",
    "7a8a2b01-88d0-4feb-a5fa-20673d7c2d79": "skipped",
    "7bd9b616-646b-433f-81c4-6bc5797b6e7e": "false_positive",
    "7e443ef7-b56e-4d3c-82f1-e670a6a3adec": "skipped",
    "93eb3743-4cfb-4087-93ab-0ca85b784875": "skipped",
    "ad9a5269-8c54-42e1-b291-4e1c2bbb8048": "skipped",
    "aeb43202-98ad-47cb-ad69-02c33939ec13": "skipped",
    "b0f1187a-ce88-40f8-ab3d-02fa1628d7de": "skipped",
    "b115bed0-6eff-47f7-bd8f-aeaa2cab98de": "skipped",
    "bad23cea-9817-4f81-a453-36985b4b486c": "skipped",
    "bb592579-e80f-4955-8c9e-61ec6db40b33": "skipped",
    "bfd4ac24-8245-43fa-8878-177f88745b89": "skipped",
    "c30cf33a-8bb2-463a-a0a3-d9e3dc0ea261": "skipped",
    "c827dbb4-133b-4d47-851a-dfe821374477": "skipped",
    "ce7062ca-4b71-4ec4-8ff4-375ec7fa89c1": "skipped",
    "d418136b-2344-4d98-9be7-da82c16cab5e": "skipped",
    "dd397a16-4301-4b87-bb16-a2196e2f1464": "skipped",
    "deca91df-4b8c-413f-8386-913f00a33ff6": "false_positive",
    "eb6a6297-91e9-4eb1-bebd-4866910fe0b7": "skipped",
    "f8f2713c-e567-4d17-89d4-0a5cf84051ec": "skipped",
    "fc85972e-570e-4047-8c21-d4819724799c": "skipped",
    # SK rows not sampled with text under 1,000 characters
    "0f8d488c-44eb-4815-ac20-714b562f650c": "false_positive",
    "14ff9195-1cde-4820-b1cc-290053b71c66": "false_positive",
    "1e082390-1970-4781-bcdc-5adbdf354bd3": "skipped",
    "388eaa73-03aa-4ab7-91cc-d43a99678aed": "skipped",
    "55c473d2-9b82-4e79-985b-9b9d0d76f63a": "skipped",
    "56845a93-425e-4fac-a377-9088d0a75f2c": "skipped",
    "60ffff40-e1f7-415a-b6f0-284f2e5947ba": "skipped",
    "799d0f65-0e2b-4694-a9cd-5211bdeda6e9": "skipped",
    "817009ec-b8ff-433e-a72b-59338a2e2ffe": "false_positive",
    "8803d7f5-f5af-4d59-8033-b9ffa664ac04": "false_positive",
    "91031fcc-c40b-4add-bed8-3ce2bafc9e43": "skipped",
    "a68fdab7-0f54-4213-9f58-09fafee246ed": "false_positive",
    "afe77d4c-b315-4ca3-be7e-a32e5302fdd3": "false_positive",
    "b2864a26-037a-47cc-bdec-01886c5286f6": "skipped",
    "bf42a435-e70d-47f7-bd12-c6d6897beb11": "skipped",
    "e19bbbe4-0146-42b1-b3ba-0115f2f2f4a5": "skipped",
    # SK rows not sampled with text of 1,000-1,500 characters
    "135068ac-2ef6-4e80-b217-0cc69c74e571": "skipped",
    "35e508ea-a2f2-407c-9d73-c2cecd7c6b8d": "skipped",
    "3e59bc38-d6a8-437b-b2ee-aa52b5d13ada": "skipped",
    "40d42cba-6e7a-4304-a445-c54a8ee37ba0": "skipped",
    "65c56d8b-c811-45fe-acc6-6612cb7bd688": "skipped",
    "6a787be4-2159-477f-8b3c-63f2ac3e590c": "skipped",
    "6d0d8a7b-8098-443b-b319-ab8adce42df8": "skipped",
    "9dc87651-dbab-41ff-a38d-8b9e5fe880d6": "skipped",
    "c737ac71-3406-404b-ac34-15d4db147259": "skipped",
    "d2835c20-c6db-42ae-aa2d-6c22752ac367": "skipped",
    "d425bd98-76f5-4d47-b481-a81cd4b8ce79": "skipped",
}


class UnknownFPActionError(ValueError):
    """An FP prediction carries an action_taken this module does not know how to attribute."""


def human_verified_ids(candidates: Iterable[tuple[str, str]]) -> set[str]:
    """Ids, among ``(article_id, labeling_status)`` pairs, whose status a human verified.

    Today the only source is ``HUMAN_RULINGS``. #143's audit log and #146's confirmed review samples
    are further sources; they need a database query, so this signature takes a session when they
    land (D030).
    """
    return {
        article_id
        for article_id, status in candidates
        if HUMAN_RULINGS.get(article_id) == status
    }


def derive_label_source(
    labeling_status: str,
    labeling_error: str | None,
    latest_fp_action: str | None,
    latest_fp_skip_reason: str | None = None,
    *,
    human_verified: bool = False,
    article_id: Any = None,
) -> str:
    """Return the source of an FP training label. See the module docstring for the vocabulary.

    ``latest_fp_action`` and ``latest_fp_skip_reason`` are the ``action_taken`` and ``skip_reason``
    of the article's most recent FP prediction, or ``None`` if it has none. Reasons are matched by
    exact text (prefix for the keyword sweep).

    Raises:
        UnknownFPActionError: ``latest_fp_action`` is not one of ``FP_ACTIONS``.
        ValueError: ``labeling_status`` is not one the FP export reads.
    """
    if latest_fp_action is not None and latest_fp_action not in FP_ACTIONS:
        raise UnknownFPActionError(
            f"Unrecognized FP action_taken {latest_fp_action!r} on article {article_id}"
        )
    llm_era = LLM if latest_fp_action is not None else PRE_CLASSIFIER

    if human_verified:
        return HUMAN

    if labeling_status == FP_NEGATIVE_STATUS:
        if labeling_error == FP_REASON_MANUAL:
            return HUMAN
        if labeling_error and labeling_error.startswith(FP_REASON_AUTO_DETECTED_PREFIX):
            return RULE
        # The pipeline stores an FP skip's reason as the prediction's skip_reason. A different
        # stored reason means something relabeled the article after the skip without recording a
        # new prediction (an LLM run with the pre-filter off and no novelty score), so the skip
        # did not decide this label.
        if latest_fp_action == FP_ACTION_SKIPPED_LLM and labeling_error == latest_fp_skip_reason:
            return FP_CLASSIFIER
        return llm_era

    if labeling_status == "skipped":
        # The FP classifier never writes 'skipped', so the latest FP action says nothing about
        # who chose this status; only the stored reason does.
        if labeling_error == MANUAL_SKIP_REASON:
            return HUMAN
        if labeling_error in UNJUDGED_SKIP_REASONS:
            return UNJUDGED
        if labeling_error in JUDGED_SKIP_REASONS:
            return llm_era
        return UNATTRIBUTED

    if labeling_status == "labeled":
        return llm_era

    raise ValueError(f"No FP training label for labeling_status {labeling_status!r}")


def human_label_criterion():
    """A brand label a human reviewed or wrote: the criterion the cohorts and the daily LLM
    analysis's samples (#252) share."""
    return or_(BrandLabel.human_reviewed.is_(True), BrandLabel.labeled_by == "human")


def latest_prediction_subquery(classifier_type: str):
    """Subquery of each article's most recent prediction from one classifier.

    Columns: ``article_id``, ``action_taken``, ``skip_reason``, ``created_at``. Outer-join it to
    ``Article`` on ``article_id``; an article with no prediction gets NULLs. Ties on ``created_at``
    are broken by ``id``, which is arbitrary but fixed.
    """
    ranked = (
        select(
            ClassifierPrediction.article_id,
            ClassifierPrediction.action_taken,
            ClassifierPrediction.skip_reason,
            ClassifierPrediction.created_at,
            func.row_number()
            .over(
                partition_by=ClassifierPrediction.article_id,
                order_by=(
                    ClassifierPrediction.created_at.desc().nulls_last(),
                    ClassifierPrediction.id.desc(),
                ),
            )
            .label("rn"),
        )
        .where(ClassifierPrediction.classifier_type == classifier_type)
        .subquery()
    )
    return (
        select(
            ranked.c.article_id, ranked.c.action_taken, ranked.c.skip_reason, ranked.c.created_at
        )
        .where(ranked.c.rn == 1)
        .subquery("latest_prediction")
    )


def fp_export_query(session, since: datetime | None = None, *, also_failed: bool = False):
    """Articles the FP export reads, each with its latest FP prediction, in a fixed order.

    The order matters: the training splits use ``train_test_split(random_state=42)``, which
    assigns rows by position, so an unordered query would split the same data differently. The
    #192 backfill selects its cohort from the same rows (``select_backfill_cohort``), plus the
    ``failed`` rows (``also_failed``), which the export never reads.
    """
    base_filter = Article.scrape_status == "success"
    if since:
        base_filter = and_(base_filter, Article.created_at >= since)
    statuses = [*FP_POSITIVE_STATUSES, FP_NEGATIVE_STATUS, *([FAILED_STATUS] if also_failed else [])]

    latest = latest_prediction_subquery("fp")
    return (
        session.query(Article, latest.c.action_taken, latest.c.skip_reason, latest.c.created_at)
        .outerjoin(latest, latest.c.article_id == Article.id)
        .filter(base_filter)
        .filter(Article.labeling_status.in_(statuses))
        .order_by(Article.id)
    )


# --- The #192 backfill cohort (D039) ---------------------------------------------------------
# Rows whose stored status the LLM decided, or could decide, without a stored judgment for each
# brand, and failed calls the LLM never judged (#86): the backfill relabels them under the
# production prompt. One arm table, so a later backfill (#224) adds an arm rather than a second
# selection.

ARM_UNJUDGED = "unjudged"  # U: a skip that never judged sportswear status, at a reason a reply writes
ARM_FALSE_POSITIVE = "false_positive"  # F: an LLM-decided negative
ARM_JUDGED_SKIP = "judged_skip"  # J: a judged skip written since the production version's first run
ARM_FAILED = "failed"  # X: a failed call the LLM never judged, most with no reply at all (#86)
BACKFILL_ARMS = (ARM_UNJUDGED, ARM_FALSE_POSITIVE, ARM_JUDGED_SKIP, ARM_FAILED)
# Statuses an arm reads
BACKFILL_STATUSES = ("skipped", FP_NEGATIVE_STATUS, FAILED_STATUS)

# --- The #245 relabel cohort ---------------------------------------------------------------
# LLM-decided rows stored under a prompt before the production version: relabeled so the stored
# labels follow one policy (#245, overriding D040 §5). The pass ends on the judgments a reply
# writes (D038), which carry the production version, so no time bound is kept. The arms run in
# this order: labeled rows first (scorecard and training positives), then skips, then negatives.

ARM_RELABEL_LABELED = "relabel_labeled"  # L: labeled under an earlier prompt
ARM_RELABEL_SKIPPED = "relabel_skipped"  # S: a skip judged under an earlier prompt
ARM_RELABEL_FALSE_POSITIVE = "relabel_false_positive"  # N: a negative judged under an earlier prompt
RELABEL_ARMS = (ARM_RELABEL_LABELED, ARM_RELABEL_SKIPPED, ARM_RELABEL_FALSE_POSITIVE)
RELABEL_STATUSES = ("labeled", "skipped", FP_NEGATIVE_STATUS)

# --- The #248 targeted relabel cohort (D046 §5) ------------------------------------------------
# The two classes v1.14.0 decides differently from what is stored: negatives typed retail_only or
# mention_only (#232's corrections), and skips typed financial with no business or product
# judgment (#248's template and price-only rows). A row is decided in this order: one already
# judged under the production version is not a candidate, then one outside both classes is not,
# and only then does relabel_arm apply #245's exclusions. Negatives run first: the class #232
# measured, and the one #154 PR2 reads.

ARM_248_FALSE_POSITIVE = "relabel_248_false_positive"  # a negative typed retail_only or mention_only
ARM_248_FINANCIAL_SKIP = "relabel_248_financial_skip"  # a skip typed financial, not business/product
RELABEL_248_ARMS = (ARM_248_FALSE_POSITIVE, ARM_248_FINANCIAL_SKIP)
RELABEL_248_STATUSES = (FP_NEGATIVE_STATUS, "skipped")
# A negative is targeted with any of these; a skip with TARGET_SKIP_TYPE and none of the others
TARGET_NEGATIVE_TYPES = frozenset({"retail_only", "mention_only"})
TARGET_SKIP_TYPE = "financial"
SKIP_TYPES_NOT_TARGETED = frozenset({"business", "product"})

# The selections, by the run tag each invocation records
MODE_BACKFILL = "192-backfill"
MODE_RELABEL = "245-relabel"
MODE_RELABEL_248 = "248-relabel"
MODE_ARMS = {MODE_BACKFILL: BACKFILL_ARMS, MODE_RELABEL: RELABEL_ARMS, MODE_RELABEL_248: RELABEL_248_ARMS}
MODE_STATUSES = {
    MODE_BACKFILL: BACKFILL_STATUSES,
    MODE_RELABEL: RELABEL_STATUSES,
    MODE_RELABEL_248: RELABEL_248_STATUSES,
}
# Modes that end at a judgment under the production version, decided by relabel_arm
RELABEL_MODES = frozenset({MODE_RELABEL, MODE_RELABEL_248})

# Unjudged reasons a reply writes, or wrote (SKIP_NO_SPORTSWEAR_BRAND: the pipeline before #200).
# The other two are written before any LLM call, and a relabel would write them again.
LLM_REACHABLE_UNJUDGED_REASONS = UNJUDGED_SKIP_REASONS - {SKIP_INSUFFICIENT_CONTENT, SKIP_NO_BRANDS}

# Why a row is left out of the cohort. Reported, never selected.
EXCLUDED_DETERMINISTIC = "deterministic_skip"  # an unjudged skip a pipeline check writes
EXCLUDED_HUMAN = "human"  # a HUMAN_RULINGS key whatever its status, or a human-sourced label
EXCLUDED_HUMAN_LABELS = "human_reviewed_labels"  # a brand label a human reviewed or wrote
EXCLUDED_POSSIBLE_HAND_MOVE = "possible_hand_move"
EXCLUDED_NO_BRANDS = "no_brands"
EXCLUDED_INSUFFICIENT_CONTENT = "insufficient_content"
EXCLUDED_HAS_JUDGMENTS = "has_judgments"  # a brand_judgments row exists: relabeled, or decided since D038
EXCLUDED_SKIPPED_AFTER_BOUND = "skipped_after_bound"  # skipped at or after the bound (U and J only)
EXCLUDED_NOT_SCRAPED = "not_scraped"  # at a cohort status, but outside the FP export's scrape filter
EXCLUDED_FAILED_IN_BACKFILL = "failed_in_backfill"  # a tagged run wrote this failure (X only)
EXCLUDED_AT_PRODUCTION = "judged_at_production"  # already judged under the production version (#245)
EXCLUDED_NOT_LLM_DECIDED = "not_llm_decided"  # a judged negative whose label source is not the LLM (#245)


def _brand_key(name: str) -> str:
    """``normalize_brand``'s key (``src/labeling/models.py``), restated so this module does not
    load the labeling package."""
    return name.strip().casefold()


def possible_hand_move(labeling_error: str | None, brands: Sequence[str]) -> bool:
    """Whether a ``false_positive`` row's reason is not in the form an LLM reply writes, for a
    brand the article mentions.

    A reply's negative is ``"<brand>: <reason>"`` per brand, joined by ``"; "`` (``dispose_reply``;
    every pipeline version since false-positive detection), so the first part names a brand that
    was sent. Fail-closed: every other reason counts as a possible hand move, since
    ``fix_label.py update`` keeps the reason, or with ``--manual`` stores a human marker, never a
    reply's form. That covers a move from
    ``failed`` (the labeler's error), from ``skipped`` (a skip reason) or from ``deduplicated``,
    a reason a human wrote, none, and an FP classifier skip's reason on a row an LLM did not
    decide. It stands in for #143's audit log, which will record the move itself.
    """
    if not labeling_error:
        return True
    name, separator, _ = labeling_error.split("; ", 1)[0].partition(": ")
    return not separator or _brand_key(name) not in {_brand_key(b) for b in brands}


class BackfillDecision(NamedTuple):
    """A row's arm, or why it is left out. Both None: not a backfill candidate at all."""

    arm: str | None
    excluded: str | None = None


NOT_A_CANDIDATE = BackfillDecision(None)


def as_utc(moment: datetime) -> datetime:
    """A naive timestamp is taken as UTC, so it compares with an aware one."""
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def backfill_arm(
    *,
    article_id: str,
    labeling_status: str,
    labeling_error: str | None,
    source: str,
    skipped_at: datetime | None,
    brands: Sequence[str],
    sufficient_content: bool,
    has_judgments: bool,
    human_labels: bool,
    judged_skip_cutoff: datetime | None,
    bound: datetime,
    tried_reason: str | None = None,
) -> BackfillDecision:
    """The backfill arm of one row, or why it is left out (D039).

    ``source`` is ``derive_label_source``'s, with ``human_verified`` from ``human_verified_ids``
    (None for a ``failed`` row, which has no FP training label).
    ``judged_skip_cutoff`` is the production prompt version's first run (None: no J arm).
    ``bound`` ends the pass: a U or J row skipped at or after it is not selected, so a row the
    backfill skipped again, and a skip the cron writes meanwhile, are not retried. A NULL
    ``skipped_at`` is before every bound.
    ``tried_reason`` is the reason a tagged run recorded for the row before relabeling it (None:
    no tagged run has). It ends arm X's pass, since a ``failed`` write records no timestamp.
    """
    if labeling_status not in BACKFILL_STATUSES:
        return NOT_A_CANDIDATE
    if source == HUMAN or article_id in HUMAN_RULINGS:
        return BackfillDecision(None, EXCLUDED_HUMAN)

    if labeling_status == FAILED_STATUS:
        arm = ARM_FAILED
    elif labeling_status == "skipped" and source == UNJUDGED:
        if labeling_error not in LLM_REACHABLE_UNJUDGED_REASONS:
            return BackfillDecision(None, EXCLUDED_DETERMINISTIC)
        arm = ARM_UNJUDGED
    elif labeling_status == FP_NEGATIVE_STATUS and source in (LLM, PRE_CLASSIFIER):
        arm = ARM_FALSE_POSITIVE
    elif (
        labeling_status == "skipped"
        and source in (LLM, PRE_CLASSIFIER)
        and judged_skip_cutoff is not None
        and skipped_at is not None
        and as_utc(skipped_at) >= as_utc(judged_skip_cutoff)
    ):
        arm = ARM_JUDGED_SKIP
    else:
        return NOT_A_CANDIDATE

    if human_labels:
        return BackfillDecision(None, EXCLUDED_HUMAN_LABELS)
    if not brands:
        return BackfillDecision(None, EXCLUDED_NO_BRANDS)
    # After the brand check: with no brands, no reason can name one
    if arm == ARM_FALSE_POSITIVE and possible_hand_move(labeling_error, brands):
        return BackfillDecision(None, EXCLUDED_POSSIBLE_HAND_MOVE)
    if not sufficient_content:
        return BackfillDecision(None, EXCLUDED_INSUFFICIENT_CONTENT)
    if has_judgments:
        return BackfillDecision(None, EXCLUDED_HAS_JUDGMENTS)
    # F needs no bound: every reply writes judgments or another status, and a row with no reply
    # keeps its status to be retried
    if (
        arm in (ARM_UNJUDGED, ARM_JUDGED_SKIP)
        and skipped_at is not None
        and as_utc(skipped_at) >= as_utc(bound)
    ):
        return BackfillDecision(None, EXCLUDED_SKIPPED_AFTER_BOUND)
    # X: a row a tagged run tried that is now failed with another reason was failed by that run
    # (a reply the labeler could not use), so it is not retried. A row whose status the run kept
    # (a transient failure, a call with no reply, an exception) still has the recorded reason.
    if arm == ARM_FAILED and tried_reason is not None and labeling_error != tried_reason:
        return BackfillDecision(None, EXCLUDED_FAILED_IN_BACKFILL)
    return BackfillDecision(arm)


def relabel_arm(
    *,
    article_id: str,
    labeling_status: str,
    labeling_error: str | None,
    source: str,
    brands: Sequence[str],
    sufficient_content: bool,
    judgment_versions: set[str],
    has_labels: bool,
    human_labels: bool,
    production_version: str,
) -> BackfillDecision:
    """The #245 relabel arm of one row, or why it is left out.

    A row is a candidate when its status is one the LLM decided and it has no judgment under the
    production version: a ``labeled`` row (judged under an earlier version, or before D038 wrote
    judgments), or a ``skipped`` or ``false_positive`` row judged under an earlier version. A
    negative with no judgment at all is #192's backfill's to decide (``backfill_arm``), not this.

    Hand moves, fail-closed as in ``backfill_arm``. A pipeline ``labeled`` write clears the reason
    and stores at least one label, so a ``labeled`` row with a reason or no label may have been
    moved by hand (``fix_label.py update --status labeled`` keeps or clears the reason and writes
    no label). A hand move that cleared the reason onto a row with older LLM labels looks like a
    pipeline write; #143's audit log will record it.
    """
    if labeling_status not in RELABEL_STATUSES:
        return NOT_A_CANDIDATE
    if source == HUMAN or article_id in HUMAN_RULINGS:
        return BackfillDecision(None, EXCLUDED_HUMAN)
    if labeling_status != "labeled" and not judgment_versions:
        return NOT_A_CANDIDATE
    if production_version in judgment_versions:
        return BackfillDecision(None, EXCLUDED_AT_PRODUCTION)

    if labeling_status == "labeled":
        arm = ARM_RELABEL_LABELED
    elif labeling_status == "skipped":
        if source == UNJUDGED and labeling_error not in LLM_REACHABLE_UNJUDGED_REASONS:
            return BackfillDecision(None, EXCLUDED_DETERMINISTIC)
        if source not in (LLM, PRE_CLASSIFIER, UNJUDGED):
            # UNATTRIBUTED: a reason no pipeline writes
            return BackfillDecision(None, EXCLUDED_POSSIBLE_HAND_MOVE)
        arm = ARM_RELABEL_SKIPPED
    else:
        if source not in (LLM, PRE_CLASSIFIER):
            return BackfillDecision(None, EXCLUDED_NOT_LLM_DECIDED)
        arm = ARM_RELABEL_FALSE_POSITIVE

    if human_labels:
        return BackfillDecision(None, EXCLUDED_HUMAN_LABELS)
    if not brands:
        return BackfillDecision(None, EXCLUDED_NO_BRANDS)
    if arm == ARM_RELABEL_LABELED and (labeling_error or not has_labels):
        return BackfillDecision(None, EXCLUDED_POSSIBLE_HAND_MOVE)
    if arm == ARM_RELABEL_FALSE_POSITIVE and possible_hand_move(labeling_error, brands):
        return BackfillDecision(None, EXCLUDED_POSSIBLE_HAND_MOVE)
    if not sufficient_content:
        return BackfillDecision(None, EXCLUDED_INSUFFICIENT_CONTENT)
    return BackfillDecision(arm)


def targeted_248(labeling_status: str, content_types: set[str]) -> bool:
    """Whether a row is in #248's targeted classes by its stored content types (D046 §5).

    A negative with any ``retail_only`` or ``mention_only`` judgment; one with neither is not
    targeted whatever its other types, including a negative moved by hand that kept a sportswear
    type. A skip with a ``financial`` judgment and no ``business`` or ``product`` one; its other
    brands' non-sportswear types do not change that.
    """
    if labeling_status == FP_NEGATIVE_STATUS:
        return bool(content_types & TARGET_NEGATIVE_TYPES)
    if labeling_status == "skipped":
        return TARGET_SKIP_TYPE in content_types and not content_types & SKIP_TYPES_NOT_TARGETED
    return False


# relabel_arm's arm for a targeted row -> the #248 arm
_ARM_248 = {
    ARM_RELABEL_FALSE_POSITIVE: ARM_248_FALSE_POSITIVE,
    ARM_RELABEL_SKIPPED: ARM_248_FINANCIAL_SKIP,
}


def relabel_248_arm(*, content_types: set[str], **relabel_args) -> BackfillDecision:
    """The #248 relabel arm of one row, or why it is left out (D046 §5).

    In order: a row already judged under the production version (by this pass or by the daily run)
    is not a candidate; nor is a row outside the targeted classes. Neither counts toward any
    exclusion, so selected plus excluded is the targeted rows not yet judged under production; the
    pass's progress is the report's, read from the tagged runs' recorded ``prior``. A remaining row
    takes ``relabel_arm``'s decision, every exclusion it has (``human``, ``deterministic_skip``,
    ``possible_hand_move``, ``not_llm_decided``, ``human_reviewed_labels``, ``no_brands``,
    ``insufficient_content``), with its arm renamed.
    """
    if relabel_args["production_version"] in relabel_args["judgment_versions"]:
        return NOT_A_CANDIDATE
    if not targeted_248(relabel_args["labeling_status"], content_types):
        return NOT_A_CANDIDATE
    decision = relabel_arm(**relabel_args)
    if decision.arm is None:
        return decision
    return BackfillDecision(_ARM_248[decision.arm])


@dataclass
class BackfillRow:
    """One row of the FP export query, as the backfill reads it."""

    article_id: str
    labeling_status: str
    labeling_error: str | None
    fp_action: str | None
    fp_skip_reason: str | None
    skipped_at: datetime | None
    created_at: datetime | None
    brands: list[str]
    content_chars: int
    sufficient_content: bool


HAND_MOVE_REASON_CHARS = 40


@dataclass
class BackfillCohort:
    """The selected ids by arm, what each had before the relabel, and what was left out."""

    judged_skip_cutoff: datetime | None
    bound: datetime
    mode: str = MODE_BACKFILL
    # the production version a RELABEL_MODES cohort ends at (None in MODE_BACKFILL)
    production_version: str | None = None
    selected: dict[str, list[str]] = field(default_factory=dict)
    # article id -> [status, reason, arm] before the relabel
    prior: dict[str, list[str | None]] = field(default_factory=dict)
    content_chars: dict[str, int] = field(default_factory=dict)
    excluded: Counter = field(default_factory=Counter)
    # (arm, stored reason) for U and J; (arm, label source) for F, whose reasons are free text;
    # (arm, the reason's first HAND_MOVE_REASON_CHARS) for X, whose errors carry request ids
    by_reason: Counter = field(default_factory=Counter)
    # (arm, UTC year-month the article was collected)
    by_month: Counter = field(default_factory=Counter)
    # selected articles that still carry non-human brand_labels rows
    stale_labels: int = 0
    # article id -> why it was left out, for each row with an exclusion
    left_out: dict[str, str] = field(default_factory=dict)
    # the first HAND_MOVE_REASON_CHARS of each possible hand move's reason, so the owner sees what
    # the fail-closed test left out
    hand_move_reasons: Counter = field(default_factory=Counter)
    # article id -> the reason the first tagged run to relabel it recorded (arm X's pass end)
    tried: dict[str, str | None] = field(default_factory=dict)
    # article id -> its stored content types before the relabel, sorted (MODE_RELABEL_248 only;
    # a relabel replaces the judgments, so this is the record of what selected each row)
    prior_content_types: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        for arm in self.arms:
            self.selected.setdefault(arm, [])

    @property
    def arms(self) -> tuple[str, ...]:
        """The mode's arms, in the order they run."""
        return MODE_ARMS[self.mode]

    @property
    def ids(self) -> list[str]:
        """Every selected id, arm by arm in the mode's order, by id within an arm."""
        return [i for arm in self.arms for i in self.selected[arm]]


def build_backfill_cohort(
    rows: Iterable[BackfillRow],
    *,
    judged_ids: set[str],
    human_label_ids: set[str],
    label_ids: set[str],
    judged_skip_cutoff: datetime | None,
    bound: datetime,
    tried: dict[str, str | None] | None = None,
    mode: str = MODE_BACKFILL,
    judgment_versions: dict[str, set[str]] | None = None,
    production_version: str | None = None,
    content_types: dict[str, set[str]] | None = None,
) -> BackfillCohort:
    """The cohort over rows in id order; ``backfill_arm`` decides each row, ``relabel_arm`` in
    MODE_RELABEL, or ``relabel_248_arm`` in MODE_RELABEL_248. ``tried`` is ``tried_reasons`` over
    the tagged runs so far. ``judgment_versions`` and ``content_types`` map an article id to the
    prompt versions and the content types of its ``brand_judgments`` rows (RELABEL_MODES, and
    MODE_RELABEL_248, only)."""
    if mode in RELABEL_MODES and not production_version:
        raise ValueError("A relabel cohort needs the production version it ends at")
    rows = list(rows)
    tried = dict(tried or {})
    judgment_versions = judgment_versions or {}
    content_types = content_types or {}
    cohort = BackfillCohort(
        judged_skip_cutoff=judged_skip_cutoff,
        bound=bound,
        mode=mode,
        production_version=production_version,
        tried=tried,
    )
    statuses = MODE_STATUSES[mode]
    verified = human_verified_ids((r.article_id, r.labeling_status) for r in rows)
    for r in rows:
        if r.labeling_status not in statuses:
            continue
        source = None if r.labeling_status == FAILED_STATUS else derive_label_source(
            r.labeling_status,
            r.labeling_error,
            r.fp_action,
            r.fp_skip_reason,
            human_verified=r.article_id in verified,
            article_id=r.article_id,
        )
        if mode in RELABEL_MODES:
            decide = relabel_arm
            extra = {}
            if mode == MODE_RELABEL_248:
                decide = relabel_248_arm
                extra = {"content_types": content_types.get(r.article_id, set())}
            decision = decide(
                **extra,
                article_id=r.article_id,
                labeling_status=r.labeling_status,
                labeling_error=r.labeling_error,
                source=source,
                brands=r.brands,
                sufficient_content=r.sufficient_content,
                judgment_versions=judgment_versions.get(r.article_id, set()),
                has_labels=r.article_id in label_ids,
                human_labels=r.article_id in human_label_ids,
                production_version=production_version,
            )
        else:
            decision = backfill_arm(
                article_id=r.article_id,
                labeling_status=r.labeling_status,
                labeling_error=r.labeling_error,
                source=source,
                skipped_at=r.skipped_at,
                brands=r.brands,
                sufficient_content=r.sufficient_content,
                has_judgments=r.article_id in judged_ids,
                human_labels=r.article_id in human_label_ids,
                judged_skip_cutoff=judged_skip_cutoff,
                bound=bound,
                tried_reason=tried.get(r.article_id),
            )
        if decision.excluded:
            cohort.excluded[decision.excluded] += 1
            cohort.left_out[r.article_id] = decision.excluded
        if decision.excluded == EXCLUDED_POSSIBLE_HAND_MOVE:
            cohort.hand_move_reasons[(r.labeling_error or "")[:HAND_MOVE_REASON_CHARS]] += 1
        if decision.arm is None:
            continue
        arm = decision.arm
        cohort.selected[arm].append(r.article_id)
        cohort.prior[r.article_id] = [r.labeling_status, r.labeling_error, arm]
        cohort.content_chars[r.article_id] = r.content_chars
        if mode in RELABEL_MODES:
            versions = judgment_versions.get(r.article_id)
            reason_key = f"{source}, judged under {', '.join(sorted(versions)) if versions else 'no version'}"
            if mode == MODE_RELABEL_248:
                types = sorted(content_types.get(r.article_id, set()))
                cohort.prior_content_types[r.article_id] = types
                reason_key = f"{source}, typed {'/'.join(types)}, judged under {', '.join(sorted(versions or ()))}"
        elif arm == ARM_FALSE_POSITIVE:
            reason_key = source
        elif arm == ARM_FAILED:
            reason_key = (r.labeling_error or "")[:HAND_MOVE_REASON_CHARS]
        else:
            reason_key = r.labeling_error
        cohort.by_reason[(arm, reason_key)] += 1
        month = as_utc(r.created_at).strftime("%Y-%m") if r.created_at else "unknown"
        cohort.by_month[(arm, month)] += 1
        if r.article_id in label_ids:
            cohort.stale_labels += 1
    return cohort


class PriorRecord(NamedTuple):
    """What a tagged run recorded for an article before relabeling it (``run_config["prior"]``)."""

    status: str
    reason: str | None
    arm: str


def first_prior(run_configs: Iterable[dict | None]) -> dict[str, PriorRecord]:
    """Article id -> the first record the tagged runs made of it, from the runs' configs in start
    order. A retried article's first record is its state from before the relabel; a later run's
    record of it is the state an earlier tagged run left."""
    prior: dict[str, PriorRecord] = {}
    for config in run_configs:
        for article_id, values in (((config or {}).get("run_config") or {}).get("prior") or {}).items():
            prior.setdefault(article_id, PriorRecord(*values))
    return prior


def tried_reasons(run_configs: Iterable[dict | None]) -> dict[str, str | None]:
    """Article id -> the reason the first tagged run to relabel it recorded (``first_prior``)."""
    return {article_id: record.reason for article_id, record in first_prior(run_configs).items()}


def _cohort_from_db(
    session,
    *,
    judged_skip_cutoff: datetime | None,
    bound: datetime,
    tried: dict[str, str | None] | None = None,
    ids: list[str] | None = None,
    mode: str = MODE_BACKFILL,
    production_version: str | None = None,
) -> BackfillCohort:
    """``build_backfill_cohort`` over the FP export's rows and the ``failed`` rows, or only
    those with the given ids."""
    # Imported here: the labeling package's __init__ loads the whole pipeline, and the FP export
    # imports this module without needing it
    from src.labeling.article_input import article_content, has_sufficient_content

    query = fp_export_query(session, also_failed=True).options(
        load_only(
            Article.id,
            Article.labeling_status,
            Article.labeling_error,
            Article.skipped_at,
            Article.created_at,
            Article.brands_mentioned,
            Article.full_content,
            Article.description,
        )
    )
    if ids is not None:
        query = query.filter(Article.id.in_(ids))
    rows = []
    for article, fp_action, fp_skip_reason, _ in query.all():
        content = article_content(article.full_content, article.description)
        rows.append(
            BackfillRow(
                article_id=str(article.id),
                labeling_status=article.labeling_status,
                labeling_error=article.labeling_error,
                fp_action=fp_action,
                fp_skip_reason=fp_skip_reason,
                skipped_at=article.skipped_at,
                created_at=article.created_at,
                brands=list(article.brands_mentioned or []),
                content_chars=len(content),
                sufficient_content=has_sufficient_content(content),
            )
        )

    def article_ids(column, *criteria) -> set[str]:
        query = session.query(column).filter(*criteria)
        if ids is not None:
            query = query.filter(column.in_(ids))
        return {str(i) for (i,) in query.distinct()}

    human = human_label_criterion()
    judgment_versions: dict[str, set[str]] = {}
    content_types: dict[str, set[str]] = {}
    if mode in RELABEL_MODES:
        # Distinct (article, version, type) rows: a relabel replaces every judgment row, so an
        # article has one version, but the set does not assume it
        query = session.query(
            BrandJudgment.article_id, BrandJudgment.prompt_version, BrandJudgment.content_type
        ).distinct()
        if ids is not None:
            query = query.filter(BrandJudgment.article_id.in_(ids))
        for article_id, version, content_type in query:
            judgment_versions.setdefault(str(article_id), set()).add(version)
            content_types.setdefault(str(article_id), set()).add(content_type)
    return build_backfill_cohort(
        rows,
        judged_ids=article_ids(BrandJudgment.article_id),
        human_label_ids=article_ids(BrandLabel.article_id, human),
        label_ids=article_ids(BrandLabel.article_id),
        judged_skip_cutoff=judged_skip_cutoff,
        bound=bound,
        tried=tried,
        mode=mode,
        judgment_versions=judgment_versions,
        production_version=production_version,
        content_types=content_types,
    )


def select_backfill_cohort(
    session,
    *,
    judged_skip_cutoff: datetime | None,
    bound: datetime,
    tried: dict[str, str | None] | None = None,
    mode: str = MODE_BACKFILL,
    production_version: str | None = None,
) -> BackfillCohort:
    """The #192 backfill cohort from the database (D039): the FP export's rows and the
    ``failed`` rows (#86), decided by ``backfill_arm``, plus a count of rows at a cohort status
    outside the export's scrape filter. In MODE_RELABEL, the #245 relabel cohort, decided by
    ``relabel_arm``; in MODE_RELABEL_248, #248's targeted cohort, decided by ``relabel_248_arm``.
    """
    cohort = _cohort_from_db(
        session,
        judged_skip_cutoff=judged_skip_cutoff,
        bound=bound,
        tried=tried,
        mode=mode,
        production_version=production_version,
    )
    unscraped = or_(Article.scrape_status.is_(None), Article.scrape_status != "success")
    if mode == MODE_RELABEL_248:
        # Only rows relabel_248_arm would decide count: targeted, and not yet judged under the
        # production version, so selected plus excluded stays the targeted rows not yet judged
        statuses: dict[str, str] = {}
        types: dict[str, set[str]] = {}
        versions: dict[str, set[str]] = {}
        query = (
            session.query(
                Article.id, Article.labeling_status, BrandJudgment.content_type, BrandJudgment.prompt_version
            )
            .join(BrandJudgment, BrandJudgment.article_id == Article.id)
            .filter(Article.labeling_status.in_(MODE_STATUSES[mode]))
            .filter(unscraped)
        )
        for article_id, status, content_type, version in query:
            statuses[str(article_id)] = status
            types.setdefault(str(article_id), set()).add(content_type)
            versions.setdefault(str(article_id), set()).add(version)
        not_scraped = sum(
            1
            for i, status in statuses.items()
            if production_version not in versions[i] and targeted_248(status, types[i])
        )
    else:
        not_scraped = (
            session.query(func.count(Article.id))
            .filter(Article.labeling_status.in_(MODE_STATUSES[mode]))
            .filter(unscraped)
            .scalar()
        )
    if not_scraped:
        cohort.excluded[EXCLUDED_NOT_SCRAPED] += not_scraped
    return cohort


def recheck_backfill_batch(
    session, cohort: BackfillCohort, ids: list[str]
) -> tuple[list[str], dict[str, str]]:
    """The batch's ids still selected as they were, and why each other one was dropped.

    The cohort is selected once per invocation, which can run for hours, so each batch is decided
    again just before it is labeled, with the same cutoff and bound. A row whose arm, status or
    reason is not what the cohort recorded is dropped, so a human's move during the run is not
    overwritten. Why: its exclusion, or ``changed``.
    """
    fresh = _cohort_from_db(
        session,
        judged_skip_cutoff=cohort.judged_skip_cutoff,
        bound=cohort.bound,
        tried=cohort.tried,
        ids=ids,
        mode=cohort.mode,
        production_version=cohort.production_version,
    )
    kept, dropped = [], {}
    for i in ids:
        if fresh.prior.get(i) == cohort.prior[i]:
            kept.append(i)
        else:
            dropped[i] = fresh.left_out.get(i, "changed")
    return kept, dropped


# --- Spot-checked strata the FP export leaves out (#154 PR2, #289) ----------------------------
# A relabel stratum whose seeded spot-check measured an error rate in the exclude band. A stratum
# is the rows one relabel pass moved from one status to another: the runs with its tag that
# started before the sample was drawn, so a later invocation of the same mode adds no row to it.
# A row stays in the stratum only while it holds that pass's verdict: the status it moved to, the
# reason the reply wrote, and brand judgments from the pass's runs alone. A status move (a human
# ruling, a fix_label.py move) or a later run's judgment takes it out, and the export reads it as
# any other row. Its label_source is unchanged, since the LLM decided it: the exclusion is a
# second, independent test. Reviewed rows are human rulings, which the export keeps.


@dataclass(frozen=True)
class SpotCheckStratum:
    name: str  # reported by the export and stamped on a row kept with --include-spot-check-strata
    record: str  # where the spot-check and its decision are recorded, printed beside the name
    run_tag: str  # the relabel mode whose tagged runs recorded the rows' prior status
    runs_before: datetime  # the pass: tagged runs that started before this (UTC)
    prior_status: str  # the status the pass recorded before relabeling
    status: str  # the status the pass moved them to
    reasons: frozenset[str]  # the stored reasons a reply writes at that status


EXCLUDED_STRATA = (
    # #289: 4 of 25 sampled rows wrong (16%, Clopper-Pearson 95% interval 4.5-36.1%), at or above
    # the 15% exclude band. The pass: 56 runs, 2026-11-22 08:38-10:02 UTC under v1.14.0, before
    # #289 drew its sample that day. 152 unreviewed rows on 2026-11-25.
    SpotCheckStratum(
        name="248-relabel false_positive -> skipped",
        record="#289",
        run_tag=MODE_RELABEL_248,
        runs_before=datetime(2026, 11, 23, tzinfo=timezone.utc),
        prior_status=FP_NEGATIVE_STATUS,
        status="skipped",
        reasons=JUDGED_SKIP_REASONS,
    ),
)


class StratumRow(NamedTuple):
    """An article's current state, as stratum membership reads it."""

    status: str
    reason: str | None
    judgment_runs: frozenset  # labeling_run_id of each of its brand_judgments rows


def stratum_members(
    stratum: SpotCheckStratum,
    prior: dict[str, PriorRecord],
    current: dict[str, StratumRow],
    pass_run_ids: set,
) -> set[str]:
    """Ids still holding the pass's verdict: the pass recorded ``stratum.prior_status`` first, and
    the row is at ``stratum.status`` with one of ``stratum.reasons`` and judged by the pass's runs
    alone (``pass_run_ids``).

    A judged reply writes a judgment for each brand it judged, and a later reply replaces them
    (``save_brand_judgments``), so a row with none, or with one from another run, was decided by
    something else since. A row at an unjudged reason is not a member: its source already
    excludes it from the export by default.
    """
    members = set()
    for article_id, record in prior.items():
        row = current.get(article_id)
        if (
            record.status == stratum.prior_status
            and row is not None
            and row.status == stratum.status
            and row.reason in stratum.reasons
            and row.judgment_runs
            and row.judgment_runs <= pass_run_ids
        ):
            members.add(article_id)
    return members


def excluded_stratum_ids(session, strata: Sequence[SpotCheckStratum] = EXCLUDED_STRATA) -> dict[str, str]:
    """Article id -> the name of the excluded stratum it is in, read from the database."""
    members: dict[str, str] = {}
    for stratum in strata:
        runs = (
            session.query(LabelingRun.id, LabelingRun.config)
            .filter(LabelingRun.config["run_tag"].as_string() == stratum.run_tag)
            .filter(LabelingRun.started_at < stratum.runs_before)
            .order_by(LabelingRun.started_at)
            .all()
        )
        prior = first_prior(config for _, config in runs)
        candidates = [UUID(i) for i, record in prior.items() if record.status == stratum.prior_status]
        if not candidates:
            continue
        state: dict[str, tuple[str, str | None, set]] = {}
        for article_id, status, reason, run_id in (
            session.query(
                Article.id, Article.labeling_status, Article.labeling_error, BrandJudgment.labeling_run_id
            )
            .outerjoin(BrandJudgment, BrandJudgment.article_id == Article.id)
            .filter(Article.id.in_(candidates))
            .all()
        ):
            entry = state.setdefault(str(article_id), (status, reason, set()))
            if run_id is not None:
                entry[2].add(run_id)
        current = {i: StratumRow(st, reason, frozenset(runs_)) for i, (st, reason, runs_) in state.items()}
        for article_id in stratum_members(stratum, prior, current, {run_id for run_id, _ in runs}):
            members.setdefault(article_id, stratum.name)
    return members


# --- #227, #273: stored text that names none of the article's brands outside a cookie notice ---
# (D044, D047). A row scraped as 'success' whose full_content names none of its brands_mentioned
# outside a cookie notice was labeled, or queued for labeling, with content that carries nothing to
# label the brand from. scripts/triage_no_brand_content.py moves the rows this selects to the state
# the scraper now gives such a scrape ('skipped', SCRAPE_NO_BRAND_ERROR, 'unlabelable').

NO_BRAND_STATUSES = ("pending", FAILED_STATUS, "labeled", "skipped", FP_NEGATIVE_STATUS)
EXCLUDED_OTHER_STATUS = "other_status"  # a status the triage does not move (deduplicated, ...)
EXCLUDED_EVAL_CASE = "eval_case"  # an article in an eval case file, whose expected verdict is the owner's


@dataclass
class NoBrandRow:
    """One stored row whose text names none of its brands outside a cookie notice, as the triage
    decides it."""

    article_id: str
    labeling_status: str
    labeling_error: str | None
    fp_action: str | None
    fp_skip_reason: str | None
    brands: list[str]
    has_labels: bool
    human_labels: bool


def no_brand_exclusion(row: NoBrandRow) -> str | None:
    """Why the triage leaves this row as stored, or None to move it.

    Fail-closed, as the backfill cohorts are. Left for the owner: a ``HUMAN_RULINGS`` row, a human
    marker as the reason (``FP_REASON_MANUAL``, ``MANUAL_SKIP_REASON``) whatever the status, a
    human-written or reviewed label, and a row that may have been moved by hand:
    - ``false_positive`` the LLM decided, unless its reason has a reply's form
      (``possible_hand_move``);
    - ``skipped`` with a reason no pipeline writes (``relabel_arm``);
    - ``labeled`` with a reason or no label (``relabel_arm``);
    - ``pending`` with a reason or a label: the pipeline writes neither before a ``pending`` row is
      labeled, so ``fix_label.py update --status pending`` put it there;
    - ``failed`` with a label.
    Moved: a ``false_positive`` the LLM decided whose reason has a reply's form, one the FP
    classifier decided (its stored reason is the classifier's own) or one the keyword sweep wrote
    (``rule``); a ``skipped`` row at a reason the pipeline writes; a ``labeled`` row with labels and
    no reason; a ``pending`` row with no reason and no label; and a ``failed`` row with no label.
    A status outside ``NO_BRAND_STATUSES`` is left as stored.
    """
    if row.article_id in HUMAN_RULINGS:
        return EXCLUDED_HUMAN
    if row.labeling_error in (FP_REASON_MANUAL, MANUAL_SKIP_REASON):
        return EXCLUDED_HUMAN
    if row.labeling_status not in NO_BRAND_STATUSES:
        return EXCLUDED_OTHER_STATUS
    if row.human_labels:
        return EXCLUDED_HUMAN_LABELS
    if row.labeling_status in (FP_NEGATIVE_STATUS, "skipped"):
        source = derive_label_source(
            row.labeling_status,
            row.labeling_error,
            row.fp_action,
            row.fp_skip_reason,
            article_id=row.article_id,
        )
        if source == HUMAN:
            return EXCLUDED_HUMAN
        if row.labeling_status == FP_NEGATIVE_STATUS:
            if source in (LLM, PRE_CLASSIFIER) and possible_hand_move(row.labeling_error, row.brands):
                return EXCLUDED_POSSIBLE_HAND_MOVE
        elif source == UNATTRIBUTED:
            return EXCLUDED_POSSIBLE_HAND_MOVE
    if row.labeling_status == "labeled" and (row.labeling_error or not row.has_labels):
        return EXCLUDED_POSSIBLE_HAND_MOVE
    if row.labeling_status == "pending" and (row.labeling_error or row.has_labels):
        return EXCLUDED_POSSIBLE_HAND_MOVE
    if row.labeling_status == FAILED_STATUS and row.has_labels:
        return EXCLUDED_POSSIBLE_HAND_MOVE
    return None


@dataclass
class NoBrandCohort:
    """The rows to move, each with the state it had at selection, and what was left out."""

    # article id -> (labeling_status, labeling_error, scrape_status, scrape_error) at selection
    selected: dict[str, tuple[str, str | None, str | None, str | None]] = field(default_factory=dict)
    brands: dict[str, list[str]] = field(default_factory=dict)
    excluded: Counter = field(default_factory=Counter)
    # article id -> why it was left out
    left_out: dict[str, str] = field(default_factory=dict)

    @property
    def by_status(self) -> Counter:
        return Counter(state[0] for state in self.selected.values())

    @property
    def by_brand(self) -> Counter:
        return Counter(b for i in self.selected for b in self.brands[i])


def select_no_brand_content(session, eval_case_ids: Iterable[str] = ()) -> NoBrandCohort:
    """Rows scraped as 'success' whose stored ``full_content`` names none of their brands outside a
    cookie notice, by the scraper's own test (``brands_in_article_text``, #273), not already
    ``unlabelable``, each decided by ``no_brand_exclusion``; a row it would move whose id is in
    ``eval_case_ids`` is left out as ``EXCLUDED_EVAL_CASE``.
    Derived on every call, in id order."""
    eval_case_ids = set(eval_case_ids)
    from .brand_match import brands_in_article_text

    latest = latest_prediction_subquery("fp")
    query = (
        session.query(Article, latest.c.action_taken, latest.c.skip_reason)
        .outerjoin(latest, latest.c.article_id == Article.id)
        .filter(Article.scrape_status == "success")
        .filter(Article.full_content.isnot(None))
        .filter(func.cardinality(Article.brands_mentioned) > 0)
        .filter(Article.labeling_status != "unlabelable")
        .options(
            load_only(
                Article.id,
                Article.labeling_status,
                Article.labeling_error,
                Article.scrape_status,
                Article.scrape_error,
                Article.brands_mentioned,
                Article.full_content,
            )
        )
        .order_by(Article.id)
    )
    candidates = []
    for article, fp_action, fp_skip_reason in query.all():
        brands = list(article.brands_mentioned or [])
        if not brands_in_article_text(article.full_content, brands):
            candidates.append((article, fp_action, fp_skip_reason, brands))

    ids = [article.id for article, *_ in candidates]
    label_ids = {str(i) for (i,) in session.query(BrandLabel.article_id).filter(
        BrandLabel.article_id.in_(ids)).distinct()} if ids else set()
    human_label_ids = {str(i) for (i,) in session.query(BrandLabel.article_id).filter(
        BrandLabel.article_id.in_(ids), human_label_criterion()).distinct()} if ids else set()

    cohort = NoBrandCohort()
    for article, fp_action, fp_skip_reason, brands in candidates:
        article_id = str(article.id)
        why = no_brand_exclusion(NoBrandRow(
            article_id=article_id,
            labeling_status=article.labeling_status,
            labeling_error=article.labeling_error,
            fp_action=fp_action,
            fp_skip_reason=fp_skip_reason,
            brands=brands,
            has_labels=article_id in label_ids,
            human_labels=article_id in human_label_ids,
        ))
        if why is None and article_id in eval_case_ids:
            why = EXCLUDED_EVAL_CASE
        if why:
            cohort.excluded[why] += 1
            cohort.left_out[article_id] = why
            continue
        cohort.selected[article_id] = (
            article.labeling_status,
            article.labeling_error,
            article.scrape_status,
            article.scrape_error,
        )
        cohort.brands[article_id] = brands
    return cohort
