# Full-source dataset inspection

These are original source-dataset counts, not pilot evaluation counts. Pilot limits and sampled paths are recorded in logs/training_subsets.json and logs/evaluation_subsets.json.


## Cross-domain data inspection

{
  "dataset": "In-The-Wild",
  "repo": "mueller91/In-The-Wild",
  "revision": "eee168f92c367f8c82ff2cf42b6f61e362fd6211",
  "archive_sha256": "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a",
  "columns": [
    "file",
    "speaker",
    "label"
  ],
  "duration_hours": 37.85055604166685,
  "license": "cc-by-sa-4.0",
  "files": 31779,
  "labels": {
    "1": 11816,
    "0": 19963
  },
  "label_mapping": {
    "bona-fide": 0,
    "spoof": 1
  },
  "mapping_decision": "Observed bona-fide maps to 0 and spoof maps to 1; no other labels accepted."
}

## Cross-domain data inspection

{
  "dataset": "ML-ITW",
  "repo": "nccm2p2/ML-ITW",
  "revision": "c4a69001e61acee4ea77404ee3d7406c9a377c5f",
  "original_source": "chlo1/ML-ITW",
  "original_commit": "13037020c09d93814ba13715b1ca085f9ba25c9e",
  "original_hash_match": true,
  "parquet_files": {
    "data/eval-00000-of-00004.parquet": "a138b6a97597c260b18bfc8c223cbc0fe65ff34f422e52527fc1772a28c91efb",
    "data/eval-00001-of-00004.parquet": "838422c0d9489d96a166609e79e47dc24d57531a4584704074ff65b042eede65",
    "data/eval-00002-of-00004.parquet": "d3ad4e19d777366007a092f848c31bb339712181d774e13be3b812b62b06b58a",
    "data/eval-00003-of-00004.parquet": "b338416488dcb8edeaddef9f0d904ea087de6b755c01fd9bb8e8d4735f053f74"
  },
  "schemas": [
    "file: struct<bytes: binary, path: string>\n  child 0, bytes: binary\n  child 1, path: string\nspeaker: string\nlanguage: string\nplatform: string\nlabel: string\n-- schema metadata --\nhuggingface: '{\"info\": {\"features\": {\"file\": {\"_type\": \"Audio\"}, \"speaker' + 193",
    "file: struct<bytes: binary, path: string>\n  child 0, bytes: binary\n  child 1, path: string\nspeaker: string\nlanguage: string\nplatform: string\nlabel: string\n-- schema metadata --\nhuggingface: '{\"info\": {\"features\": {\"file\": {\"_type\": \"Audio\"}, \"speaker' + 193",
    "file: struct<bytes: binary, path: string>\n  child 0, bytes: binary\n  child 1, path: string\nspeaker: string\nlanguage: string\nplatform: string\nlabel: string\n-- schema metadata --\nhuggingface: '{\"info\": {\"features\": {\"file\": {\"_type\": \"Audio\"}, \"speaker' + 193",
    "file: struct<bytes: binary, path: string>\n  child 0, bytes: binary\n  child 1, path: string\nspeaker: string\nlanguage: string\nplatform: string\nlabel: string\n-- schema metadata --\nhuggingface: '{\"info\": {\"features\": {\"file\": {\"_type\": \"Audio\"}, \"speaker' + 193"
  ],
  "observed_labels": {
    "spoof": 6361,
    "bona-fide": 18168
  },
  "languages": {
    "english": 8529,
    "chinese": 8473,
    "japanese": 1569,
    "german": 1060,
    "hindi": 650,
    "portuguese": 1019,
    "korean": 928,
    "russian": 373,
    "spanish": 646,
    "hungarian": 299,
    "french": 256,
    "turkish": 588,
    "ukrainian": 90,
    "hebrew": 49
  },
  "duration_hours": 28.385892100694427,
  "license": "not declared in dataset card",
  "files": 24529,
  "labels": {
    "1": 6361,
    "0": 18168
  },
  "label_mapping": {
    "bona-fide": 0,
    "spoof": 1
  },
  "mapping_decision": "Observed bona-fide maps to 0 and spoof maps to 1; no other labels accepted."
}
