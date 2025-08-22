music-recommender-thesis/
│
├── .gitignore
├── README.md
├── requirements.txt          # Python deps (backend + experiments)
├── package.json              # Vue frontend deps
│
├── docs/                     # All thesis-related documentation
│   ├── proposal.md
│   ├── abstract.md
│   ├── notes/
│   └── figures/
│
├── data/                     # Raw & processed datasets (gitignored)
│   ├── raw/                  # original mp3/wav (ignored by git)
│   ├── processed/            # spectrograms, numpy (ignored)
│   └── metadata/             # CSV/JSON metadata (can keep small samples in git)
│
├── experiments/              # ML training/evaluation code
│   ├── notebooks/            # Jupyter notebooks for exploration
│   ├── models/               # CNN, CRNN, etc.
│   ├── utils/                # feature extraction, preprocessing
│   ├── train.py              # training entrypoint
│   └── evaluate.py           # evaluation script
│
├── backend/                  # FastAPI app
│   ├── app/
│   │   ├── main.py
│   │   ├── api/              # endpoints
│   │   ├── db/               # database models, config
│   │   ├── recommender/      # wrapper around FAISS / re-ranking
│   │   └── schemas/          # Pydantic schemas
│   ├── tests/                # unit tests for backend
│   └── Dockerfile
│
├── frontend/                 # Vue demo app
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── assets/
│   ├── public/
│   └── Dockerfile
│
├── outputs/                  # model checkpoints, logs (ignored)
│   ├── logs/
│   ├── checkpoints/
│   └── results/
│
└── scripts/                  # small utility scripts (data import, migrations)
    ├── import_metadata.py
    ├── build_index.py
    └── run_user_study.py
