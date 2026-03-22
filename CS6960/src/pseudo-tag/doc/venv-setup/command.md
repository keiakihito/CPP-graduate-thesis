# Make sure python 3.11!

rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install "setuptools<70" wheel # For avoid conflict Music2Emo

pip install -r requirements.txt
cd Music2Emo
pip install -r requirements.txt

# Check
python -c "import torch, transformers, librosa, sklearn, soundfile"
python -c "import hydra, pytorch_lightning, pretty_midi, music21, gradio"

## Smaple output only warnings
~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag/Music2Emotion main* 55s
.venv ❯ python -c "import torch, transformers, librosa, sklearn, soundfile"

/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag/.venv/lib/python3.11/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.2.0)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(

~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag/Music2Emotion main* 48s
.venv ❯ python -c "import hydra, pytorch_lightning, pretty_midi, music21, gradio"

/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag/.venv/lib/python3.11/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.2.0)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(


# Run test
 pytest tests/test_embedding_extractors_smoke.py -m smoke -v -s   
