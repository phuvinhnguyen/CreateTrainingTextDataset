# CREATE TRAINING DATASET FROM TXT FILE

## How to install

```bash
pip install "git+https://github.com/phuvinhnguyen/CreateTrainingTextDataset.git"
```

## How to use

```python
from processTrainingText.maxilen import from_text_file, from_text_with_tokenizer
from transformers import AutoTokenizer

text_file = '/path/to/text/file'
tokenizer = AutoTokenizer.from_pretrained('gpt2')

dataset = from_text_file(tokenizer, text_file)

print(dataset)
```

```python
from processTrainingText.maxilen import from_text_with_tokenizer

text_file = '/path/to/text/file'

dataset = from_text_with_tokenizer('gpt2', text_file)

print(dataset)
```

