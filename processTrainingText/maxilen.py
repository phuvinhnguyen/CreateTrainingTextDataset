from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def from_text_file(tokenizer,
                   file_path: str,
                   max_len: int = 512,
                   num_proc: int = 8,
                   ):
    text_dataset = load_dataset("text", data_files=file_path, split='train')

    column_names = list(text_dataset.column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = text_dataset.map(
        lambda x: tokenizer(x[text_column_name]),
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc,
    )

    temp_input_ids = []
    the_dataset = {
        'input_ids': [],
        'attention_mask': [],
    }

    for i in tqdm(tokenized_datasets):
        temp_input_ids += i['input_ids']

        while len(temp_input_ids) >= max_len:
            tmp_input_ids = temp_input_ids[0:max_len]
            temp_input_ids = temp_input_ids[max_len:]

            the_dataset['input_ids'].append(tmp_input_ids)
            the_dataset['attention_mask'].append([1]*len(tmp_input_ids))

    tmp_input_ids = temp_input_ids[0:512]
    the_dataset['input_ids'].append(tmp_input_ids)
    the_dataset['attention_mask'].append([1]*len(tmp_input_ids))

    return Dataset.from_dict(the_dataset)


def from_text_with_tokenizer(tokenizer_huggingface: str,
                             file_path: str,
                             max_len: int = 512
                             ):
    return from_text_file(AutoTokenizer.from_pretrained(tokenizer_huggingface),
                          file_path=file_path,
                          max_len=max_len
                          )
