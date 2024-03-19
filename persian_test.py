import spacy
import PyPDF2

# Load the spaCy model for Persian language
nlp = spacy.blank("fa")

# Add the "sentencizer" component to the spaCy pipeline
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def extract_text_from_file(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    return text


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def split_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


from transformers import BertTokenizer
# from transformers import AutoTokenizer,AutoConfig

def tokenize_sentences(sentences):
    # model_name_or_path = "HooshvareLab/bert-fa-zwnj-base"
    model_name_or_path = "HooshvareLab/bert-fa-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    # config = AutoConfig.from_pretrained(model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_sentences = []
    for sentence in sentences:
        encoded_input = tokenizer(sentence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"].squeeze(0))
        tokenized_sentences.append(tokens)
    return tokenized_sentences


def process_persian_text(file_path, is_pdf=False):
    """
    Extracts text from a file, splits into sentences, and tokenizes them.

    Args:
        file_path: Path to the text file or PDF file.
        is_pdf (bool, optional): Whether the file is a PDF. Defaults to False.

    Returns:
        list: A list of lists containing tokens for each sentence.
    """

    if is_pdf:
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_file(file_path)

    sentences = split_sentences(text)
    tokenized_sentences = tokenize_sentences(sentences)

    return sentences ,tokenized_sentences


# Example usage
file_path = r"D:\MyPythpnProjects\localGPT\localgpt\SOURCE_DOCUMENTS\Train_Strategy.pdf"  # Replace with your file path
is_pdf = True  # Set to True if it's a PDF

sents, tokenized_sentences = process_persian_text(file_path, is_pdf)

for sentence, tokens in zip(sents, tokenized_sentences):
    print(f"Sentence: {sentence}")
    print(f"Tokens: {tokens}\n")
