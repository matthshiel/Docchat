import readline
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import spacy

from dotenv import load_dotenv
load_dotenv()

def llm(messages, temperature=1):
    '''
    This function is my interface for calling the LLM.
    The messages argument should be a list of dictionaries.

    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user', 'content': 'What is the capital of France?'},
    ...     ], temperature=0)
    'The capital of France is Paris!'
    '''
    import groq
    client = groq.Groq()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content



def chunk_text_by_words(text, max_words=5, overlap=2):
    """
    Splits text into overlapping chunks by word count.

    Examples:
        >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day and the birds were singing."
        >>> chunks = chunk_text_by_words(text, max_words=5, overlap=2)
        >>> len(chunks)
        7
        >>> chunks[0]
        'The quick brown fox jumps'
        >>> chunks[1]
        'fox jumps over the lazy'
        >>> chunks[4]
        'sunny day and the birds'
        >>> chunks[-1]
        'singing.'
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks




def load_spacy_model(language: str):
    """
    Loads a spaCy model for the specified language.
    Args:
        language (str): The language for which to load the spaCy model.

    Returns:
        spacy.Language: The loaded spaCy language model.

    Examples:
        >>> model = load_spacy_model('english')
        >>> isinstance(model, spacy.Language)
        True

        >>> model = load_spacy_model('french')
        >>> isinstance(model, spacy.Language)
        True

        >>> load_spacy_model('unsupported_language')
        Traceback (most recent call last):
        ...
        ValueError: Unsupported language: unsupported_language
    """
    LANGUAGE_MODELS = {
        'french': 'fr_core_news_sm',
        'german': 'de_core_news_sm',
        'spanish': 'es_core_news_sm',
        'english': 'en_core_web_sm',
    }

    if language not in LANGUAGE_MODELS:
        raise ValueError(f"Unsupported language: {language}")

    return spacy.load(LANGUAGE_MODELS[language])


def score_chunk(chunk: str, query: str, language: str = "french") -> float:
    """
    Scores a chunk against a user query using Jaccard similarity of lemmatized word sets
    with stopword removal, using spaCy for multilingual support.

    Examples (French):
        >>> round(score_chunk("Le soleil est brillant et chaud.", "Quelle est la température du soleil ?", language="french"), 2)
        0.25
        >>> round(score_chunk("La voiture rouge roule rapidement.", "Quelle est la couleur de la voiture ?", language="french"), 2)
        0.2
        >>> score_chunk("Les bananes sont jaunes.", "Comment fonctionnent les avions ?", language="french")
        0.0

    Examples (Spanish):
        >>> round(score_chunk("El sol es brillante y caliente.", "¿Qué temperatura tiene el sol?", language="spanish"), 2)
        0.25
        >>> round(score_chunk("El coche rojo va muy rápido.", "¿De qué color es el coche?", language="spanish"), 2)
        0.25
        >>> score_chunk("Los plátanos son amarillos.", "¿Cómo vuelan los aviones?", language="spanish")
        0.0

    Examples (English):
        >>> round(score_chunk("The sun is bright and hot.", "How hot is the sun?", language="english"), 2)
        0.67
        >>> round(score_chunk("The red car is speeding down the road.", "What color is the car?", language="english"), 2)
        0.2
        >>> score_chunk("Bananas are yellow.", "How do airplanes fly?", language="english")
        0.0
    """
    nlp = load_spacy_model(language)

    def preprocess(text):
        doc = nlp(text.lower())
        return set(
            token.lemma_ for token in doc
            if token.is_alpha and not token.is_stop
        )

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return len(intersection) / len(union)


def load_text(filepath_or_url):
    """
    Loads text from a file path or URL. Supports text, HTML, and PDF files.

    Args:
        filepath_or_url (str): The file path or URL to load text from.

    Returns:
        str: The extracted text content.

    Examples:
        >>> load_text('doctests/example.txt')  # Assuming example.txt contains 'Hello World'
        'Hello World'
    """
    if filepath_or_url.startswith(('http://', 'https://')):
        response = requests.get(filepath_or_url)
        content_type = response.headers.get('Content-Type', '')

        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
        elif 'application/pdf' in content_type or filepath_or_url.endswith('.pdf'):
            print(f"Detected PDF file. Content-Type: {content_type}")  # Debugging information
            try:
                # Save the PDF temporarily
                with open('temp.pdf', 'wb') as temp_pdf:
                    temp_pdf.write(response.content)

                # Validate if the file is a valid PDF
                if not response.content.startswith(b'%PDF'):
                    raise ValueError("The file does not appear to be a valid PDF.")

                # Read the PDF using PdfReader
                reader = PdfReader('temp.pdf')
                text = "\n".join(page.extract_text() for page in reader.pages)

                # Clean up the temporary file
                os.remove('temp.pdf')
                return text
            except Exception as pdf_error:
                print(f"Error processing PDF file: {pdf_error}")
                raise ValueError("Failed to process PDF file.")
        else:
            print(f"Unsupported Content-Type or file extension: {content_type}")  # Debugging information
            raise ValueError("Unsupported URL content type or file extension.")
    elif filepath_or_url.endswith('.pdf'):
        reader = PdfReader(filepath_or_url)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif filepath_or_url.endswith('.txt'):
        with open(filepath_or_url, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise ValueError("Unsupported file type")


def find_relevant_chunks(text, query, num_chunks=5, max_words=100, overlap=50):
    """
    Finds the most relevant chunks of text based on a query.

    Args:
        text (str): The input text to split into chunks.
        query (str): The query to score relevance against.
        num_chunks (int): The number of top chunks to return.
        max_words (int): Maximum words per chunk.
        overlap (int): Overlap between chunks.

    Returns:
        list: The top `num_chunks` most relevant chunks.

    Examples:
        >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day."
        >>> query = "What is the weather like?"
        >>> find_relevant_chunks(text, query, num_chunks=1)
        ['The quick brown fox jumps over the lazy dog. It was a sunny day.']
    """
    chunks = chunk_text_by_words(text, max_words=max_words, overlap=overlap)
    scored_chunks = [(chunk, score_chunk(chunk, query)) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:num_chunks]]





if __name__ == '__main__':
    messages = []
    messages.append({
        'role': 'system',
        'content': 'You are a helpful assistant. You are trying to be accurate. You can answer up to 4 sentences.'
    })
    while True:
        # Get input from the user
        text = input('docchat> ')
        
        # Check if the input is a file path
        if text.startswith("file:"):
            file_path = text[5:].strip()  # Extract the file path
            try:
                # Load the document
                document_text = load_text(file_path)
                print("Document loaded successfully.")
                
                # Chunk the document using chunk_text_by_words
                chunks = chunk_text_by_words(document_text, max_words=100, overlap=50)

                # Find the most relevant chunks
                query = "Summarize this document"
                relevant_chunks = find_relevant_chunks(document_text, query, num_chunks=5)

                # Add the relevant chunks to the messages list
                for chunk in relevant_chunks:
                    messages.append({
                        'role': 'user',
                        'content': f"Relevant Chunk: {chunk}",
                    })

                print("Relevant chunks identified and added to the conversation.")
            except Exception as e:
                print(f"Error processing document: {e}")
            continue
        
        # Pass the input to the LLM
        messages.append({
            'role': 'user',
            'content': text,
        })
        result = llm(messages)
        
        # Add the "assistant" role to the messages list
        messages.append({
            'role': 'assistant',
            'content': result,
        })

        # Print the LLM's response to the user
        print('result=', result)
        import pprint
        pprint.pprint(messages)










"""
if __name__ == '__main__':
    messages = []
    messages.append({
        'role': 'system',
        'content': 'You are a helpful assistant.  You are trying to be accurate.  You can answer up to 4 sentences. The first batch of text you receive is a document. Use the docuemnt w=to help answer the questions'
    })
    import argparse
    parser = argparse.ArgumentParser(
        prog='docsum',
        description='summarize the input document',
    )
    parser.add_argument('filename')
    args = parser.parse_args()
    while True:
        # get input from the user
        text = input('docchat> ')
        # pass that input to llm
        messages.append({
            'role': 'user',
            'content': text,
        })
        result = llm(messages)
        # Add the "assistant" role to the messages list
        messages.append({
            'role': 'assistant',
            'content': result,
        })

        # print the llm's response to the user
        print('result=', result)
        import pprint
        pprint.pprint(messages)
"""