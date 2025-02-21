from markitdown import MarkItDown

def convert_to_md(file_path: str):
    md = MarkItDown()
    result = md.convert(file_path)

    return result.text_content


import tiktoken

def count_tokens(string: str) -> int:
    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(string))
    print(f"Number of tokens: {num_tokens}")
    return num_tokens


def divide_chunks(string, max_tokens_per_chunk = 100000):
    #divide if string is too long
    tokens = count_tokens(string)
    if tokens > max_tokens_per_chunk:
        chunks = []
        chunk = ""
        for line in string.split("\n"):
            if count_tokens(chunk + line) > max_tokens_per_chunk:
                chunks.append(chunk)
                chunk = ""
            chunk += line + "\n"
        chunks.append(chunk)
        return chunks
    else:
        return [string]
    
