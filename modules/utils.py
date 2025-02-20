from markitdown import MarkItDown

def convert_to_md(file_path: str):
    md = MarkItDown()
    result = md.convert(file_path)

    return result.text_content
