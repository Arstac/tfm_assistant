from tika import parser 
def convert_to_md(file_path: str):
    result = parser.from_file(file_path)
    return result['content']