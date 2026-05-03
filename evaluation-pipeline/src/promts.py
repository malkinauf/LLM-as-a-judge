#


from pathlib import Path

def load_promt_template(promt_file: str)-> str:
    """
    Load a promt template from text file.
    The template schould contain placeholder like:
    {question}
    {answer}
    """
    promt_path = Path(promt_file)
    if not promt_path.exists():
        raise FileNotFoundError(f"Promt file not found: {promt_path}")
    return promt_path.read_text(encoding="utf-8")


def build_prompt(question: str, answer:str , promt_template:str) ->str:
    return promt_template.format(question=question, answer=answer)

