import re


def escape(str):
    return repr(str)


def unescape(str):
    return eval(str)


def clear_str(str):
    str = re.sub(r'\[[^\]]+\]', '', str)  # [*] 제외
    str = re.sub(r'\([^\)]+\)', '', str)  # (*) 제거
    str = re.sub(r'\<[^\>]+\>', '', str)  # <*> 제거
    str = re.sub(r'[^\u0020-\u007E\uAC00-\uD7AF]', ' ', str)
    str = re.sub(r'\s{2,}', ' ', str)
    return str


def doc_to_sentences(doc):
    '''
        Args
            doc: string
        Return: [n_sentences]
    '''
    brackets = {
        '(': ')',
        '[': ']',
        '<': '>',
        '"': '"”“',
        '“': '"”“',
        '”': '"”“',
        "'": "'’‘",
        "’": "'’‘",
        "‘": "'’‘"}
    EOS = ['.', '!', '?', '\n']

    sentences = []
    sentence = ''
    needed_brackets = []
    for ch in doc:
        if len(needed_brackets):
            if ch in needed_brackets:
                needed_brackets = []
            sentence += ch
        else:
            if ch in brackets:
                needed_brackets = [*brackets[ch]]
            sentence += ch

            if ch in EOS:
                if len(sentence.strip()):
                    sentences.append(sentence)
                sentence = ''

    if len(sentence.strip()):
        sentences.append(sentence)
    return sentences
