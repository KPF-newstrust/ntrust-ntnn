categories = {
    '정치': 1,
    '경제': 2,
    '사회': 3,
    '국제': 4,
    'IT 과학': 5,
    '문화 예술': 6,
    '교육': 7,
    '연예': 8,
    '스포츠': 9,
    '라이프스타일': 10,
    '사설·칼럼': 11,
    '기타': 12
}


class token:
    Cho, Jung, Jong = 19, 21, 28

    Unk = '\u00e0'  # unknown chars. should be lowercase
    Ascii = range(0x20, 0x7e+1)
    Hangul = range(0xac00, 0xd7a3+1)
    Symbol = range(0x2000, 0x206f+1)
    Symbol2 = range(0x318d, 0x318d+1)

    PadId = 0
    UnkId = PadId + 1
    AsciiId = UnkId + 1
    HangulId = AsciiId + len(Ascii)
    ChoId = HangulId
    JungId = ChoId + Cho
    JongId = JungId + Jung
    SymbolId = JongId + Jong
    Symbol2Id = SymbolId + len(Symbol)

    Total = Symbol2Id + len(Symbol2)
