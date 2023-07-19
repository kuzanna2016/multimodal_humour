import re

REGEXPS = (
    (r'(\w*[уэоаыяиюъ]|^)\*+б', r'\1еб'),
    (r'(\w*[уэоаыяиюъ]|^)\*+(а|ок)', r'\1еб\2'),
    (r'(\w*[^уэоаыяиюъ*])\*+б', r'\1ъеб'),
    (r'е\*+(л|т)', r'еба\1'),
    (r'е\*+([ауе](?:л|т|ная|ны[йе]|ное|нут|$))', r'еб\1'),
    (r'е\*+а?н(с|$|ов$|а$)', r'еблан\1'),
    (r'е\*+н', r'ебан'),
    (r'еб\*+(ись|ый)', r'ебан\1'),
    (r'еб\*+о', r'ебало'),
    (r'е\*+и?сь', r'ебись'),
    (r'з\*+ись', r'заебись'),
    (r'ё\*+т', r'ёбывают'),
    (r'раз\*+ал', r'разъебал'),
    (r'раз\*+м', r'разъебем'),
    (r'у\*+о?к', r'уебок'),
    (r'^е\*+$', r'ебать'),
    (r'г\*+ндон', r'гандон'),
    (r'долб\*+($|[аыо])', r'долбаеб\1'),
    (r'с\*+к', r'сук'),
    (r'бл?\*+(л?я?т?ь|$)', r'блять'),
    (r'х\*+([йеёяю](?!шь))', r'ху\1'),
    (r'ох?\*+(ешь|н)', r'охуе\1'),
    (r'ох?\*+([оеаяыйх]+)$', r'охуенн\1'),
    (r'х\*+н', r'хуйн'),
    (r'^(на|по)?х\*+(у?й|$)', r'\1хуй'),
    (r'\*уй', r'хуй'),
    (r'(ни)?х\*+($|[кр])', r'\1хуя\2'),
    (r'х\*+в', r'хуев'),
    (r'х\*+(а\wт)', r'хуев\1'),
    (r'х\*+л', r'хул'),
    (r'п\*+о?р', r'пидор'),
    (r'пи?\*+з?д?е?ц', r'пиздец'),
    (r'пи?\*+з?д?([иаеоуыя])', r'пизд\1'),
    (r'пи?\*+$', r'пизда'),
    (r'п\*+з?ж', r'пизж'),
    (r'п\*+(нуть|т|$)', r'пизда\1'),
)


def preproc_swear_word(word):
    for reg, rep in REGEXPS:
        word = re.sub(reg, rep, word, flags=re.IGNORECASE)
        if '*' not in word:
            return word
    return word
