import re
from num2words import num2words

ordinal_endings = r'(\d+)-?(е|ы?й|о?м|о?го|х|ую|я)'
endings = {
    'е': [1, 'е'],
    'ый': [2, 'ый'],
    'й': [1, 'й'],
    'ом': [2, 'ом'],
    'м': [1, 'м'],
    'ого': [2, 'ого'],
    'го': [2, 'ого'],
    'х': [1, 'х'],
    'ую': [2, 'ую'],
    'я': [2, 'ая']
}


def maybe_to_numeric(val):
    if not isinstance(val, str):
        return val
    val_ = re.sub(r'\s', '', val)
    if '.' in val_ and ',' in val_:
        val_ = val_.replace(',', '')
    elif ',' in val_:
        val_ = val_.replace(',', '')
    val_ = val_.replace(',', '.')
    if val_.isnumeric():
        type_ = float if '.' in val_ else int
        try:
            val_ = type_(val_)
        except ValueError:
            val_ = val
        val = val_
    elif not val_:
        val = None
    return val


def change_numeric_rus(w):
    ordinal = False
    m = re.match(ordinal_endings, w)
    if m is not None:
        w = m.group(1)
        ending = m.group(2)
        ordinal = True

    w_numeric = maybe_to_numeric(w)
    if isinstance(w_numeric, int):
        w = num2words(w_numeric, lang='ru', ordinal=ordinal)
    elif isinstance(w_numeric, float):
        if w_numeric.is_integer():
            w_numeric = int(w_numeric)
        w = num2words(w_numeric, lang='ru', ordinal=ordinal)
    if ordinal:
        i, suff = endings.get(ending, [1, ending])
        w = w[:-i] + suff
    return w


NUMBER_REGEXP = r"\d+(?:[,\d]*\d)?(?:\.\d+)?"


def change_numeric_eng(n):
    if '911' in n:
        n = n.replace('911', 'nine one one')

    if '$' in n:
        n = n.replace('$', '')

        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en') + ' dollars', n)
        return text

    if ':' in n:
        h, m = n.split(':')
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en'), h)
        text += ' ' + re.sub(NUMBER_REGEXP,
                             lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en') if maybe_to_numeric(
                                 x.group(0)) != 0 else '',
                             m)
        return text

    if re.search(r"1\d{3}", n) is not None:
        if re.search(r"\d\d00", n) is not None:
            n = re.sub(r'00', r' hundred', n)
            text = re.sub(NUMBER_REGEXP, lambda x: ' ' + num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        else:
            text = re.sub(r'\d{2}', lambda x: ' ' + num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        text = text.replace("'s", 's')
        text = text.replace('tys', 'ties')
        return text

    if '%' in n:
        n = n.replace('%', ' percent')
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        return text

    if re.search(r"\d'?s", n) is not None:
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        text = text.replace("'s", 's')
        text = text.replace('tys', 'ties')
        return text

    if re.search(r'\d"', n) is not None:
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        text = text.replace('"', ' inches')
        return text

    if re.search(r'\d+(?:th|st|nd|rd)', n) is not None:
        n = re.sub(r'(\d+)(?:th|st|nd|rd)', r'\1', n)
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en', to='ordinal'), n)
        return text

    if re.search(r'\d+[-.,!?]', n) is not None:
        text = re.sub(NUMBER_REGEXP, lambda x: ' ' + num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        return text
    text = re.sub(NUMBER_REGEXP, lambda x: f" {num2words(maybe_to_numeric(x.group(0)), lang='en')} ", n)
    return text
