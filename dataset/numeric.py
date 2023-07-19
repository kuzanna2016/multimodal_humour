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


def change_numeric(w):
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
