import re


def clean_title(title):
    title = re.sub(r'[|.,"/]', r'', title)
    title = re.sub(r'й', r'й', title)
    title = re.sub(r'ё', r'ё', title)
    return title


def get_tier_by_name(textgrid_obj, tier_name):
    for tier in textgrid_obj:
        if tier.name == tier_name:
            return tier
    return None
