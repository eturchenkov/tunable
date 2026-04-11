import re, json
from users import users


def parse(text: str):
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        extracted = json.loads(json_str)
        if "customers" in extracted:
            return str(shrink_list(users, extracted["customers"]))
        elif "result" in extracted:
            return extracted["result"]
    else:
        return "No JSON found in string"


def shrink_list(items: list, fields: list[str]) -> list:
    return [{key: item[key] for key in fields} for item in items]
