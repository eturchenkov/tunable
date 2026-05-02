import re, json
from users import users


def parse(text: str) -> str:
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            extracted = json.loads(json_str)
        except:
            return "Error while JSON parsing"
        if "customers" in extracted:
            try:
                user_list = shrink_list(users, extracted["customers"])
                return f"```json\n{str(user_list)}\n```\n"
            except:
                return "No such property in an object"
        elif "result" in extracted:
            return extracted["result"]
        return ""
    else:
        return "No JSON found in string"


def shrink_list(items: list, fields: list[str]) -> list:
    return [{key: item[key] for key in fields} for item in items]
