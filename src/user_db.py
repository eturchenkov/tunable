import re, json
from users import users


def parse(text: str) -> tuple[str, int]:
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        extracted = json.loads(json_str)
        if "customers" in extracted:
            try:
                user_list = shrink_list(users, extracted["customers"])
                return (
                    f"```json\n{str(user_list)}\n```\n",
                    1,
                )
            except:
                return "No such property in an object", 0
        elif "result" in extracted:
            return extracted["result"], 1
        return "", 0
    else:
        return "No JSON found in string", 0


def shrink_list(items: list, fields: list[str]) -> list:
    return [{key: item[key] for key in fields} for item in items]
