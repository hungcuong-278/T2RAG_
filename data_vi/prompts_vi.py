"""Vietnamese OpenIE prompts (M4): NER, then triples conditioned on the entities.

Same two-step structure and JSON keys as the repo's English prompts
(src/prompts/templates/ner.py, triple_extraction.py), with Vietnamese instructions and a
one-shot example written for this project (Ngo Quyen, outside the corpus period).
Bump PROMPT_VERSION whenever any text below changes; the cache is keyed by it.
"""
import json

PROMPT_VERSION = "v2"

EXAMPLE_PASSAGE = """Ngô Quyền
Ngô Quyền (chữ Hán: 吳權; 898–944) là người lập ra nhà Ngô. Năm 938, ông chỉ huy quân dân đánh tan quân Nam Hán trên sông Bạch Đằng. Trong trận này, ông cho cắm cọc gỗ đầu bịt sắt dưới lòng sông để đánh thuyền giặc khi nước triều rút. Năm 939, ông xưng vương, đóng đô ở Cổ Loa. Ông ở ngôi 6 năm thì mất, thọ 47 tuổi. Vợ ông là Dương Thị Như Ngọc, con gái của Dương Đình Nghệ."""

EXAMPLE_ENTITIES = ["Ngô Quyền", "898", "944", "nhà Ngô", "938", "Nam Hán", "sông Bạch Đằng", "939",
                    "Cổ Loa", "6 năm", "47 tuổi", "Dương Thị Như Ngọc", "Dương Đình Nghệ"]

EXAMPLE_TRIPLES = [
    ["Ngô Quyền", "sinh năm", "898"],
    ["Ngô Quyền", "mất năm", "944"],
    ["Ngô Quyền", "lập ra", "nhà Ngô"],
    ["Ngô Quyền", "chỉ huy quân dân đánh tan", "quân Nam Hán"],
    ["Ngô Quyền", "đánh tan quân Nam Hán trên sông Bạch Đằng năm", "938"],
    ["Ngô Quyền", "cho cắm cọc gỗ đầu bịt sắt dưới lòng", "sông Bạch Đằng"],
    ["Ngô Quyền", "xưng vương năm", "939"],
    ["Ngô Quyền", "đóng đô ở", "Cổ Loa"],
    ["Ngô Quyền", "ở ngôi trong", "6 năm"],
    ["Ngô Quyền", "mất khi thọ", "47 tuổi"],
    ["Ngô Quyền", "có vợ là", "Dương Thị Như Ngọc"],
    ["Dương Thị Như Ngọc", "là con gái của", "Dương Đình Nghệ"],
]

NER_SYSTEM = """Nhiệm vụ của bạn là trích xuất các thực thể có tên từ một đoạn văn về lịch sử Việt Nam.
Thực thể gồm: người (kể cả miếu hiệu, tên húy, tước hiệu, tên gọi khác), triều đại, quốc gia, địa danh, công trình, trận đánh, cuộc chiến, sự kiện, văn bản và tác phẩm, niên hiệu, mốc thời gian (năm), số tuổi và khoảng thời gian (ví dụ "6 tuổi", "17 năm").
Không lấy chữ Hán: bỏ phần chữ Hán trong ngoặc, chỉ giữ tên bằng chữ quốc ngữ.
Không lấy danh xưng chung đứng một mình như "vua", "hoàng đế", "Thượng hoàng", "hoàng tử", "hoàng hậu", "đại sứ"; nếu danh xưng đi kèm tên thì chỉ lấy tên người.
Giữ nguyên cách viết như trong đoạn văn, không dịch, không thêm thực thể không có trong đoạn văn.
Chỉ trả lời bằng JSON dạng {"named_entities": [...]}."""

TRIPLE_SYSTEM = """Nhiệm vụ của bạn là xây dựng đồ thị tri thức từ một đoạn văn về lịch sử Việt Nam và danh sách thực thể của nó.
Trả lời bằng JSON dạng {"triples": [[chủ thể, quan hệ, đối tượng], ...]}.

Yêu cầu:
- Mỗi bộ ba có đúng 3 phần; đọc liền "chủ thể quan hệ đối tượng" phải thành một câu tiếng Việt đúng và tự đủ nghĩa.
- Thay mọi đại từ và danh xưng như "ông", "bà", "vua", "nhà vua", "ngài", "người này" bằng tên đầy đủ của người được nói tới.
- Dùng tên thực thể đúng như trong đoạn văn; mỗi bộ ba nên chứa ít nhất một, tốt nhất là hai thực thể trong danh sách.
- Năm, niên hiệu, địa danh là thực thể. Có thể đưa thời gian vào phần quan hệ để giữ ngữ cảnh, ví dụ ["Ngô Quyền", "xưng vương năm", "939"].
- Khi đoạn văn nêu tên gọi khác (tên húy, miếu hiệu, tước hiệu), tạo bộ ba nối hai tên, ví dụ ["Lý Công Uẩn", "có miếu hiệu là", "Lý Thái Tổ"].
- Ghi đủ các quan hệ gia đình, kế vị, chức vụ, trận đánh, năm sinh, năm mất có trong đoạn văn.
- Ghi đủ số tuổi và khoảng thời gian có trong đoạn văn, ví dụ ["Trần Dụ Tông", "lên ngôi khi", "6 tuổi"], ["Lê Nhân Tông", "trị vì trong", "17 năm"].
- Viết quan hệ ngắn gọn, ở thể chủ động, chọn chủ thể là người thực hiện hành động; không dùng các cụm như "đối với", "của việc".
- Không tạo bộ ba về chữ Hán và không đưa chữ Hán vào chủ thể hay đối tượng.
- Không lặp lại một dữ kiện theo hai chiều (đã có "A soạn chung với B" thì không thêm "B soạn chung với A").
- Không thêm kiến thức ngoài đoạn văn.
"""

TRIPLE_FRAME = """Chuyển đoạn văn sau thành JSON gồm danh sách bộ ba.
Đoạn văn:
```
{passage}
```

{named_entity_json}
"""


def ner_messages(passage):
    return [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user", "content": EXAMPLE_PASSAGE},
        {"role": "assistant", "content": json.dumps({"named_entities": EXAMPLE_ENTITIES}, ensure_ascii=False)},
        {"role": "user", "content": passage},
    ]


def triple_messages(passage, entities):
    def frame(p, ents):
        return TRIPLE_FRAME.format(passage=p, named_entity_json=json.dumps({"named_entities": ents}, ensure_ascii=False))
    return [
        {"role": "system", "content": TRIPLE_SYSTEM},
        {"role": "user", "content": frame(EXAMPLE_PASSAGE, EXAMPLE_ENTITIES)},
        {"role": "assistant", "content": json.dumps({"triples": EXAMPLE_TRIPLES}, ensure_ascii=False)},
        {"role": "user", "content": frame(passage, entities)},
    ]
