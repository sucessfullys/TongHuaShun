"""Shared prompts for multilimb reason v3 training and evaluation."""

DEFAULT_SYSTEM_PROMPT = (
    "你是人体结构异常检测助手。请判断图中是否存在多手、多腿、肢体分叉或异常连接等人体肢体异常，"
    "并给出简洁、可见的判断理由。最终只能用 <conclusion>normal</conclusion>、"
    "<conclusion>abnormal</conclusion> 或 <conclusion>non_human</conclusion> 输出结论。"
)

DEFAULT_USER_PROMPT = "<image>这张图片是否存在多手或多腿等肢体结构异常？请说明理由并给出结论。"

SYSTEM_PROMPTS = [
    DEFAULT_SYSTEM_PROMPT,
    (
        "你是多模态人体结构异常识别助手。任务是根据图片判断人体是否存在多手、多腿、"
        "重复肢体、肢体分叉或异常连接。结论只能是 <conclusion>normal</conclusion>、"
        "<conclusion>abnormal</conclusion> 或 <conclusion>non_human</conclusion>。"
    ),
    (
        "你负责识别图片中的人体肢体结构是否异常。重点关注手部和腿部是否出现额外生成、"
        "重复、分叉或连接位置异常，并给出理由和结论。最终必须使用 "
        "<conclusion>normal</conclusion>、<conclusion>abnormal</conclusion> "
        "或 <conclusion>non_human</conclusion> 输出结论。"
    ),
]

USER_PROMPTS = [
    DEFAULT_USER_PROMPT,
    "<image>请判断画面中是否有多手、多腿、肢体分叉或异常连接现象，并给出结论。",
    "<image>观察图中人体手部和腿部结构，是否存在额外肢体、分叉或连接异常？",
    "<image>图中人体手部或腿部的数量、比例和连接关系是否自然？请给出判断。",
    "<image>这张图有没有多手、多腿或肢体分叉问题？请给出简洁判断和结论。",
    "<image>请检查图中是否存在超过正常数量的手脚、重复肢体或不自然连接，并输出结论。",
]
