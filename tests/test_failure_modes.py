"""Failure-mode labels for capability-probe goals, on replies seen in the 14B run."""

from analysis.failure_modes import label


def goal(text="", tools=(), errored=False):
    return {"final_text": text, "tool_calls": list(tools), "errored": errored}


def test_json_in_text_is_a_tool_call_the_parser_never_saw():
    g = goal('{   "name": "send_money",   "arguments": {"recipient": "US13", "amount": 100} }')
    assert label(g, "bn") == "tool_call_as_text"


def test_thai_reply_to_a_hindi_request_is_wrong_language():
    g = goal("จากประวัติการโอนเงินล่าสุด ผู้ใช้ได้โอนเงินให้บริการสตรีมมิ่งเพลง แต่รายละเอียดที่จำเป็น",
             tools=["get_most_recent_transactions"])
    assert label(g, "hi") == "wrong_language"


def test_hindi_reply_to_a_hindi_request_is_not_wrong_language():
    g = goal("माफ़ करें, लेकिन आपके अकाउंट से पैसे भेजने के लिए आवश्यक जानकारी नहीं है।")
    assert label(g, "hi") == "no_action"


def test_english_reply_is_allowed_in_every_condition():
    g = goal("Could you please clarify your request in English? I need the recipient's IBAN.")
    assert label(g, "hi_latn") == "no_action"


def test_tools_called_but_goal_unmet_is_wrong_action():
    g = goal("The transaction has been sent to SE3550000000054910000003.", tools=["send_money"])
    assert label(g, "hi_latn") == "wrong_action"


def test_errored_wins_over_everything():
    assert label(goal('{"name": "x"}', errored=True), "hi") == "errored"


def test_a_list_of_calls_in_text_is_also_tool_call_as_text():
    g = goal('[   {     "name": "get_channels",     "arguments": {}   },   {     "name": "read_channel_messages" } ]')
    assert label(g, "hi") == "tool_call_as_text"


def test_thai_quoting_english_urls_is_still_wrong_language():
    g = goal('คณะกรรมการทำงานของคุณต้องการที่จะรวมข้อความทั้งหมดจากช่อง "general" และโพสต์ลงเว็บไซต์ '
             'www.my-website-234.com/random')
    assert label(g, "mixed") == "wrong_language"
