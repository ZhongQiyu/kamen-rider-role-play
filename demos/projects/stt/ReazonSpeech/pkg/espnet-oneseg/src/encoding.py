# This file is a derivative work of "assdumper.cc" by Kohei Suzuki.
# ----
# Copyright (c) 2014 Kohei Suzuki
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def decode_cprofile(buf):
    """Decode c-profile string.

    This is a custom encoding specific to Japanese television. It's
    basically EUC-JP, with addition of extended characters.
    """
    text = ""
    while buf:
        if 0xa0 < buf[0] and buf[0] < 0xff:
            try:
                text += bytes([buf[0], buf[1]]).decode("euc-jp")
            except UnicodeDecodeError:
                text += _gaiji(buf)
            except IndexError:
                break
            buf = buf[2:]
        elif 0x80 < buf[0] and buf[0] < 0x87:
            buf = buf[1:]
        elif buf[0] in (0x0d, 0x0c, 0x20):
            text += ""
            buf = buf[1:]
        else:
            buf = buf[1:]
    return text

def _gaiji(buf):
    code = (buf[0] & 0x7f) << 8 | (buf[1] & 0x7f)
    return _GAIJI_TABLE.get(code, "")

_GAIJI_TABLE = {
    0x7A50: "【HV】",
    0x7A51: "【SD】",
    0x7A52: "【Ｐ】",
    0x7A53: "【Ｗ】",
    0x7A54: "【MV】",
    0x7A55: "【手】",
    0x7A56: "【字】",
    0x7A57: "【双】",
    0x7A58: "【デ】",
    0x7A59: "【Ｓ】",
    0x7A5A: "【二】",
    0x7A5B: "【多】",
    0x7A5C: "【解】",
    0x7A5D: "【SS】",
    0x7A5E: "【Ｂ】",
    0x7A5F: "【Ｎ】",
    0x7A62: "【天】",
    0x7A63: "【交】",
    0x7A64: "【映】",
    0x7A65: "【無】",
    0x7A66: "【料】",
    0x7A67: "【年齢制限】",
    0x7A68: "【前】",
    0x7A69: "【後】",
    0x7A6A: "【再】",
    0x7A6B: "【新】",
    0x7A6C: "【初】",
    0x7A6D: "【終】",
    0x7A6E: "【生】",
    0x7A6F: "【販】",
    0x7A70: "【声】",
    0x7A71: "【吹】",
    0x7A72: "【PPV】",

    0x7A60: "■",
    0x7A61: "●",
    0x7A73: "（秘）",
    0x7A74: "ほか",

    0x7C21: "→",
    0x7C22: "←",
    0x7C23: "↑",
    0x7C24: "↓",
    0x7C25: "●",
    0x7C26: "○",
    0x7C27: "年",
    0x7C28: "月",
    0x7C29: "日",
    0x7C2A: "円",
    0x7C2B: "㎡",
    0x7C2C: "㎥",
    0x7C2D: "㎝",
    0x7C2E: "㎠",
    0x7C2F: "㎤",
    0x7C30: "０.",
    0x7C31: "１.",
    0x7C32: "２.",
    0x7C33: "３.",
    0x7C34: "４.",
    0x7C35: "５.",
    0x7C36: "６.",
    0x7C37: "７.",
    0x7C38: "８.",
    0x7C39: "９.",
    0x7C3A: "氏",
    0x7C3B: "副",
    0x7C3C: "元",
    0x7C3D: "故",
    0x7C3E: "前",
    0x7C3F: "[新]",
    0x7C40: "０,",
    0x7C41: "１,",
    0x7C42: "２,",
    0x7C43: "３,",
    0x7C44: "４,",
    0x7C45: "５,",
    0x7C46: "６,",
    0x7C47: "７,",
    0x7C48: "８,",
    0x7C49: "９,",
    0x7C4A: "(社)",
    0x7C4B: "(財)",
    0x7C4C: "(有)",
    0x7C4D: "(株)",
    0x7C4E: "(代)",
    0x7C4F: "(問)",
    0x7C50: "▶",
    0x7C51: "◀",
    0x7C52: "〖",
    0x7C53: "〗",
    0x7C54: "⟐",
    0x7C55: "^2",
    0x7C56: "^3",
    0x7C57: "(CD)",
    0x7C58: "(vn)",
    0x7C59: "(ob)",
    0x7C5A: "(cb)",
    0x7C5B: "(ce",
    0x7C5C: "mb)",
    0x7C5D: "(hp)",
    0x7C5E: "(br)",
    0x7C5F: "(p)",
    0x7C60: "(s)",
    0x7C61: "(ms)",
    0x7C62: "(t)",
    0x7C63: "(bs)",
    0x7C64: "(b)",
    0x7C65: "(tb)",
    0x7C66: "(tp)",
    0x7C67: "(ds)",
    0x7C68: "(ag)",
    0x7C69: "(eg)",
    0x7C6A: "(vo)",
    0x7C6B: "(fl)",
    0x7C6C: "(ke",
    0x7C6D: "y)",
    0x7C6E: "(sa",
    0x7C6F: "x)",
    0x7C70: "(sy",
    0x7C71: "n)",
    0x7C72: "(or",
    0x7C73: "g)",
    0x7C74: "(pe",
    0x7C75: "r)",
    0x7C76: "(R)",
    0x7C77: "(C)",
    0x7C78: "(箏)",
    0x7C79: "DJ",
    0x7C7A: "[演]",
    0x7C7B: "Fax",

    0x7D21: "㈪",
    0x7D22: "㈫",
    0x7D23: "㈬",
    0x7D24: "㈭",
    0x7D25: "㈮",
    0x7D26: "㈯",
    0x7D27: "㈰",
    0x7D28: "㈷",
    0x7D29: "㍾",
    0x7D2A: "㍽",
    0x7D2B: "㍼",
    0x7D2C: "㍻",
    0x7D2D: "№",
    0x7D2E: "℡",
    0x7D2F: "〶",
    0x7D30: "○",
    0x7D31: "〔本〕",
    0x7D32: "〔三〕",
    0x7D33: "〔二〕",
    0x7D34: "〔安〕",
    0x7D35: "〔点〕",
    0x7D36: "〔打〕",
    0x7D37: "〔盗〕",
    0x7D38: "〔勝〕",
    0x7D39: "〔敗〕",
    0x7D3A: "〔Ｓ〕",
    0x7D3B: "［投］",
    0x7D3C: "［捕］",
    0x7D3D: "［一］",
    0x7D3E: "［二］",
    0x7D3F: "［三］",
    0x7D40: "［遊］",
    0x7D41: "［左］",
    0x7D42: "［中］",
    0x7D43: "［右］",
    0x7D44: "［指］",
    0x7D45: "［走］",
    0x7D46: "［打］",
    0x7D47: "㍑",
    0x7D48: "㎏",
    0x7D49: "㎐",
    0x7D4A: "ha",
    0x7D4B: "㎞",
    0x7D4C: "㎢",
    0x7D4D: "㍱",
    0x7D4E: "・",
    0x7D4F: "・",
    0x7D50: "1/2",
    0x7D51: "0/3",
    0x7D52: "1/3",
    0x7D53: "2/3",
    0x7D54: "1/4",
    0x7D55: "3/4",
    0x7D56: "1/5",
    0x7D57: "2/5",
    0x7D58: "3/5",
    0x7D59: "4/5",
    0x7D5A: "1/6",
    0x7D5B: "5/6",
    0x7D5C: "1/7",
    0x7D5D: "1/8",
    0x7D5E: "1/9",
    0x7D5F: "1/10",
    0x7D60: "☀",
    0x7D61: "☁",
    0x7D62: "☂",
    0x7D63: "☃",
    0x7D64: "☖",
    0x7D65: "☗",
    0x7D66: "▽",
    0x7D67: "▼",
    0x7D68: "♦",
    0x7D69: "♥",
    0x7D6A: "♣",
    0x7D6B: "♠",
    0x7D6C: "⌺",
    0x7D6D: "⦿",
    0x7D6E: "‼",
    0x7D6F: "⁉",
    0x7D70: "(曇/晴)",
    0x7D71: "☔",
    0x7D72: "(雨)",
    0x7D73: "(雪)",
    0x7D74: "(大雪)",
    0x7D75: "⚡",
    0x7D76: "(雷雨)",
    0x7D77: "　",
    0x7D78: "・",
    0x7D79: "・",
    0x7D7A: "♬",
    0x7D7B: "☎",

    0x7E21: "Ⅰ",
    0x7E22: "Ⅱ",
    0x7E23: "Ⅲ",
    0x7E24: "Ⅳ",
    0x7E25: "Ⅴ",
    0x7E26: "Ⅵ",
    0x7E27: "Ⅶ",
    0x7E28: "Ⅷ",
    0x7E29: "Ⅸ",
    0x7E2A: "Ⅹ",
    0x7E2B: "Ⅺ",
    0x7E2C: "Ⅻ",
    0x7E2D: "⑰",
    0x7E2E: "⑱",
    0x7E2F: "⑲",
    0x7E30: "⑳",
    0x7E31: "⑴",
    0x7E32: "⑵",
    0x7E33: "⑶",
    0x7E34: "⑷",
    0x7E35: "⑸",
    0x7E36: "⑹",
    0x7E37: "⑺",
    0x7E38: "⑻",
    0x7E39: "⑼",
    0x7E3A: "⑽",
    0x7E3B: "⑾",
    0x7E3C: "⑿",
    0x7E3D: "㉑",
    0x7E3E: "㉒",
    0x7E3F: "㉓",
    0x7E40: "㉔",
    0x7E41: "(A)",
    0x7E42: "(B)",
    0x7E43: "(C)",
    0x7E44: "(D)",
    0x7E45: "(E)",
    0x7E46: "(F)",
    0x7E47: "(G)",
    0x7E48: "(H)",
    0x7E49: "(I)",
    0x7E4A: "(J)",
    0x7E4B: "(K)",
    0x7E4C: "(L)",
    0x7E4D: "(M)",
    0x7E4E: "(N)",
    0x7E4F: "(O)",
    0x7E50: "(P)",
    0x7E51: "(Q)",
    0x7E52: "(R)",
    0x7E53: "(S)",
    0x7E54: "(T)",
    0x7E55: "(U)",
    0x7E56: "(V)",
    0x7E57: "(W)",
    0x7E58: "(X)",
    0x7E59: "(Y)",
    0x7E5A: "(Z)",
    0x7E5B: "㉕",
    0x7E5C: "㉖",
    0x7E5D: "㉗",
    0x7E5E: "㉘",
    0x7E5F: "㉙",
    0x7E60: "㉚",
    0x7E61: "①",
    0x7E62: "②",
    0x7E63: "③",
    0x7E64: "④",
    0x7E65: "⑤",
    0x7E66: "⑥",
    0x7E67: "⑦",
    0x7E68: "⑧",
    0x7E69: "⑨",
    0x7E6A: "⑩",
    0x7E6B: "⑪",
    0x7E6C: "⑫",
    0x7E6D: "⑬",
    0x7E6E: "⑭",
    0x7E6F: "⑮",
    0x7E70: "⑯",
    0x7E71: "❶",
    0x7E72: "❷",
    0x7E73: "❸",
    0x7E74: "❹",
    0x7E75: "❺",
    0x7E76: "❻",
    0x7E77: "❼",
    0x7E78: "❽",
    0x7E79: "❾",
    0x7E7A: "❿",
    0x7E7B: "⓫",
    0x7E7C: "⓬",
    0x7E7D: "㉛",

    0x7521: "㐂",
    0x7522: "亭",
    0x7523: "份",
    0x7524: "仿",
    0x7525: "侚",
    0x7526: "俉",
    0x7527: "傜",
    0x7528: "儞",
    0x7529: "冼",
    0x752A: "㔟",
    0x752B: "匇",
    0x752C: "卡",
    0x752D: "卬",
    0x752E: "詹",
    0x752F: "吉",
    0x7530: "呍",
    0x7531: "咖",
    0x7532: "咜",
    0x7533: "咩",
    0x7534: "唎",
    0x7535: "啊",
    0x7536: "噲",
    0x7537: "囤",
    0x7538: "圳",
    0x7539: "圴",
    0x753A: "塚",
    0x753B: "墀",
    0x753C: "姤",
    0x753D: "娣",
    0x753E: "婕",
    0x753F: "寬",
    0x7540: "﨑",
    0x7541: "㟢",
    0x7542: "庬",
    0x7543: "弴",
    0x7544: "彅",
    0x7545: "德",
    0x7546: "怗",
    0x7547: "恵",
    0x7548: "愰",
    0x7549: "昤",
    0x754A: "曈",
    0x754B: "曙",
    0x754C: "曺",
    0x754D: "曻",
    0x754E: "桒",
    0x754F: "・",
    0x7550: "椑",
    0x7551: "椻",
    0x7552: "橅",
    0x7553: "檑",
    0x7554: "櫛",
    0x7555: "・",
    0x7556: "・",
    0x7557: "・",
    0x7558: "毱",
    0x7559: "泠",
    0x755A: "洮",
    0x755B: "海",
    0x755C: "涿",
    0x755D: "淊",
    0x755E: "淸",
    0x755F: "渚",
    0x7560: "潞",
    0x7561: "濹",
    0x7562: "灤",
    0x7563: "・",
    0x7564: "・",
    0x7565: "煇",
    0x7566: "燁",
    0x7567: "爀",
    0x7568: "玟",
    0x7569: "・",
    0x756A: "珉",
    0x756B: "珖",
    0x756C: "琛",
    0x756D: "琡",
    0x756E: "琢",
    0x756F: "琦",
    0x7570: "琪",
    0x7571: "琬",
    0x7572: "琹",
    0x7573: "瑋",
    0x7574: "㻚",
    0x7575: "畵",
    0x7576: "疁",
    0x7577: "睲",
    0x7578: "䂓",
    0x7579: "磈",
    0x757A: "磠",
    0x757B: "祇",
    0x757C: "禮",
    0x757D: "・",
    0x757E: "・",

    0x7621: "・",
    0x7622: "秚",
    0x7623: "稞",
    0x7624: "筿",
    0x7625: "簱",
    0x7626: "䉤",
    0x7627: "綋",
    0x7628: "羡",
    0x7629: "脘",
    0x762A: "脺",
    0x762B: "・",
    0x762C: "芮",
    0x762D: "葛",
    0x762E: "蓜",
    0x762F: "蓬",
    0x7630: "蕙",
    0x7631: "藎",
    0x7632: "蝕",
    0x7633: "蟬",
    0x7634: "蠋",
    0x7635: "裵",
    0x7636: "角",
    0x7637: "諶",
    0x7638: "跎",
    0x7639: "辻",
    0x763A: "迶",
    0x763B: "郝",
    0x763C: "鄧",
    0x763D: "鄭",
    0x763E: "醲",
    0x763F: "鈳",
    0x7640: "銈",
    0x7641: "錡",
    0x7642: "鍈",
    0x7643: "閒",
    0x7644: "雞",
    0x7645: "餃",
    0x7646: "饀",
    0x7647: "髙",
    0x7648: "鯖",
    0x7649: "鷗",
    0x764A: "麴",
    0x764B: "麵",
}
