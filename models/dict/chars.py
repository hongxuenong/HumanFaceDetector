import string
import re
import os
import torch

enChars = string.digits + string.ascii_letters + string.punctuation + ' ' + '°'
cnChars = ''
with open("ptocr/dict/char_cn_uni.txt") as fr:
    for ch in fr:
        cnChars += ch.strip()

symFilter = {
    "en": enChars + "®™",
    "cn": enChars + cnChars + '￥「」【】、《。》' + '㫋㫖㝸㳒≤≥™±○㎡÷©®≈①②③④⑤⑥',
    'th': enChars +
    'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮฯะัาำิีึืฺุูเแโใไๆ็่้๊๋์ํ๑๒๓๔๕๖๗๘๙๐฿',
    'vn': enChars +
    'ÁáÀàẢảÃãẠạĂăẮắẰằẲẳẴẵẶặÂâẤấẦầẨẩẪẫẬậĐđÉéÈèẺẻẼẽẸẹÊêẾếỀềỂểỄễỆệÍíÌìỈỉĨĩỊịÓóÒòỎỏÕõỌọÔôỐốỒồỔổỖỗỘộƠơỚớỜờỞởỠỡỢợÚúÙùỦủŨũỤụƯưỨứỪừỬửỮữỰựÝýỲỳỶỷỸỹỴỵ',
    'pt': enChars + 'çÇÁáÉéÍíÓóÚúÂâÊêÔôÃãÕõÀàÜü',
    'es': enChars + 'ÑñÁÉÍÓÚÝáéàíóúýÜüï¿¡',
    'pt_es': enChars + 'çÇÁáÉéÍíÓóÚúÂâÊêÔôÃãÕõÀàÜüÑñÝýï¿¡',
}


def strQ2B(ustring):
    """full to half"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def replace_similar_symbols(s):
    dict_sym = {
        "`": "'",
        "～": "~",
        "·": ".",
        "！": "!",
        "…": "...",
        "（": "(",
        "）": ")",
        "—": "-",
        "｜": "|",
        " ▏": "|",
        "：": ":",
        "；": ";",
        "“": '"',
        "”": '"',
        "‘": "'",
        "，": ",",
        "？": "?",
        "✚": "+",
        "•": ".",
        "×": "x",
        "〔": "[",
        "〕": "]",
        "＋": "+",
        '﹣': "-",
        '－': "-",
        '﹤':'<',
        '∣':'|',
        "»":">>",
        '℃':"°C",
        "﹝":"[",
        "Ⅰ":"I",
        "Ⅵ":"VI",
        "─":"-",
        "Ⅱ":"II",
        "›":">",
        '‖':"||",
        "Ⅲ":"III",
        "’":"'",
        '〃':'"',
        'Ⅻ':'XII',
        'Ⅸ':'IX',
        'Ⅴ':'V',
        "℉":"°F",
        "✘":"x",
        'Ⅳ':'IV',
        '¥':'￥'
    }
    for sym in dict_sym:
        if sym in s:
            s = s.replace(sym, dict_sym[sym])
    return s


def del_extra_space(s):
    for ws in string.whitespace:
        if ws in s: s = s.replace(ws, ' ')
    s = re.sub(' +', ' ', s)  # merge multi spaces to one space
    s = s.strip()
    commonSym = string.punctuation + '·~！@#￥%……&*（）-——=+【「】」、|；：‘“，《。》、？'

    spaceDel = set()

    for i, c in enumerate(s):
        if c == ' ':
            if s[i - 1] in commonSym or s[
                    i + 1] in commonSym:  # ignore space adjacent to symbols
                spaceDel.add(i)
            elif s[i - 1].isdigit() and s[i + 1].isalpha() or s[i - 1].isalpha(
            ) and s[i + 1].isdigit():  # ignore space between digit and alpha
                spaceDel.add(i)

    strRet = ""
    if len(spaceDel) != 0:
        for i in range(0, len(s)):
            if i not in spaceDel:
                strRet = strRet + s[i]
    else:
        strRet = s

    return strRet


def gen_dict_file():
    langList = ['en', 'cn', 'th', 'vn', 'pt_es']
    langList = ['en']
    for lang in langList:
        fw = open(os.path.join("ptocr/dict", lang + "_char.txt"), "w")
        charList = symFilter[lang]
        for ch in charList:
            fw.write(ch + '\n')
        fw.close()


class CharsetMapper(object):
    """A simple class to map ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.
    """

    def __init__(self, filename='', max_length=30, null_char=u'\u2591'):
        """Creates a lookup table.

        Args:
          filename: Path to charset file which maps characters to ids.
          max_sequence_length: The max length of ids and string.
          null_char: A unicode character used to replace '<null>' character.
            the default value is a light shade block '░'.
        """
        self.null_char = null_char
        self.max_length = max_length

        self.label_to_char = self._read_charset(filename)
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)

    def _read_charset(self, filename):
        """Reads a charset definition from a tab separated text file.

        Args:
          filename: a path to the charset file.

        Returns:
          a dictionary with keys equal to character codes and values - unicode
          characters.
        """
        import re
        # pattern = re.compile(r'(\d+)\t(.+)')
        charset = {}
        self.null_label = 0
        charset[self.null_label] = self.null_char
        with open(filename, 'rb') as f:
            for i, line in enumerate(f):
                # m = pattern.match(line)
                # assert m, f'Incorrect charset file. line #{i}: {line}'
                # label = int(m.group(1)) + 1
                # char = m.group(2)
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                charset[i+1] = line
        return charset

    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, '')

    def get_text(self, labels, length=None, padding=True, trim=False):
        """ Returns a string corresponding to a sequence of character ids.
        """
        length = length if length else self.max_length
        labels = [
            l.item() if isinstance(l, torch.Tensor) else int(l) for l in labels
        ]
        if padding:
            labels = labels + [self.null_label] * (length - len(labels))
        text = ''.join([self.label_to_char[label] for label in labels])
        if trim: text = self.trim(text)
        return text

    def get_labels(self,
                   text,
                   length=None,
                   padding=True,
                   case_sensitive=False):
        """ Returns the labels of the corresponding text.
        """
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length

        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return '0123456789'

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                valid_chars.append(c)
        return ''.join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)


if __name__ == '__main__':
    gen_dict_file()