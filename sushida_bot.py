"""
寿司打 自動入力ボット（個人の実験用）
画面のローマ字表示部分をOCRで読み取り、キー入力を送ります。

必要なもの:
  pip install mss pillow pytesseract pyautogui keyboard
  Tesseract本体（Windowsなら UB-Mannheim 版のインストーラ）

使い方:
  1. python sushida_bot.py calib
     → マウスをローマ字表示の「左上」「右下」に置いて F8。座標が表示される
  2. REGION は自動で書き換わる
  3. python sushida_bot.py test   → OCR結果と debug.png / debug_full.png を確認（閾値調整用）
  4. python sushida_bot.py        → 寿司打の画面で Enter/Space でゲーム開始すると自動で入力開始
                                     （F8 で手動開始も可）。F9 で停止（終了は手動）
     ※ 止まったら、入力済み（グレー）の部分は打たず、白い未入力の部分だけ打ち直す
     ※ 非常停止: マウスを画面の左上隅に動かす（pyautogui の FAILSAFE）
"""
import ctypes
import difflib
import os
import re
import sys
import time

import keyboard
import mss
import pyautogui
import pytesseract
from PIL import Image, ImageOps

# Windowsで tesseract に PATH が通っていない場合はコメントを外す
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ローマ字表示の範囲（calib で調べた値に書き換える）
REGION = {"left": 1449, "top": 926, "width": 970, "height": 76}

THRESHOLD = 150        # 未入力（白い文字）だけを読むしきい値。入力済みのグレー（明るさ90〜100）は消える
FULL_THRESHOLD = 80    # グレーも含めて単語全体を読むしきい値（背景は明るさ30〜50。低すぎると文字が太ってつぶれる）
CROP_THRESHOLD = 65    # 文字の範囲を探すときのしきい値
SCALE = 2              # OCR前の拡大倍率
INVERT = False         # 文字が背景より暗いなら True
KEY_INTERVAL = 0.035   # 1打ごとの間隔(秒)。小さすぎるとゲーム側が取りこぼす
STUCK_TIME = 0.8       # 打ってからこの秒数たっても同じ皿なら、白い（未入力の）部分だけを打ち直す
MAX_FIX = 3            # 1皿で打ち直す最大回数（超えたら次の皿まで待つ）
SAME_PLATE = 0.9       # 全体の読み取りがこの類似度以上（かつ長さの差2文字以内）なら同じ皿とみなす（OCRの揺れ対策）
POLL = 0.05            # 画面チェック間隔(秒)
GAME_TITLE = "寿司打"    # このウィンドウが前面のときだけ Enter/Space で自動開始＆入力する（空文字ならチェックしない）
# 皿カウントで自動停止はしない（ずれるため）。終わらせるときは F9 で手動停止

ALLOWED = "abcdefghijklmnopqrstuvwxyz-,.!?"
# 英語の辞書補正はローマ字では逆効果なので切る
OCR_CONFIG = (f"--psm 7 -c tessedit_char_whitelist={ALLOWED}"
              " -c load_system_dawg=0 -c load_freq_dawg=0")

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def capture(sct):
    shot = sct.grab(REGION)
    return Image.frombytes("RGB", shot.size, shot.rgb).convert("L")


def text_span(gray):
    """明るい文字のまとまりのうち、範囲の中央にあるものの左右端を返す。
    左の皿カウンターや右の湯のみなど、離れたところにある明るいものは除外される"""
    w, h = gray.size
    px = gray.load()
    gap = int(h * 0.6)  # 文字間のすき間より広く、単語と端のゴミの間より狭い
    runs, start, prev = [], None, None
    for x in range(w):
        if any(px[x, y] > CROP_THRESHOLD for y in range(0, h, 2)):
            if start is None:
                start = x
            elif x - prev > gap:
                runs.append((start, prev))
                start = x
            prev = x
    if start is not None:
        runs.append((start, prev))
    if not runs:
        return None
    cx = w / 2
    return min(runs, key=lambda r: 0 if r[0] <= cx <= r[1] else min(abs(r[0] - cx), abs(r[1] - cx)))


def prepare(gray, threshold, span):
    """文字の部分だけ切り出し → 拡大 → 2値化 → 反転（白地に黒文字）→ 余白をつける"""
    if span is None:
        return Image.new("L", (10, 10), 255)
    x0, x1 = span
    g = gray.crop((max(0, x0 - 4), 0, min(gray.width, x1 + 5), gray.height))
    g = g.resize((g.width * SCALE, g.height * SCALE), Image.LANCZOS)
    img = g.point(lambda p: 255 if p > threshold else 0)
    if not INVERT:
        img = ImageOps.invert(img)  # Tesseract は白背景に黒文字が得意
    return ImageOps.expand(img, border=30, fill=255)


def grab(sct, threshold=THRESHOLD):
    gray = capture(sct)
    return prepare(gray, threshold, text_span(gray))


def ocr(img):
    text = pytesseract.image_to_string(img, config=OCR_CONFIG)
    return "".join(c for c in text.strip().lower() if c in ALLOWED)


def read_both(sct):
    """(単語全体, 未入力の白い部分) を同じ画面・同じ切り出し範囲から読む"""
    gray = capture(sct)
    span = text_span(gray)
    return ocr(prepare(gray, FULL_THRESHOLD, span)), ocr(prepare(gray, THRESHOLD, span))


def remaining_part(word, rest):
    """白い部分のOCR結果 rest を、正しい単語 word の末尾に合わせて返す"""
    if not rest:
        return ""
    best, best_score = "", 0.0
    for i in range(len(word)):
        cand = word[i:]
        if abs(len(cand) - len(rest)) > 2:
            continue
        score = difflib.SequenceMatcher(None, cand, rest).ratio()
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= 0.5 else ""


# ---- 単語帳による補正 -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORDS_FILE = os.path.join(BASE_DIR, "words.txt")      # 正しい単語（1行1単語）
UNKNOWN_FILE = os.path.join(BASE_DIR, "unknown.txt")  # 単語帳になかったOCR結果
MATCH_CUTOFF = 0.85   # 単語帳とどのくらい似ていたら置き換えるか（0〜1、大きいほど厳しい）

# ローマ字として正しい並びか（子音の重ね・拗音・n・記号を含む）
_SYL = r"(?:[aiueo]|nn|n(?![aiueoy])|([bcdfghjkmnprstvwyzx])\1?(?:[yhsw])?[aiueo]|[-,.!?])"
VALID_ROMAJI = re.compile(rf"^(?:{_SYL})+$")


def is_romaji(word):
    return bool(VALID_ROMAJI.match(word))


def load_words():
    if not os.path.exists(WORDS_FILE):
        return set()
    with open(WORDS_FILE, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


WORDS = load_words()
_logged_unknown = set()


def fix_romaji(word):
    """日本語ローマ字のルールでOCRの読み違いを直す"""
    # v の後ろは u（ヴ）しか来ない → それ以外は y の読み違い
    word = re.sub(r"v(?!u)", "y", word)
    # 寿司打の表示に l は出てこない → i の読み違い
    word = word.replace("l", "i")
    # q も出てこない → g の読み違い
    word = word.replace("q", "g")
    # ! は文の最後にしか来ない → 途中（後ろに文字が続く）の ! は i の読み違い
    word = re.sub(r"!(?=[a-z])", "i", word)
    # ローマ字は n 以外の子音で終わらない → 子音の直後の ! は i の読み違い（例: sush! → sushi）
    word = re.sub(r"(?<=[bcdfghjkmprstvwxyz])!", "i", word)
    # 枠の端のゴミを削る
    #   先頭: 「-」や記号で始まる単語はない
    word = word.lstrip("-,.!?")
    #   末尾: n 以外の子音で終わる単語はない（右の湯のみの縁が y と読まれる対策）
    word = re.sub(r"[bcdfghjkmpqrstvwxyz]+$", "", word)
    return word


def learn(word):
    """未登録だったが打ち切れた（次の皿に進んだ）単語を単語帳に追加する"""
    if word in WORDS or not is_romaji(word):
        return
    WORDS.add(word)
    with open(WORDS_FILE, "a", encoding="utf-8") as f:
        f.write(word + "\n")
    print(f"  ＋単語帳に追加: {word}（{len(WORDS)}語）")
    log(f"LEARN {word!r}")


def correct(word):
    """OCR結果を単語帳に合わせて補正する。戻り値: (補正後, 補正方法)"""
    if word in WORDS:
        return word, "単語帳"
    fixed = fix_romaji(word)
    if fixed in WORDS:
        return fixed, "ルール補正→単語帳"
    if is_romaji(fixed):
        # ローマ字として正しく読めているなら、似た単語に置き換えずそのまま信じる
        return fixed, "未登録"
    if WORDS:
        candidates = [w for w in WORDS if abs(len(w) - len(fixed)) <= 2]
        match = difflib.get_close_matches(fixed, candidates, n=1, cutoff=MATCH_CUTOFF)
        if match:
            return match[0], "近い単語"
    if fixed not in _logged_unknown:
        _logged_unknown.add(fixed)
        with open(UNKNOWN_FILE, "a", encoding="utf-8") as f:
            f.write(fixed + "\n")
    return fixed, "未登録"


def calib():
    points = []
    print("どの画面にいてもOK。マウスを合わせて F8 を押してください（クリック不要）")
    for label in ("左上", "右下"):
        print(f"マウスをローマ字表示の{label}に置いて F8 ...", flush=True)
        keyboard.wait("f8")
        p = pyautogui.position()
        time.sleep(0.3)  # 押しっぱなしで2回分記録されないように
        print(f"  {label}: {p}")
        points.append(p)
    (x1, y1), (x2, y2) = points
    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)
    if width < 5 or height < 5:
        print("範囲が小さすぎます。もう一度やり直してください")
        return

    new_line = f'REGION = {{"left": {left}, "top": {top}, "width": {width}, "height": {height}}}'
    path = os.path.abspath(__file__)
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read()
    src, n = re.subn(r"^REGION = \{[^\r\n]*\}", new_line, src, count=1, flags=re.M)
    if n != 1:
        print("REGION の行が見つかりませんでした。手動で書き換えてください:")
        print(" ", new_line)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(src)
    print("書き換えました:", new_line)


def test():
    with mss.mss() as sct:
        gray = capture(sct)
        span = text_span(gray)
        prepare(gray, FULL_THRESHOLD, span).save("debug_full.png")
        prepare(gray, THRESHOLD, span).save("debug.png")
        full, rest = read_both(sct)
        word, how = correct(full)
        print("OCR全体  :", repr(full), " 未入力部分:", repr(rest))
        print(f"補正後   : {word!r}（{how}） 未入力として打つ: {remaining_part(word, rest)!r} 単語帳: {len(WORDS)}語")
        print("debug_full.png（グレー込み）と debug.png（白い文字だけ）を保存しました")


LOG_FILE = os.path.join(BASE_DIR, "log.txt")
STUCK_DIR = os.path.join(BASE_DIR, "stuck")
_stuck_count = 0


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def save_stuck(sct, raw, word):  # raw は単語全体のOCR結果
    """止まったときの画面を保存（原因調査用、最大30枚）"""
    global _stuck_count
    log(f"STUCK raw={raw!r} word={word!r}")
    if _stuck_count >= 30:
        return
    _stuck_count += 1
    os.makedirs(STUCK_DIR, exist_ok=True)
    big = dict(REGION)
    big["left"] -= 150
    big["width"] += 300
    big["top"] -= 10
    big["height"] += 20
    shot = sct.grab(big)
    name = f"{_stuck_count:02d}_{word}"
    Image.frombytes("RGB", shot.size, shot.rgb).save(os.path.join(STUCK_DIR, name + "_screen.png"))
    grab(sct).save(os.path.join(STUCK_DIR, name + "_ocr.png"))
    grab(sct, FULL_THRESHOLD).save(os.path.join(STUCK_DIR, name + "_ocr_full.png"))


def foreground_title():
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        return buf.value
    except Exception:
        return ""


def game_focused():
    return not GAME_TITLE or GAME_TITLE in foreground_title()


def run():
    print("寿司打の画面で Enter / Space を押してゲームを始めると自動で入力開始します")
    print("F8: 手動で開始 / F9: 停止（次のゲームもまた Enter で自動開始） / Ctrl+C: 終了")
    # 起動時に押していた Enter（ターミナルでの実行）を開始の合図と誤認しないよう、離すまで待つ
    while keyboard.is_pressed("enter") or keyboard.is_pressed("space"):
        time.sleep(0.05)
    active = False
    last = ""
    prev_raw = ""
    fixes = 0       # 今の皿で未入力部分を打ち直した回数
    typed_at = 0.0
    last_how = ""
    plates = 0  # 打った枚数（目安の表示のみ）
    with mss.mss() as sct:
        while True:
            if not active:
                start = keyboard.is_pressed("f8")
                if not start and (keyboard.is_pressed("enter") or keyboard.is_pressed("space")):
                    if game_focused():
                        start = True
                    else:
                        print(f"  （前面のウィンドウ「{foreground_title()}」は寿司打ではないので開始しません）")
                        while keyboard.is_pressed("enter") or keyboard.is_pressed("space"):
                            time.sleep(0.05)
                if start:
                    active, last, prev_raw, fixes = True, "", "", 0
                    plates = 0
                    print("開始（F9 で停止）")
            if keyboard.is_pressed("f9") and active:
                active = False
                print(f"停止（{plates} 枚入力）")
                log(f"MANUAL-STOP plates={plates}")
            if not active:
                time.sleep(0.1)
                continue

            if not game_focused():
                # 寿司打以外のウィンドウにキーを送らないよう、前面に戻るまで待つ
                time.sleep(0.1)
                continue

            full, rest = read_both(sct)
            # 切り替わり途中の画面を読まないよう、同じ結果が2回続いたときだけ採用
            if (full, rest) != prev_raw:
                prev_raw = (full, rest)
                time.sleep(POLL)
                continue
            if len(full) < 2:
                time.sleep(POLL)
                continue

            word, how = correct(full)
            same_plate = last and (
                word == last
                or (abs(len(word) - len(last)) <= 2
                    and difflib.SequenceMatcher(None, word, last).ratio() >= SAME_PLATE)
            )

            if same_plate:
                # 打ち終わったのに同じ皿のまま → 白い（未入力の）部分だけ打ち直す
                if time.time() - typed_at < STUCK_TIME:
                    time.sleep(POLL)
                    continue
                todo = remaining_part(last, rest)
                if not todo or fixes >= MAX_FIX:
                    time.sleep(POLL)
                    continue
                if fixes == 0:
                    save_stuck(sct, full, last)
                fixes += 1
                print(f"  ※未入力の部分だけ打ち直し({fixes}/{MAX_FIX}): {todo}  (白い部分のOCR: {rest})")
                log(f"fix #{plates} rest={rest!r} todo={todo!r}")
                pyautogui.write(todo, interval=KEY_INTERVAL)
                typed_at = time.time()
                prev_raw = ""
                continue

            # 新しい皿
            # 前の単語が未登録でも、打ち直しなしで次の皿に進んだ＝正しかったので覚える
            if last and fixes == 0 and last_how == "未登録":
                learn(last)
            fixes = 0
            plates += 1
            # 皿が出た直後なら全部白いはずだが、念のため白い部分に合わせる（入力済みは打たない）
            todo = remaining_part(word, rest) or word

            note = f"  (OCR: {full}, {how})" if word != full or how == "未登録" else ""
            if todo != word:
                note += f"  [入力済みを除く: {todo}]"
            print(f"[{plates:3d}] 入力:", word + note)
            log(f"type #{plates} raw={full!r} rest={rest!r} word={word!r} how={how}")
            pyautogui.write(todo, interval=KEY_INTERVAL)
            last, last_how, typed_at = word, how, time.time()
            prev_raw = ""
            time.sleep(0.15)  # 次の皿に切り替わるまで少し待つ

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"calib": calib, "test": test, "run": run}.get(mode, run)()