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

ゲーム中の流れ（目標: ミスタイプ0）:
  OCRで読む → 単語帳を検索
    ・単語帳にある（または読み違いを直すと単語帳の単語になる）→ 単語帳の綴りを打つ（＝正解が確定している）
    ・単語帳にない → 読んだものをそのまま打つ。ミスなく通ったら「候補」に記録し、
                     LEARN_CONFIRM 回同じ読みで通ったら単語帳（words.txt）に昇格させる
  ゲームを繰り返すほど単語帳が充実し、単語帳から打つ割合（＝正答率）が100%に近づく。
"""
import ctypes
import difflib
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import keyboard
import mss
import pyautogui
import pytesseract
from PIL import Image, ImageOps

# Windowsで tesseract に PATH が通っていない場合はコメントを外す
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ローマ字表示の範囲（calib で調べた値に書き換える）
REGION = {"left": 1449, "top": 930, "width": 970, "height": 65}

THRESHOLD = 150        # 未入力（白い文字）だけを読むしきい値。入力済みのグレー（明るさ90〜100）は消える
FULL_THRESHOLD = 80    # グレーも含めて単語全体を読むしきい値（背景は明るさ30〜50。低すぎると文字が太ってつぶれる）
CROP_THRESHOLD = 65    # 文字の範囲を探すときのしきい値
SCALE = 2              # OCR前の拡大倍率
INVERT = False         # 文字が背景より暗いなら True
KEY_INTERVAL = 0.012   # 1打ごとの間隔(秒)。速すぎてゲームが取りこぼすと「単語帳の単語で止まった」警告が出るので、そのときは上げる（元は 0.035）
STUCK_TIME = 0.5       # 打ってからこの秒数たっても同じ皿なら、白い（未入力の）部分だけを打ち直す
MAX_FIX = 3            # 1皿で打ち直す最大回数（超えたら次の皿まで待つ）
SAME_PLATE = 0.9       # 全体の読み取りがこの類似度以上（かつ長さの差2文字以内）なら同じ皿とみなす（OCRの揺れ対策）
POLL = 0.02            # 画面チェック間隔(秒)
FREEZE_TIME = 5        # この秒数新しい皿が来なければ「終了 or 固まった」とみなし、画面全体を stuck/freeze_*.png に保存
PLATE_WAIT = 0.8       # 新しい皿が出てから打ち始めるまで、毎回この秒数待つ（スループットを下げる。0 で待たない）
GAME_TITLE = "寿司打"    # このウィンドウが前面のときだけ Enter/Space で自動開始＆入力する（空文字ならチェックしない）
# 皿カウントで自動停止はしない（ずれるため）。終わらせるときは F9 で手動停止

ALLOWED = "abcdefghijklmnopqrstuvwxyz-,.!?"
# 英語の辞書補正はローマ字では逆効果なので切る
OCR_CONFIG = (f"--psm 7 -c tessedit_char_whitelist={ALLOWED}"
              " -c load_system_dawg=0 -c load_freq_dawg=0")

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

_pool = ThreadPoolExecutor(max_workers=2)  # 全体と白い部分のOCRを同時に走らせる


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


def snap(sct):
    """(単語全体の画像, 未入力の白い部分の画像, 文字が範囲の端で切れているか) を
    同じ画面・同じ切り出し範囲から作る"""
    gray = capture(sct)
    span = text_span(gray)
    cut = span is not None and (span[0] <= 2 or span[1] >= gray.width - 3)
    return prepare(gray, FULL_THRESHOLD, span), prepare(gray, THRESHOLD, span), cut


def read_imgs(full_img, rest_img):
    f1 = _pool.submit(ocr, full_img)
    f2 = _pool.submit(ocr, rest_img)
    return f1.result(), f2.result()


def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def remaining_part(word, rest):
    """白い部分のOCR結果 rest を、正しい単語 word の末尾に合わせて返す"""
    if not rest:
        return ""
    best, best_score = "", 0.0
    for i in range(len(word)):
        cand = word[i:]
        if abs(len(cand) - len(rest)) > 2:
            continue
        score = similar(cand, rest)
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= 0.5 else ""


# ---- 単語帳 -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORDS_FILE = os.path.join(BASE_DIR, "words.txt")            # 正解が確定した単語（1行1単語）
CANDIDATES_FILE = os.path.join(BASE_DIR, "candidates.txt")  # 単語帳に入る前の候補（「単語 回数」）
UNKNOWN_FILE = os.path.join(BASE_DIR, "unknown.txt")        # ローマ字として読めなかったOCR結果
# 単語帳の似た単語に寄せる基準。ログの実測では、本当の読み違いは正しい単語との類似度 0.919 以上、
# 別々の単語どうしは最大 0.865 だったので、その間に置く
FUZZY_CUTOFF = 0.90
FUZZY_MARGIN = 0.04   # 1位と2位の候補の差がこれ未満なら、どちらか決められないので寄せない
LEARN_CONFIRM = 2     # 未登録の単語が何回ミスなく通ったら単語帳に入れるか（1回だけだと読み違いが紛れ込む）
MIN_LEARN_LEN = 4     # これより短い読み取りは切れ端の可能性が高いので覚えない

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


def load_candidates():
    cands = {}
    if os.path.exists(CANDIDATES_FILE):
        with open(CANDIDATES_FILE, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    cands[parts[0]] = int(parts[1])
    return cands


def save_candidates():
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        for w, n in sorted(CANDIDATES.items()):
            f.write(f"{w} {n}\n")


WORDS = load_words()
CANDIDATES = load_candidates()
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
    #   先頭: 左の皿カウンターが「y,」のように読まれる → 1文字＋記号で始まる部分を落とす
    word = re.sub(r"^[a-z]?[,.]+", "", word)
    #   先頭: 「-」や記号で始まる単語はない
    word = word.lstrip("-,.!?")
    #   末尾: n 以外の子音で終わる単語はない（右の湯のみの縁が y と読まれる対策）
    word = re.sub(r"[bcdfghjkmpqrstvwxyz]+$", "", word)
    return word


def _ambiguous_i(raw):
    """i の前後に l や ! がくっついた読みは、i が1つなのか2つなのか決められない
    （例: kailsisimasu は kaisi が正しかったが、sekalissyuu は sekaii が正しかった）"""
    return bool(re.search(r"[il!][l!]|[l!][il]", raw))


def lookup(full, rest):
    """OCR結果を単語帳で引く。戻り値: (打つ単語, 方法, 候補として記録してよいか)"""
    f_full, f_rest = fix_romaji(full), fix_romaji(rest)
    reads = [r for r in dict.fromkeys((f_full, f_rest)) if len(r) >= 2]
    if not reads:
        return "", "", False
    # 1. 単語帳にそのままある
    for r in reads:
        if r in WORDS:
            return r, "単語帳", False
    # 2. 単語帳に十分似た単語がある（OCRの読み違い・端のゴミ）→ 単語帳の綴りで打つ
    best = (0.0, 0.0, "")
    for r in reads:
        scores = sorted((similar(r, w), w) for w in WORDS if abs(len(w) - len(r)) <= 3)
        if scores:
            top, second = scores[-1][0], (scores[-2][0] if len(scores) > 1 else 0.0)
            if top > best[0]:
                best = (top, second, scores[-1][1])
    if best[0] >= FUZZY_CUTOFF and best[0] - best[1] >= FUZZY_MARGIN:
        return best[2], "単語帳(読み違い補正)", False
    # 3. 単語帳にない → 読んだものをそのまま打つ
    word = f_full if is_romaji(f_full) or not is_romaji(f_rest) else f_rest
    if not is_romaji(word) and word not in _logged_unknown:
        _logged_unknown.add(word)
        with open(UNKNOWN_FILE, "a", encoding="utf-8") as f:
            f.write(word + "\n")
    # 2通りの2値化で同じに読めて、ローマ字として正しく、i の数があいまいでないものだけ記録する
    learnable = (f_full == f_rest and is_romaji(word) and len(word) >= MIN_LEARN_LEN
                 and not _ambiguous_i(full))
    return word, "未登録", learnable


def learn(word):
    """未登録だったがミスなく通った単語を候補に記録し、LEARN_CONFIRM 回で単語帳に昇格させる"""
    if word in WORDS:
        return
    CANDIDATES[word] = CANDIDATES.get(word, 0) + 1
    n = CANDIDATES[word]
    if n < LEARN_CONFIRM:
        save_candidates()
        print(f"  ・候補に記録: {word}（{n}/{LEARN_CONFIRM}）")
        log(f"CANDIDATE {word!r} {n}")
        return
    del CANDIDATES[word]
    save_candidates()
    WORDS.add(word)
    with open(WORDS_FILE, "a", encoding="utf-8") as f:
        f.write(word + "\n")
    print(f"  ＋単語帳に追加: {word}（{len(WORDS)}語）")
    log(f"LEARN {word!r}")


def is_same_plate(word, full, last):
    """今読んだものが、直前に打った皿 last の続きかどうか"""
    if not last:
        return False
    if word == last:
        return True
    if abs(len(word) - len(last)) <= 2 and similar(word, last) >= SAME_PLATE:
        return True
    # 入力済み（グレー）の部分が読めず、白い残りだけが読めた → 前の皿の続き（新しい皿と間違えると、
    # 切れ端を単語として覚えてしまう）。ただし単語帳にある単語なら完全な単語なので新しい皿とみなす
    if word in WORDS:
        return False
    f = fix_romaji(full)
    return any(similar(last[i:], f) >= 0.8
               for i in range(1, len(last)) if abs(len(last) - i - len(f)) <= 2)


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
        full_img, rest_img, cut = snap(sct)
        if cut:
            print("！文字が読み取り範囲の端で切れています。calib で範囲を広げてください")
        full_img.save("debug_full.png")
        rest_img.save("debug.png")
        t = time.time()
        full, rest = read_imgs(full_img, rest_img)
        ms = (time.time() - t) * 1000
        word, how, learnable = lookup(full, rest)
        print("OCR全体  :", repr(full), " 未入力部分:", repr(rest), f"（OCR {ms:.0f}ms）")
        print(f"打つ単語 : {word!r}（{how}{'・候補に記録可' if learnable else ''}） 単語帳: {len(WORDS)}語")
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
    print(f"単語帳: {len(WORDS)}語 / 候補: {len(CANDIDATES)}語")
    # 起動時に押していた Enter（ターミナルでの実行）を開始の合図と誤認しないよう、離すまで待つ
    while keyboard.is_pressed("enter") or keyboard.is_pressed("space"):
        time.sleep(0.05)
    active = False
    last = ""             # 今の皿として打った単語
    last_learnable = False
    last_from_dict = False
    fixes = 0             # 今の皿で未入力部分を打ち直した回数
    stuck_logged = False  # 今の皿で「止まった」を記録済みか
    typed_at = 0.0
    prev_sig = None       # 前回の画面（画像）。同じ画像が2回続いたら読む＝切り替わり途中を読まない
    cache_sig, cache_read = None, ("", "")
    plates = dict_plates = 0
    game_start = plate_at = 0.0  # ゲーム開始時刻 / 最後に新しい皿を打った時刻
    total_keys = 0
    freeze_logged = False
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
                    active, last, fixes, prev_sig = True, "", 0, None
                    plates = dict_plates = total_keys = 0
                    game_start = plate_at = time.time()
                    freeze_logged = False
                    print("開始（F9 で停止）")
            if keyboard.is_pressed("f9") and active:
                active = False
                rate = f"{dict_plates * 100 // plates}%" if plates else "-"
                print(f"停止（{plates} 枚入力、うち単語帳から {dict_plates} 枚 = {rate}）")
                log(f"MANUAL-STOP plates={plates} dict={dict_plates}")
            if not active:
                time.sleep(0.1)
                continue

            if not game_focused():
                # 寿司打以外のウィンドウにキーを送らないよう、前面に戻るまで待つ
                time.sleep(0.1)
                continue

            if plates and not freeze_logged and time.time() - plate_at > max(FREEZE_TIME, PLATE_WAIT + 3):
                # 新しい皿が来なくなった＝ゲームが終わったか固まった。画面全体を保存して原因を調べられるようにする
                freeze_logged = True
                name = time.strftime("freeze_%H%M%S.png")
                os.makedirs(STUCK_DIR, exist_ok=True)
                shot = sct.grab(sct.monitors[0])
                Image.frombytes("RGB", shot.size, shot.rgb).save(os.path.join(STUCK_DIR, name))
                info = (f"plates={plates} keys={total_keys} "
                        f"elapsed={plate_at - game_start:.0f}s last={last!r}")
                print(f"  ！しばらく新しい皿が来ません。画面を stuck/{name} に保存（{info}）")
                log(f"FREEZE {info} screen={name}")

            # 画像が前回と同じ（＝表示が落ち着いている）ときだけOCRする。
            # OCRを2回して比べるより速く、同じ画面を何度もOCRしないで済む
            full_img, rest_img, cut = snap(sct)
            sig = (full_img.size, hash(full_img.tobytes()), hash(rest_img.tobytes()))
            if sig != prev_sig:
                prev_sig = sig
                time.sleep(POLL)
                continue
            if sig != cache_sig:
                cache_sig, cache_read = sig, read_imgs(full_img, rest_img)
            full, rest = cache_read
            if len(fix_romaji(full)) < 2:
                time.sleep(POLL)
                continue

            word, how, learnable = lookup(full, rest)
            if cut:
                # 端で切れた読みは単語の一部が欠けているかもしれないので覚えない
                # （例: 範囲が狭く …kudasai の i が外にはみ出し、…kudasa と読んでいた）
                learnable = False

            if is_same_plate(word, full, last):
                # 打ち終わったのに同じ皿のまま → 白い（未入力の）部分だけ打ち直す
                if time.time() - typed_at < STUCK_TIME:
                    time.sleep(POLL)
                    continue
                # 時間内に通らなかった皿は、そのあと流れて消えても「正しかった」とはみなさない
                last_learnable = False
                if not stuck_logged:
                    stuck_logged = True
                    save_stuck(sct, full, last)
                    if cut:
                        print("  ！文字が読み取り範囲の端で切れています。calib で範囲を広げてください")
                if fixes >= MAX_FIX:
                    time.sleep(POLL)
                    continue
                # 1回目は打った単語の末尾に合わせる（OCRの揺れを吸収）。
                # それでもだめなら打った単語自体が違うので、白い部分の読みをそのまま打つ
                fixed_rest = fix_romaji(rest)
                todo = remaining_part(last, rest) if fixes == 0 else fixed_rest
                if not todo:
                    time.sleep(POLL)
                    continue
                if fixes == 0:
                    if last_from_dict:
                        print(f"  ！単語帳の単語で止まりました: {last}"
                              "（キーの取りこぼし→KEY_INTERVAL を上げる / 単語帳の綴り違い→words.txt を確認）")
                        log(f"DICT-STUCK {last!r}")
                fixes += 1
                print(f"  ※未入力の部分だけ打ち直し({fixes}/{MAX_FIX}): {todo}  (白い部分のOCR: {rest})")
                log(f"fix #{plates} rest={rest!r} todo={todo!r}")
                pyautogui.write(todo, interval=KEY_INTERVAL)
                typed_at = time.time()
                prev_sig = None
                continue

            # 新しい皿 → 前の皿が打ち直しなしで通ったなら、その単語は正しかった
            if last and fixes == 0 and last_learnable:
                learn(last)
            fixes, stuck_logged = 0, False
            plates += 1
            from_dict = how.startswith("単語帳")
            dict_plates += from_dict
            # 皿が出た直後なら全部白いはず。白い部分が明らかに短いとき（途中から開始した等）だけ、
            # 入力済みを除いて打つ（OCRで1〜2文字欠けただけのときに先頭を飛ばさないように）
            fixed_rest = fix_romaji(rest)
            todo = word
            if len(fixed_rest) <= len(word) - 3:
                todo = remaining_part(word, fixed_rest) or word

            # 毎回一律に待ってから打つ（スループットを下げる）
            if PLATE_WAIT > 0:
                time.sleep(PLATE_WAIT)
                if not game_focused():  # 待っている間に別のウィンドウに切り替わった
                    plates -= 1
                    dict_plates -= from_dict
                    prev_sig = None
                    continue

            note = f"  (OCR: {full})" if word != full else ""
            if todo != word:
                note += f"  [入力済みを除く: {todo}]"
            print(f"[{plates:3d}] {'◎' if from_dict else '？'} {word}  {how}{note}")
            log(f"type #{plates} raw={full!r} rest={rest!r} word={word!r} how={how}")
            pyautogui.write(todo, interval=KEY_INTERVAL)
            total_keys += len(todo)
            last, last_learnable, last_from_dict = word, learnable, from_dict
            typed_at = plate_at = time.time()
            prev_sig = None


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"calib": calib, "test": test, "run": run}.get(mode, run)()
