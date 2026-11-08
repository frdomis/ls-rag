
# -*- coding: utf-8 -*-
# app/ingest.py  --  W4D2 语料取数管线（HTML 优先，可重跑、可增量）
#
# 取数直线：抓取 -> 正文提取(去导航/页脚) -> 清洗 -> 命名(带 year/source) -> 存 docs/raw/
# 设计目标（对应 W4D2）：
#   - HTML 优先：只抓有文本层的政策/通知/章程页；扫描件 PDF 一律不碰（OCR 时间黑洞）
#   - 可重跑  ：文件名稳定；默认跳过已抓且内容未变的老文件（--force 强制覆盖）
#   - 可增量  ：往 SOURCES 追加一条 URL 即可，不动老数据
#   - 来源登记：每次运行把来源同步进 docs/raw/sources.md（年份 + 来源 + 类型 + URL）

import re
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

import requests

try:
    import trafilatura  # 首选：自动去导航/页脚/广告，只留正文
    HAS_TRAFILATURA = True
except Exception:
    HAS_TRAFILATURA = False

try:
    from bs4 import BeautifulSoup  # 兜底解析 + 标题提取
except Exception:
    BeautifulSoup = None

# ------------------------------------------------------------------
# 1. 来源配置（取数清单）—— 换/加 URL 只改这里，天然支持增量
#    格式：(year, source, doc_type, url)；doc_type: policy/notice/charter
# ------------------------------------------------------------------
SOURCES = [
    # -- 省厅政策 / 通知 --
    ("2026", "河南省教育厅", "policy",
     "https://jyt.henan.gov.cn/2025/10-24/3237885.html"),
    ("2026", "河南省教育厅", "notice",
     "https://jyt.henan.gov.cn/2026/03-02/3329765.html"),
    ("2026", "河南省教育考试院", "notice",
     "https://www.haeea.cn/html/2026/03-02/3329765.html"),

    # -- 公办院校 2026 专升本招生章程 --
    ("2026", "河南理工大学", "charter", "https://www6.hpu.edu.cn/web5/info/10488/93115.htm"),
    ("2026", "河南工业大学", "charter", "https://zs.haut.edu.cn/info/1067/10072.htm"),
    ("2026", "河南财经政法大学", "charter", "https://zs.huel.edu.cn/"),
    ("2026", "河南中医药大学", "charter", "https://ptzs.hactcm.edu.cn/info/2096/14786.htm"),
    ("2026", "华北水利水电大学", "charter", "https://www2.ncwu.edu.cn/zhaoshengwang/info/1015/2905.htm"),
    ("2026", "郑州航空工业管理学院", "charter", "https://zsxxw.zua.edu.cn/info/1035/2211.htm"),
    ("2026", "河南科技学院", "charter", "https://zsxx.hist.edu.cn/"),
    ("2026", "安阳师范学院", "charter", "https://zs.aynu.edu.cn/info/1083/2131.htm"),
    ("2026", "南阳师范学院", "charter", "https://sites.lynu.edu.cn/zsxx/info/1003/2326.htm"),
    ("2026", "洛阳师范学院", "charter", "https://zsb.lynu.edu.cn/"),
    ("2026", "商丘师范学院", "charter", "https://zhaoban.sqnu.edu.cn/info/1044/3536.htm"),
    ("2026", "周口师范学院", "charter", "https://zs.zknu.edu.cn/"),
    ("2026", "许昌学院", "charter", "https://zs.xcu.edu.cn/info/1014/8411.htm"),
    ("2026", "河南城建学院", "charter", "https://zs.huuc.edu.cn/info/1103/4747.htm"),
    ("2026", "洛阳理工学院", "charter", "https://zs.lit.edu.cn/info/1007/4411.htm"),
    ("2026", "河南工程学院", "charter", "https://hauezs.university-hr.com/index.php?sys=home&module=article&param=29&act=view&article_id=932G1QaMGEL6"),
    ("2026", "河南牧业经济学院", "charter", "https://zhaosheng.hnuahe.edu.cn/info/1700/3698.htm"),
    ("2026", "河南工学院", "charter", "https://zsxx.hait.edu.cn/"),
    ("2026", "河南财政金融学院", "charter", "https://zsjyc.hafu.edu.cn/info/1650/2485.htm"),
    ("2026", "黄淮学院", "charter", "https://zhaosheng.huanghuai.edu.cn/bkzn/c9c6d29265f64ba296a758ed59e093dd.htm"),
    ("2026", "平顶山学院", "charter", "https://zsxx.pdsu.edu.cn/info/1059/2735.htm"),
    ("2026", "安阳工学院", "charter", "https://xxgk.ayit.edu.cn/info/2178/2613.htm"),
    ("2026", "南阳理工学院", "charter", "https://zsw.nyist.edu.cn/"),
    ("2026", "新乡学院", "charter", "https://zs.xxu.edu.cn/info/1209/2738.htm"),
    ("2026", "郑州师范学院", "charter", "https://zs.zztc.edu.cn/"),
    ("2026", "信阳农林学院", "charter", "https://www.xyafu.edu.cn/zsxxw/info/1028/2141.htm"),
    ("2026", "郑州工程技术学院", "charter", "https://zs.zzut.edu.cn/info/1133/2780.htm"),

    # -- 民办院校 2026 专升本招生章程 --
    ("2026", "黄河科技学院", "charter", "http://www.hhstu.edu.cn/info/1042/6802.htm"),
    ("2026", "郑州科技学院", "charter", "https://zs.zit.edu.cn/info/1026/2024.htm"),
    ("2026", "郑州工业应用技术学院", "charter", "https://zsb.zzuit.edu.cn/zszc.htm"),
    ("2026", "商丘工学院", "charter", "https://www.sqgxy.edu.cn/zs/main.htm"),
    ("2026", "商丘学院", "charter", "https://www.sqxy.edu.cn/squ_zsb/"),
    ("2026", "郑州升达经贸管理学院", "charter", "https://zs.si.edu.cn/"),
    ("2026", "郑州商学院", "charter", "https://zsw.zbu.edu.cn/1129/list.htm"),
    ("2026", "郑州财经学院", "charter", "https://zb.zzife.edu.cn/2026_01/08_10/content-82070.html"),
    ("2026", "黄河交通学院", "charter", "https://zs.zjtu.edu.cn/"),
    ("2026", "信阳学院", "charter", "https://zs.xyu.edu.cn/info/1067/1931.htm"),
    ("2026", "安阳学院", "charter", "https://www.ayxy.edu.cn/zs/"),
    ("2026", "郑州工商学院", "charter", "https://zsb.ztbu.edu.cn/2026_01/06_08/content-68937.html"),
    ("2026", "郑州西亚斯学院", "charter", "https://zs.sias.edu.cn/"),
    ("2026", "郑州经贸学院", "charter", "https://zsw.zueb.edu.cn/zszc.htm"),
    ("2026", "河南开封科技传媒学院", "charter", "https://www.humc.edu.cn/"),
    ("2026", "中原科技学院", "charter", "https://zs.zykeji.edu.cn/"),
    ("2026", "新乡工程学院", "charter", "https://www.xxgc.edu.cn/zszx/info/1068/1978.htm"),
    ("2026", "豫北医学院", "charter", "https://zs.nhmu.edu.cn/2026/0106/c1230a129535/page.htm"),
    ("2026", "漯河食品工程职业大学", "charter", "https://zs.lhfpc.edu.cn/"),
]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

RAW_DIR = Path("docs/raw")
MIN_CHARS = 200          # 正文 < 200 字视为登录页/空页/抓取失败，跳过
MAX_RETRY = 3
RETRY_WAIT = 2           # 秒
ARGS = argparse.Namespace(force=False)   # 由 __main__ 用真实参数覆盖


# ------------------------------------------------------------------
# 2. 抓取（带重试）
# ------------------------------------------------------------------
def fetch(url):
    for i in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.encoding = r.apparent_encoding
            if r.status_code == 200 and r.text:
                return r.text
            print(f"  [warn] 状态码 {r.status_code}，第 {i}/{MAX_RETRY} 次重试")
        except Exception as e:
            print(f"  [warn] 请求异常：{e}，第 {i}/{MAX_RETRY} 次重试")
        time.sleep(RETRY_WAIT)
    print("  [skip] 抓取失败（可能需登录 / 临时不可达），已放弃")
    return None


# ------------------------------------------------------------------
# 3. 标题提取（稳定命名用，不依赖正文前几十字）
# ------------------------------------------------------------------
def _soup_strip(html):
    """去脚本/样式/导航等噪声后返回 soup。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "iframe", "link", "meta", "form"]):
        tag.decompose()
    return soup


def extract_title(html, hint):
    if BeautifulSoup is None:
        return hint
    soup = _soup_strip(html)
    for sel in ("h1", "title", "h2"):
        node = soup.find(sel)
        if node and node.get_text(strip=True):
            t = re.sub(r"\s*(来源.*|发布时间.*|字号.*|打印.*|分享到.*)$", "",
                       node.get_text(strip=True))
            if len(t) >= 6:
                return t.strip()
    return hint


# ------------------------------------------------------------------
# 4. 正文提取（trafilatura 首选，BS4 选择器兜底）
# ------------------------------------------------------------------
def extract_body(html):
    if HAS_TRAFILATURA:
        text = trafilatura.extract(
            html, include_links=False, include_comments=False, favor_recall=0.4)
        if text and len(text) >= MIN_CHARS:
            return text
    if BeautifulSoup is None:
        return ""
    soup = _soup_strip(html)
    for a in soup.find_all("a"):
        a.decompose()
    for sel in ("div.article", "div.content", "div.TRS_Editor",
                "div.v_news_content", "div#content", "div.article-content"):
        node = soup.select_one(sel)
        if node and len(node.get_text(strip=True)) >= MIN_CHARS:
            return node.get_text("\n", strip=True)
    return soup.get_text("\n", strip=True)


# ------------------------------------------------------------------
# 5. 清洗（去不可见字符、压缩空行、丢弃页脚噪声行）
# ------------------------------------------------------------------
def clean(text):
    text = re.sub(r"\r", "", text)
    text = re.sub("[\ufeff\u200b\u00a0]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    drop_pat = re.compile(
        r"^\s*(上一篇|下一篇|打印|关闭|返回顶部|责任编辑|来源：\s*$"
        r"|分享到.*|版权所有.*|工信部备案.*|公网安备.*)\s*$")
    lines = [ln.rstrip() for ln in text.split("\n") if not drop_pat.match(ln)]
    return "\n".join(lines).strip()


# ------------------------------------------------------------------
# 6. 命名（带 year/source）+ 写盘（front matter 元数据 + MD5 跳过）
# ------------------------------------------------------------------
def safe_name(s: str) -> str:
    s = s.strip()                          # ← 加这行
    return re.sub(r"[^\w\-]", "_", s)


def build_filename(year, source, doc_type, title):
    return f"{year}_{safe_name(source)}_{safe_name(doc_type)}_{safe_name(title.strip())}.md"


def write_raw(year, source, doc_type, url, title, body):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / build_filename(year, source, doc_type, title)
    header = (
        "---\n"
        f"source: {source}\n"
        f"year: {year}\n"
        f"doc_type: {doc_type}\n"
        f"url: {url}\n"
        f"fetched_at: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"char_count: {len(body)}\n"
        "---\n\n"
    )
    payload = header + body + "\n"
    new_hash = hashlib.md5(payload.encode()).hexdigest()
    existed = path.exists()
    if existed and not ARGS.force:
        old = path.read_text(encoding="utf-8")
        if hashlib.md5(old.encode()).hexdigest() == new_hash:
            return path, "skip(未变化)"
    path.write_text(payload, encoding="utf-8")
    return path, ("覆盖" if existed else "新建")


# ------------------------------------------------------------------
# 7. 来源清单 sources.md（登记本轮成功入库的源）
# ------------------------------------------------------------------
def update_sources_md(records):
    md = [
        "# 语料来源清单 sources.md\n",
        "> 由 app/ingest.py 自动维护；手动新增源请同步改 SOURCES。\n",
        "| 年份 | 来源 | 文件类型 | 获取方式 | URL |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        md.append(f"| {r['year']} | {r['source']} | {r['doc_type']} | "
                  f"ingest.py 抓取 | {r['url']} |")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "sources.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# 8. 主流程
# ------------------------------------------------------------------
def main():
    records = []
    ok = 0
    skip = 0
    for year, source, doc_type, url in SOURCES:
        print(f"[fetch] {year} {source} :: {url}")
        html = fetch(url)
        if not html:
            skip += 1
            continue

        title = extract_title(html, source)
        body = clean(extract_body(html))

        if len(body) < MIN_CHARS:
            print(f"  [skip] 正文仅 {len(body)} 字（< {MIN_CHARS}），疑似登录页/空页，放弃")
            skip += 1
            continue

        path, status = write_raw(year, source, doc_type, url, title, body)
        print(f"  [{status}] {path.name}  正文 {len(body)} 字")
        records.append(dict(year=year, source=source, doc_type=doc_type,
                            url=url, body=body))
        ok += 1

    if records:
        update_sources_md(records)

    total_words = sum(len(re.sub(r"\s", "", r["body"])) for r in records)
    print("\n" + "=" * 50)
    print(f"完成：成功 {ok} / {len(SOURCES)} 个源，本轮入库正文约 {total_words} 字")
    print(f"跳过 {skip} 个（抓取失败 / 正文过短）")
    print(f"产物目录：{RAW_DIR.resolve()}")
    print(f"来源清单：{(RAW_DIR / 'sources.md').resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="W4D2 语料取数管线")
    ap.add_argument("--force", action="store_true", help="强制覆盖已存在的文件")
    ARGS = ap.parse_args()
    main()