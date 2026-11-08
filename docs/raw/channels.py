#from pathlib import Path
#import fitz

#HERE = Path(__file__).parent
#for f in HERE.glob("*.pdf"):
 #   doc = fitz.open(f)
  #  print(f.name, doc.page_count, "页")
   # print(doc[0].get_text()[:100])   # 只看第一页开头
    #print("-" * 50)
    #doc.close()
import pdfplumber
from pathlib import Path

HERE = Path(__file__).parent
pdf = HERE / "2026河南省普通高校专升本招生录取控制分数.pdf"

with pdfplumber.open(pdf) as pdf_file:
    print("共", len(pdf_file.pages), "页")

    # 抽某一页的文字
    text = pdf_file.pages[0].extract_text()
    print(text)

    # 抽这一页的表格
    table = pdf_file.pages[0].extract_table()
    for row in table:
        print(row)          # ['软件工程', '080902', '12000', '2年']