# W5D2 元数据过滤对比

- BM25：不过滤 13/33，过滤 22/33
- 向量：不过滤 20/33，过滤 24/33
- 兜底触发 0 次

| 题号 | 问题 | must | BM25 | 向量 |
|---|---|---|---|---|
| Q1 | 2026年河南专升本报名截止时间是什么时候？ | [{'key': 'year', 'value': '2026'}, {'key': 'doc_type', 'value': 'policy'}] | True→True | True→True |
| Q2 | 退役大学生士兵专升本怎么报名？ | [{'key': 'doc_type', 'value': 'policy'}] | True→True | True→True |
| Q3 | 河南专升本考试考哪几科？ | [{'key': 'doc_type', 'value': 'policy'}] | False→True | False→True |
| Q4 | 专升本考试成绩什么时候公布？ | [{'key': 'doc_type', 'value': 'policy'}] | False→True | False→True |
| Q5 | 志愿填报的时间安排是怎样的？ | [{'key': 'doc_type', 'value': 'policy'}] | True→True | True→True |
| Q6 | 专升本能报几个志愿？怎么录取的？ | [{'key': 'doc_type', 'value': 'policy'}] | False→True | False→True |
| Q7 | 2026年河南专升本录取控制分数线是多少？ | [{'key': 'year', 'value': '2026'}, {'key': 'doc_type', 'value': 'score'}] | False→True | True→True |
| Q8 | 计算机科学与技术专业的录取分数线是多少？ | [{'key': 'doc_type', 'value': 'score'}] | True→True | True→True |
| Q9 | 征集志愿是什么意思？什么时候征集？ | [{'key': 'doc_type', 'value': 'policy'}] | False→True | True→True |
| Q10 | 哪些院校参加了征集志愿？ | [{'key': 'doc_type', 'value': 'policy'}] | False→True | True→True |
| Q11 | 2026年报名时间？ | [{'key': 'year', 'value': '2026'}, {'key': 'doc_type', 'value': 'policy'}] | True→True | True→True |
| Q14 | 考试前有什么需要注意的？ | [{'key': 'doc_type', 'value': 'policy'}] | False→True | True→True |
| Q15 | 怎么查各院校的招生章程？ | [{'key': 'doc_type', 'value': 'charter'}] | False→False | False→False |
| Q16 | 征集志愿和第一次填报的志愿有什么区别？ | [{'key': 'doc_type', 'value': 'policy'}] | True→True | True→False |
| Q17 | 专升本考上后学制几年？ | [{'key': 'doc_type', 'value': 'policy'}] | False→False | False→False |
| Q18 | 有没有官方的政策文件链接？ | [{'key': 'doc_type', 'value': 'policy'}] | False→False | False→False |
| Q19 | 我大一现在能报名吗？ | [{'key': 'doc_type', 'value': 'policy'}] | True→True | True→True |
| Q20 | 上面说的分数线是指最低分还是平均分？ | [{'key': 'doc_type', 'value': 'score'}] | False→True | True→True |
| Q21 | 我是27届的，大二现在准备专升本晚不晚啊 | - | False→False | True→True |
| Q22 | 专科挂科了还能报名专升本吗 | [{'key': 'doc_type', 'value': 'policy'}] | True→True | True→True |
| Q23 | 河南专升本一辈子就一次机会是真的吗 | - | True→True | True→True |
| Q24 | 我能跨专业考汉语言文学吗，专科是电子商务 | - | False→False | False→False |
| Q25 | 医学类是不是完全不能跨专业 | - | False→False | False→False |
| Q26 | 退役士兵免试还要考试吗 | [{'key': 'doc_type', 'value': 'policy'}] | True→True | True→True |
| Q27 | 建档立卡专项计划是啥，我能报吗 | [{'key': 'doc_type', 'value': 'policy'}] | True→True | True→True |
| Q28 | 专升本报名一共要花多少钱 | [{'key': 'doc_type', 'value': 'policy'}] | True→True | False→True |
| Q30 | 成考大专能不能考统招专升本 | - | False→False | True→True |
| Q31 | 2026年专升本分数线什么时候出来 | [{'key': 'year', 'value': '2026'}, {'key': 'doc_type', 'value': 'score'}] | False→True | False→True |
| Q33 | 征集志愿是啥时候填，我滑档了还有救吗 | [{'key': 'doc_type', 'value': 'policy'}] | False→False | False→False |
| Q36 | 上面说的分数线是指最低分还是平均分 | [{'key': 'doc_type', 'value': 'score'}] | False→True | True→True |
| Q37 | 专升本考上后去本科学校读几年 | - | False→False | True→True |
| Q38 | 学费和专科一样吗？民办是不是很贵 | [{'key': 'doc_type', 'value': 'policy'}] | False→False | False→False |
| Q39 | 有没有官方的政策文件链接发我 | [{'key': 'doc_type', 'value': 'policy'}] | True→False | False→False |
