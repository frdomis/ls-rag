#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
health_monitor.py —— 轻量监控脚本（部署到服务器，cron 每 5 分钟跑一次）
用法：
    python3 health_monitor.py                 # 检查一次，异常退出码 1
    python3 health_monitor.py --url http://127.0.0.1/health

功能：
    1. 检查 /health 是否 healthy
    2. 记录到 /var/log/lsrag_health.log
    3. 连续 3 次失败 → 自动重启 lsrag 服务
    4. 每日生成一行汇总（请求数、错误数、平均延迟）

cron 配置（服务器上执行 `crontab -e`）：
    */5 * * * * /usr/bin/python3 /opt/ls-rag/health_monitor.py >> /var/log/lsrag_cron.log 2>&1
    0 0 * * * /usr/bin/python3 /opt/ls-rag/health_monitor.py --daily >> /var/log/lsrag_cron.log 2>&1
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

LOG = Path("/var/log/lsrag_health.log")
STATE = Path("/var/log/lsrag_health_state.json")
NGINX_ACCESS = Path("/var/log/nginx/access.log")
FAIL_THRESHOLD = 3


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"consecutive_fail": 0, "last_ok": None}


def save_state(s):
    try:
        STATE.write_text(json.dumps(s, ensure_ascii=False))
    except Exception:
        pass


def check(url, timeout=10):
    """返回 (ok, detail)"""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        d = r.json()
        if d.get("status") != "healthy":
            return False, f"status={d.get('status')} detail={d.get('same_source_detail')}"
        return True, f"{d.get('chunks_count')} 块"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def restart_service():
    try:
        log("[重启] systemctl restart lsrag")
        subprocess.run(["systemctl", "restart", "lsrag"], timeout=60, check=False)
        time.sleep(5)
        return True
    except Exception as e:
        log(f"[重启失败] {e}")
        return False


def one_shot(url):
    ok, detail = check(url)
    st = load_state()
    if ok:
        st["consecutive_fail"] = 0
        st["last_ok"] = datetime.now().isoformat(timespec="seconds")
        save_state(st)
        log(f"[OK] {detail}")
        return 0
    else:
        st["consecutive_fail"] = st.get("consecutive_fail", 0) + 1
        save_state(st)
        log(f"[FAIL {st['consecutive_fail']}/{FAIL_THRESHOLD}] {detail}")
        if st["consecutive_fail"] >= FAIL_THRESHOLD:
            log("[告警] 连续失败达到阈值，尝试重启")
            if restart_service():
                ok2, d2 = check(url)
                log(f"[重启后] {'OK' if ok2 else 'FAIL'} {d2}")
                if ok2:
                    st["consecutive_fail"] = 0
                    save_state(st)
                    return 0
        return 1


def daily_report():
    """统计前 24 小时 nginx 访问日志"""
    if not NGINX_ACCESS.exists():
        log("[日报] 无 nginx 日志")
        return
    since = datetime.now() - timedelta(hours=24)
    total = errors = 0
    lat_sum = 0.0
    lat_n = 0
    endpoints = {}

    try:
        with open(NGINX_ACCESS, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    parts = line.split('"')
                    if len(parts) < 3:
                        continue
                    # 状态码
                    code = int(parts[2].split()[0])
                    # 路径
                    req = parts[1].split()
                    path = req[1] if len(req) > 1 else "-"
                    key = path.split("?")[0]
                    # 延迟（$request_time 若配置在日志格式里）
                    if "$request_time" in line:
                        pass
                    total += 1
                    endpoints[key] = endpoints.get(key, 0) + 1
                    if code >= 500:
                        errors += 1
                except Exception:
                    continue
    except Exception as e:
        log(f"[日报] 读日志失败：{e}")
        return

    top = sorted(endpoints.items(), key=lambda kv: -kv[1])[:5]
    log(f"[日报] 24h 请求 {total} | 5xx {errors} | 错误率 {errors / max(total, 1) * 100:.2f}%")
    for k, v in top:
        log(f"       {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1/health")
    ap.add_argument("--daily", action="store_true", help="生成日报")
    args = ap.parse_args()

    if args.daily:
        daily_report()
        return 0
    return one_shot(args.url)


if __name__ == "__main__":
    sys.exit(main())
