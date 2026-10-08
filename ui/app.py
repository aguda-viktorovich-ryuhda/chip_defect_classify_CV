# -*- coding: utf-8 -*-
"""app.py —— 芯片质检（焊点与飞线检查）：本地浏览器界面（交付件）

只用 Python 标准库 + 现有依赖（numpy/opencv/torch 仅在子进程里被用到）：
  · 页面与样式全部内嵌，离线可用，不拉任何外部资源
  · 起一个只监听本机 127.0.0.1 的服务，浏览器打开即用
  · 真正的分析交给 run_pipeline.py（界面不碰算法）

启动：
  python ui/app.py            # 默认 8000 端口
  python ui/app.py --port 8080

两种使用方式（页面里可切换）：
  ① 选本机文件夹：直接填服务器上图片目录的绝对路径（当前可用）
  ② 上传图片：在页面里选若干图片上传，按任务名（默认按时间）建目录后分析
                —— 上传路径依赖"输入通用化"改造（见 STATUS.md 交接块），
                   改造完成前上传会提示未就绪。

接口：
  GET  /                 页面
  GET  /health           健康检查
  GET  /status?run=…     运行进度
  GET  /result?run=…     结果摘要与文件清单
  GET  /files/<run>/<相对路径>  取产物（报告图/裁图/CSV）
  POST /run              发起分析（folder 或 upload 两种）
"""
import argparse
import html
import json
import os
import re
import sys
import threading
import time
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)          # chip_defect_classify/
def _pick_python():
    """挑一个装了依赖的解释器：CHIP_PY 环境变量 → 当前解释器 → 系统 python3/python。

    只认"能 import numpy/cv2/torch 成功"的解释器；都不行时退回当前解释器并给出提示。
    """
    import subprocess as _sp
    import shutil as _sh
    cands = []
    if os.environ.get("CHIP_PY"):
        cands.append(os.environ["CHIP_PY"])
    cands.append(sys.executable)
    for _n in ("python3", "python"):
        _p = _sh.which(_n)
        if _p:
            cands.append(_p)
    for c in cands:
        if c and os.path.exists(c):
            try:
                if _sp.run([c, "-c", "import numpy,cv2,torch"], capture_output=True).returncode == 0:
                    return c
            except Exception:
                pass
    print("[WARN] 没有找到同时装有 numpy/opencv-contrib-python/torch 的解释器，"
          "将用当前解释器启动（可能报缺模块）。可用 CHIP_PY 或 本地配置.env 指定。")
    return sys.executable


PY = _pick_python()
RUNS = {}                            # run_id -> {state, msg, dir, task, t0}
LOCK = threading.Lock()

# 页面"本机文件夹"的默认值：Linux 保持原交付路径，Windows 指向项目自带样本目录
if os.name == "nt":
    SAMPLE_DEFAULT = os.path.join(APP, "chip_samples__20260128")
else:
    SAMPLE_DEFAULT = "./chip_samples__20260128"

PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>芯片质检 · 焊点与飞线检查</title>
<style>body{font-family:sans-serif;margin:24px;max-width:1000px}
h1{font-size:20px}fieldset{margin:12px 0;border:1px solid #bbb;border-radius:6px}
button{padding:8px 16px;font-size:15px}input[type=text]{width:60%;padding:6px}
table{border-collapse:collapse;margin-top:10px}td,th{border:1px solid #ccc;padding:4px 8px;font-size:14px}
.msg{color:#444;white-space:pre-wrap;font-family:monospace;font-size:13px}
img{max-width:100%;border:1px solid #ccc;margin-top:8px}</style></head><body>
<h1>芯片质检 · 焊点与飞线检查</h1>
<fieldset><legend>① 输入</legend>
<input type="text" id="folder" placeholder="例如 /path/to/images" value="__SAMPLE_DIR__">
<label><input type="radio" name="mode" value="folder" checked> 本机文件夹</label>
<br><br>
<label><input type="radio" name="mode" value="upload"> 上传图片</label>
<input type="file" id="files" multiple accept="image/*">
<br><br>
任务名：<input type="text" id="task" placeholder="留空则按时间生成">
</fieldset>
<fieldset><legend>② 运行</legend>
<button onclick="start()">开始分析</button>
<span class="msg" id="status">未开始</span></fieldset>
<fieldset><legend>③ 结果</legend>
<div id="summary"></div><div id="filelist"></div></fieldset>
<script>
function jreq(url,opts){
  return new Promise(function(resolve,reject){
    if(window.fetch){
      fetch(url,opts).then(function(r){
        return r.text().then(function(t){
          if(!r.ok){reject(new Error('HTTP '+r.status+' '+t.slice(0,200)));return;}
          try{resolve(JSON.parse(t));}catch(e){reject(new Error('返回内容不是 JSON：'+t.slice(0,200)));}
        });
      },function(e){reject(new Error('网络请求失败：'+(e&&e.message?e.message:e)));});
      return;
    }
    var x=new XMLHttpRequest();
    x.open((opts&&opts.method)||'GET',url,true);
    if(opts&&opts.headers){for(var k in opts.headers)x.setRequestHeader(k,opts.headers[k]);}
    x.onload=function(){
      if(x.status<200||x.status>=300){reject(new Error('HTTP '+x.status+' '+x.responseText.slice(0,200)));return;}
      try{resolve(JSON.parse(x.responseText));}catch(e){reject(new Error('返回内容不是 JSON：'+x.responseText.slice(0,200)));}
    };
    x.onerror=function(){reject(new Error('网络请求失败（服务是否还在运行？）'));};
    x.send((opts&&opts.body)||null);
  });
}
async function start(){
  const mode=document.querySelector('input[name=mode]:checked').value;
  const task=document.getElementById('task').value;
  const st=document.getElementById('status'); st.textContent='提交中…';
  document.getElementById('summary').innerHTML=''; document.getElementById('filelist').innerHTML='';
  try{
    let j;
    if(mode==='folder'){
      j=await jreq('/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({mode:'folder',folder:document.getElementById('folder').value,task:task})});
    }else{
      const fs=document.getElementById('files').files;
      if(!fs.length){st.textContent='请先选择图片';return;}
      const fd=new FormData(); for(const f of fs) fd.append('files',f);
      if(task) fd.append('task',task);
      st.textContent='上传 '+fs.length+' 张…';
      j=await jreq('/run',{method:'POST',body:fd});
    }
    if(!j.ok){st.textContent='失败：'+(j.error||'未知');return;}
    poll(j.run);
  }catch(e){
    st.textContent='提交失败：'+(e&&e.message?e.message:e);
  }
}
async function poll(run){
  const st=document.getElementById('status');
  let done=false;
  const t=setInterval(tick,2000);
  async function tick(){
    if(done)return;
    try{
      const j=await jreq('/status?run='+encodeURIComponent(run));
      st.textContent=j.state+(j.msg?('\\n'+j.msg):'');
      if(j.state==='完成'||j.state==='失败'){done=true;clearInterval(t);show(run);}
    }catch(e){
      st.textContent='读取进度失败：'+(e&&e.message?e.message:e);
    }
  }
  tick();
}
async function show(run){
  let j;
  try{ j=await jreq('/result?run='+encodeURIComponent(run)); }
  catch(e){ document.getElementById('summary').textContent='读取结果失败：'+(e&&e.message?e.message:e); return; }
  if(!j.ok){document.getElementById('summary').textContent=JSON.stringify(j);return;}
  let h='<table><tr><th>项目</th><th>数值</th></tr>';
  for(const k in j.summary) h+='<tr><td>'+k+'</td><td>'+j.summary[k]+'</td></tr>';
  h+='</table>';
  if(j.rows && j.rows.length){
    h+='<table><tr><th>产品</th><th>结论</th><th>原因</th></tr>';
    for(const r of j.rows){
      const col=(r['结论']==='合格')?'#0a0':'#c00';
      h+='<tr><td>'+r['产品']+'</td><td style="color:'+col+';font-weight:bold">'+r['结论']+'</td><td>'+(r['原因']||'')+'</td></tr>';
    }
    h+='</table>';
  }
  document.getElementById('summary').innerHTML=h;
  let f='';
  for(const it of j.files) f+='<div><a href="/files/'+encodeURIComponent(run)+'/'+encodeURIComponent(it.rel)+'" target="_blank">'+it.rel+'</a> ('+it.kb+' KB)</div>';
  const img=j.files.find(x=>x.rel.indexOf('产品结果/')>=0)||j.files.find(x=>x.rel.endsWith('.png'));
  if(img) f+='<img src="/files/'+encodeURIComponent(run)+'/'+encodeURIComponent(img.rel)+'">';
  document.getElementById('filelist').innerHTML=f;
}
</script></body></html>"""


def _latest_run(out_root):
    if not os.path.isdir(out_root):
        return None
    ds = [os.path.join(out_root, d) for d in os.listdir(out_root) if d.endswith('_run')]
    ds = [d for d in ds if os.path.isdir(d)]
    return max(ds, key=os.path.getmtime) if ds else None


def _collect_result(run_dir):
    """汇总结果：以【产品结论 CSV + 逐产品结果图】为准（新版口径）。"""
    out = {"ok": True, "summary": {}, "rows": [], "files": []}
    rep = os.path.join(run_dir, "state", "report")
    csv_p = os.path.join(rep, "产品结论.csv")
    if os.path.exists(csv_p):
        import csv as _csv
        with open(csv_p, encoding="utf-8-sig") as fh:
            rd = list(_csv.DictReader(fh))
        out["rows"] = rd
        n = len(rd); bad = sum(1 for r in rd if r.get("结论") != "合格")
        out["summary"] = {"产品数": n, "合格": n - bad, "不合格": bad}
        out["files"].append({"rel": "state/report/产品结论.csv", "kb": os.path.getsize(csv_p) // 1024})
    pd_ = os.path.join(rep, "产品结果")
    if os.path.isdir(pd_):
        out["summary"]["结果图"] = len(os.listdir(pd_))
        for f in sorted(os.listdir(pd_)):
            if f.endswith(".png"):
                out["files"].append({"rel": "state/report/产品结果/" + f,
                                     "kb": os.path.getsize(os.path.join(pd_, f)) // 1024})
    if os.path.isdir(rep):
        for f in sorted(os.listdir(rep)):
            if f.startswith("裸焊盘_") and f.endswith(".png"):
                out["files"].append({"rel": "state/report/" + f, "kb": os.path.getsize(os.path.join(rep, f)) // 1024})
    sd = os.path.join(run_dir, "state", "stagefig")
    if os.path.isdir(sd):
        for f in sorted(os.listdir(sd)):
            if f.endswith(".png"):
                out["files"].append({"rel": "state/stagefig/" + f, "kb": os.path.getsize(os.path.join(sd, f)) // 1024})
    if not out["rows"]:
        out["summary"]["提示"] = "尚无产品结论（流程可能未跑完，见运行日志）"
    return out


def _task_out_root(task):
    """每个任务一个输出根，避免读到别的任务的产物。"""
    safe = re.sub(r"[^\w\u4e00-\u9fa5.\-]", "_", task) or "task"
    d = os.path.join(APP, "outputs", "_tasks", safe)
    os.makedirs(d, exist_ok=True)
    return d


def _run_pipeline(task, src_dir, note):
    out_root = _task_out_root(task)
    cmd = [PY, os.path.join(APP, "run_pipeline.py"), "--input", src_dir, "--out", out_root]
    with LOCK:
        RUNS[task]["state"] = "运行中"
        RUNS[task]["msg"] = note + "\n" + " ".join(cmd)
    import subprocess
    p = subprocess.Popen(cmd, cwd=APP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        with LOCK:
            RUNS[task]["msg"] = (RUNS[task]["msg"] + "\n" + line.rstrip())[-4000:]
    rc = p.wait()
    run_dir = _latest_run(out_root) or _latest_run(os.path.join(APP, "outputs"))
    with LOCK:
        RUNS[task].update({"state": "完成" if rc == 0 else "失败",
                           "dir": run_dir, "rc": rc, "t1": time.time()})


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            page = PAGE.replace("__SAMPLE_DIR__", html.escape(SAMPLE_DEFAULT, quote=True))
            return self._send(200, page, "text/html; charset=utf-8")
        if u.path == "/health":
            return self._send(200, json.dumps({"ok": True, "app": APP, "python": PY}))
        if u.path == "/status":
            run = (q.get("run") or [""])[0]
            return self._send(200, json.dumps(RUNS.get(run, {"state": "未知", "msg": ""}), ensure_ascii=False))
        if u.path == "/result":
            run = (q.get("run") or [""])[0]
            d = RUNS.get(run, {}).get("dir")
            if not d:
                return self._send(200, json.dumps({"ok": False, "error": "该运行还没有产物"}, ensure_ascii=False))
            return self._send(200, json.dumps(_collect_result(d), ensure_ascii=False))
        if u.path.startswith("/files/"):
            rest = unquote(u.path[len("/files/"):])
            run, _, rel = rest.partition("/")
            d = RUNS.get(run, {}).get("dir") or run
            p = os.path.normpath(os.path.join(d, rel))
            if not p.startswith(os.path.abspath(d)) or not os.path.exists(p):
                return self._send(404, json.dumps({"ok": False, "error": "文件不存在"}))
            ct = "image/png" if p.endswith(".png") else "application/octet-stream"
            return self._send(200, open(p, "rb").read(), ct)
        return self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if urlparse(self.path).path != "/run":
            return self._send(404, "not found", "text/plain; charset=utf-8")
        n = int(self.headers.get("Content-Length") or 0)
        ct = self.headers.get("Content-Type") or ""
        raw = self.rfile.read(n)
        if ct.startswith("application/json"):
            d = json.loads(raw.decode("utf-8"))
            folder = (d.get("folder") or "").strip()
            task = (d.get("task") or "").strip() or ("任务_" + time.strftime("%Y%m%d_%H%M%S"))
            if not folder or not os.path.isdir(folder):
                return self._send(200, json.dumps({"ok": False, "error": "文件夹不存在：" + folder}, ensure_ascii=False))
            with LOCK:
                RUNS[task] = {"state": "已提交", "msg": "输入 " + folder, "dir": None, "t0": time.time()}
            threading.Thread(target=_run_pipeline, args=(task, folder, "本机文件夹模式"), daemon=True).start()
            return self._send(200, json.dumps({"ok": True, "run": task}, ensure_ascii=False))
        if ct.startswith("multipart/form-data"):
            # 手写边界切分：不依赖 email 解析，中文文件名也稳
            _m = re.search(r'boundary=([^;]+)', ct)
            _bd = ('--' + _m.group(1).strip('"')).encode() if _m else b''
            _parts = [p for p in raw.split(_bd) if b'\r\n\r\n' in p]
            task = "任务_" + time.strftime("%Y%m%d_%H%M%S")
            up = os.path.join(APP, "outputs", "_uploads", task)
            os.makedirs(up, exist_ok=True)
            n_saved = 0
            for _p in _parts:
                _head, _, _body = _p.partition(b'\r\n\r\n')
                _body = _body[:-2] if _body.endswith(b'\r\n') else _body
                _hs = _head.decode('utf-8', 'ignore')
                _nm = re.search(r'name="([^"]*)"', _hs)
                _fld = _nm.group(1) if _nm else ''
                _fn_m = re.search(r'filename="([^"]*)"', _hs)
                if not _fn_m:
                    if _fld == 'task':
                        task = _body.decode('utf-8', 'ignore').strip() or task
                    continue
                _fn = _fn_m.group(1)
                safe = re.sub(r"[^\w.\-]", "_", os.path.basename(_fn)) or 'upload.jpg'
                open(os.path.join(up, safe), "wb").write(_body)
                n_saved += 1
            if not n_saved:
                return self._send(200, json.dumps({"ok": False, "error": "没收到图片"}, ensure_ascii=False))
            with LOCK:
                RUNS[task] = {"state": "已上传", "msg": "{} 张 -> {}".format(n_saved, up),
                              "dir": None, "t0": time.time(),
                              "warning": "上传模式的端到端分析依赖【输入通用化】改造（见 STATUS.md 交接块）"}
            threading.Thread(target=_run_pipeline, args=(task, up, "上传模式：{} 张图".format(n_saved)),
                             daemon=True).start()
            return self._send(200, json.dumps({"ok": True, "run": task, "saved": n_saved}, ensure_ascii=False))
        return self._send(200, json.dumps({"ok": False, "error": "不支持的请求类型"}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true",
                    help="服务就绪后自动用默认浏览器打开页面（Windows 启动器使用）")
    a = ap.parse_args()
    try:
        import numpy, cv2, torch  # noqa: F401
        print("依赖检查 ✓ numpy / opencv / torch")
    except Exception as e:
        print("依赖检查 ✗ {}——请用装好依赖的解释器启动，例如：\n"
              "  <python-env>/bin/python ui/app.py --port 8000".format(e))
    url = "http://{}:{}".format("127.0.0.1" if a.host in ("0.0.0.0", "") else a.host, a.port)
    print("芯片质检（焊点与飞线检查）· 本地界面")
    print("  软件目录：", APP)
    print("  打开浏览器访问： " + url)
    print("  关闭：在本窗口按 Ctrl+C")
    srv = ThreadingHTTPServer((a.host, a.port), H)
    if a.open:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
