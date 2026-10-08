## run_pipeline.py （194 行，命中 37 行）
  L55   run_s1_s3              md = os.path.join(run, 'masks')
  L57   run_s1_s3              os.makedirs(os.path.join(md, k), exist_ok=True)
  L60   run_s1_s3              g0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  L65   run_s1_s3              grp, tail = _split_board(name)
  L67   run_s1_s3              r = dict(r); r['sample'] = grp + '/' + tail   # S4/直度报告用【组名/尾4位】
  L68   run_s1_s3              r['sample_alt'] = grp + '-' + tail
  L70   run_s1_s3              base = grp + '_' + tail + '.png'                  # 掩膜文件名用【组名_尾4位】（S4 的 load 约定）
  L73   run_s1_s3              cv2.imwrite(os.path.join(md, k, base), (np.asarray(m) > 0).astype('uint8') * 255)
  L75   run_s1_s3              json.dump({'rows': rows_all}, open(os.path.join(md, 'solder_table.json'), 'w', encoding='utf-8'),
  L78   run_s1_s3              return os.path.join(md, 'solder_table.json')
  L84   main                   ap.add_argument('--out', default=os.path.join(HERE, 'outputs'))
  L85   main                   ap.add_argument('--cfg', default=os.path.join(HERE, 'cfg', 'default.json'))
  L90   main                   run = os.path.join(os.path.abspath(a.out), time.strftime('%Y%m%d_%H%M%S') + '_run')
  L92   main                   os.makedirs(os.path.join(run, d), exist_ok=True)
  L93   main                   log = open(os.path.join(run, 'run.log'), 'w', encoding='utf-8')
  L102  main                   _stub = os.path.join(run, 'inputs', 'annot', 'auto')
  L106  main                   _man.append({'id': _i, 'group': _grp, 'tail': _tail, 'name': _nm, 'path': os.path.abspath(_path)})
  L107  main                   _ph = os.path.join(_stub, _grp, _tail + '_m.png')
  L112  main                   _cv.imwrite(_ph, _img)
  L115  main                   open(os.path.join(run, 'inputs', '输入清单.json'), 'w', encoding='utf-8'),
  L117  main                   say('输入清单 %d 张 -> %s' % (len(_man), os.path.join(run, 'inputs', '输入清单.json')))
  L119  main                   _s4src = os.path.join(run, 'inputs', 'src')
  L122  main                   _d = os.path.join(_s4src, _it['group'])
  L124  main                   _dst = os.path.join(_d, _it['tail'] + '.jpg')
  L127  main                   _mapping[_it['tail']] = _it['path']
  L128  main                   json.dump(_mapping, open(os.path.join(run, 'inputs', '尾号到原图.json'), 'w', encoding='utf-8'),
  L141  main                   r4 = s4_classify.run(conf, run, os.path.join(run, 'inputs', 'src'),
  L143  main                   annot_dir=os.path.join(run, 'inputs', 'annot', 'auto'))
  L154  main                   _bp_res = BP.build(run, os.path.join(run, "masks", "solder_table.json"))
  L163  main                   say('[报告图] %s' % s_report.build(run, src, os.path.join(run, 'state', 'report', '全局判定报告.png')))
  L175  _stagefig              for grp in sorted(os.listdir(src)):
  L176  _stagefig              d = os.path.join(src, grp)
  L179  _stagefig              for f in sorted(os.listdir(d)):
  L181  _stagefig              key = os.path.splitext(f)[0]
  L182  _stagefig              tail = key.split('_')[-1][-4:]
  L183  _stagefig              boards.append(grp + "/" + tail)
  L188  _stagefig              say("  阶段图已出 -> " + os.path.join(run, "state", "stagefig"))

## core/s4_classify.py （58 行，命中 13 行）
  L15   <模块级>                  ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  L16   <模块级>                  _V = os.path.join(ROOT, 'core', 'vendor')
  L28   _install_on_the_fly_mask base = os.path.basename(rel)[:-6] if os.path.basename(rel).endswith('_m.png') else os.path.splitext(os.path.basename(rel))[0]
  L29   _install_on_the_fly_mask key = base[-4:]
  L31   _install_on_the_fly_mask grp = os.path.dirname(rel)
  L32   _install_on_the_fly_mask d = os.path.join(src_dir, grp)
  L33   _install_on_the_fly_mask cand = [f for f in os.listdir(d) if os.path.splitext(f)[0][-4:] == key]
  L35   _install_on_the_fly_mask g = _mask.read_gray(os.path.join(d, cand[0]))
  L47   run                    S.UN = run_dir if os.path.isdir(os.path.join(run_dir, 'TTA')) else os.path.join(run_dir, 'masks')
  L48   run                    os.makedirs(os.path.join(S.UN, 'reports'), exist_ok=True)
  L49   run                    S.SRC = src_dir
  L51   run                    S.SOLD = sold_json or os.path.join(ROOT, 'refs', 'solder_table.json')
  L52   run                    S.ROOT = annot_dir or os.path.join(ROOT, 'refs', 'annot')

## core/s5_straightness.py （97 行，命中 18 行）
  L16   <模块级>                  ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  L17   <模块级>                  _V = os.path.join(ROOT, 'core', 'vendor')
  L34   run                    UN = os.path.join(run_dir, 'masks') if os.path.isdir(os.path.join(run_dir, 'masks')) else run_dir
  L35   run                    os.makedirs(os.path.join(UN, 'reports'), exist_ok=True)
  L36   run                    os.makedirs(os.path.join(run_dir, 'state', 'wires'), exist_ok=True)
  L38   run                    SRC=src_dir, UN=UN, PAN=run_dir, OUT=run_dir,
  L39   run                    SOLD=solder_json or os.path.join(ROOT, 'refs', 'solder_table.json'),
  L42   run                    BOARD_FILE=dict(
  L46   run                    ROOT=(os.path.join(run_dir, 'inputs', 'annot', 'auto') if os.path.isdir(os.path.join(run_dir, 'inputs', 'annot', 'auto')) else os.
  L48   run                    BOARDS=[os.path.basename(os.path.dirname(_pth)) + '/' + os.path.splitext(os.path.basename(_pth))[0][-4:]
  L59   run                    base = os.path.basename(rel)[:-6] if os.path.basename(rel).endswith('_m.png') else os.path.splitext(os.path.basename(rel))[0]
  L62   run                    d = os.path.join(src_dir, os.path.dirname(rel))
  L63   run                    cand = [f for f in os.listdir(d) if os.path.splitext(f)[0][-4:] == k]
  L65   run                    g = _mask.read_gray(os.path.join(d, cand[0])); _cache[k] = _mask.board_mask(g, C)[0]
  L75   run                    _pb_fn = _pb_grp + '_' + _pb_tail + '.png'
  L76   run                    got = m255.load(m255.ALL['TTA'], _pb_fn)
  L78   run                    raise RuntimeError('探针失败：load() 读不到 pred_masks_tta/{}（检查 UN/目录名 ✓）'.format(_pb_fn))
  L94   run                    open(os.path.join(run_dir, 's5_step_errors.log'), 'a', encoding='utf-8').write(

## core/vendor/s4_core.py （202 行，命中 25 行）
  L7    <模块级>                  import glob
  L21   load                   p = os.path.join(UN, "pred_masks_chan9", k[5:] + "_" + fn) if k.startswith("chan:") else os.path.join(UN, k, fn)
  L22   load                   v = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
  L43   <模块级>                  SRC = "chip_samples__20260128"
  L44   <模块级>                  UN = "masks_from_s3"          # S3 现场产出的掩膜目录（由 s4_classify 注入真实路径）
  L46   <模块级>                  SOLD = "refs/solder_table.json"  # 焊点表（由 s4_classify 指向现行 246 点表）
  L51   run_reference          rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"]
  L60   run_reference          for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
  L61   run_reference          grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
  L62   run_reference          key = grp + "/" + base[-4:]; name = grp + "-" + base[-4:]
  L63   run_reference          fn = grp + "_" + base[-4:] + ".png"
  L64   run_reference          g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE)
  L67   run_reference          m = load(ALL[nm], fn)
  L76   run_reference          P_pts = pts.get(name, [])
  L80   run_reference          m4 = P.read_archive_mask(os.path.join(grp, base + ".jpg"))
  L82   run_reference          for q, (x, y) in enumerate(P_pts):
  L93   run_reference          own = [q for q, (x, y) in enumerate(P_pts)
  L105  run_reference          x, y = P_pts[q]
  L125  run_reference          for q, (x, y) in enumerate(P_pts):
  L140  run_reference          px, py = P_pts[q]
  L150  run_reference          w["cov"] = [bool(m[int(round(P_pts[q][1])), int(round(P_pts[q][0]))]) for q in w["pts"]]
  L159  run_reference          for (x, y) in P_pts:
  L192  run_reference          od = os.path.join(PAN, "state", "registry"); os.makedirs(od, exist_ok=True)
  L193  run_reference          fn2 = os.path.join(od, "总名单分类.png")
  L196  run_reference          json.dump(summary, open(os.path.join(UN, "reports/registry.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

## core/vendor/s5_steps/s255_straightness.py （418 行，命中 41 行）
  L3    <模块级>                  import glob
  L63   <模块级>                  SRC = os.path.join(HERE, "chip_samples__20260128")   # 输入（运行时由驱动器覆盖为 --input）
  L64   <模块级>                  UN  = os.path.join(HERE, "outputs", "_masks")        # 掩膜目录（运行时覆盖）
  L65   <模块级>                  PAN = os.path.join(HERE, "outputs")                  # 出图目录（运行时覆盖）
  L66   <模块级>                  SOLD = os.path.join(HERE, "refs", "solder_table.json")
  L67   <模块级>                  BOARD_FILE = os.path.join(HERE, "refs", "boards.json")
  L68   <模块级>                  BOARDS = []
  L72   <模块级>                  ROOT = os.path.join(HERE, "refs", "annot")      # 板子清单驱动器（*_m.png 枚举；无标注时由 run_pipeline 生成占位）
  L73   <模块级>                  ANN  = ROOT
  L74   <模块级>                  MASKDIR = os.path.join(HERE, "outputs", "_masks")
  L79   <模块级>                  sys.path.insert(0, os.path.join(HERE, "lib"))
  L93   <模块级>                  NG_BOARDS = {"样例板A", "样例板B", "样例板C", "样例板G", "样例板H"}
  L97   load                   p = os.path.join(UN, "pred_masks_chan9", k[5:] + "_" + fn) if k.startswith("chan:") else os.path.join(UN, k, fn)
  L98   load                   v = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
  L221  main                   rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"]
  L226  main                   for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
  L227  main                   grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
  L228  main                   key = grp + "/" + base[-4:]; name = grp + "-" + base[-4:]
  L229  main                   fn = grp + "_" + base[-4:] + ".png"
  L230  main                   g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE)
  L233  main                   m = load(ALL[nm], fn)
  L242  main                   P_pts = pts.get(name, [])
  L245  main                   m4 = P.read_archive_mask(os.path.join(grp, base + ".jpg"))
  L247  main                   for q, (x, y) in enumerate(P_pts):
  L254  main                   own = [q for q, (x, y) in enumerate(P_pts)
  L261  main                   x, y = P_pts[q]
  L299  main                   for (px, py) in P_pts:
  L307  main                   with open(os.path.join(UN, "reports/wire_reasons.jsonl"), "a", encoding="utf-8") as _fh:
  L310  main                   _own = [q for q, (x, y) in enumerate(P_pts) if w["mask"][int(round(y)), int(round(x))]]
  L311  main                   _dir = os.path.join(UN, "pred_masks_all"); os.makedirs(_dir, exist_ok=True)
  L313  main                   cv2.imwrite(os.path.join(_dir, _f), (w["mask"] * 255).astype("uint8"))
  L320  main                   own = [q for q, (x, y) in enumerate(P_pts)
  L371  main                   _d = float(np.hypot(P_pts[w["pts2"][0]][0] - P_pts[w["pts2"][1]][0],
  L372  main                   P_pts[w["pts2"][0]][1] - P_pts[w["pts2"][1]][1]))
  L377  main                   x_p, y_p = P_pts[q]
  L386  main                   with open(os.path.join(UN, "reports/wire_dropped.jsonl"), "a", encoding="utf-8") as _fd:
  L394  main                   od_w = os.path.join(UN, "pred_masks_wires"); os.makedirs(od_w, exist_ok=True)
  L396  main                   cv2.imwrite(os.path.join(od_w, wf), (m * 255).astype(np.uint8))
  L403  main                   out.sort(key=lambda r: -r["jit_max"])
  L409  main                   json.dump(out, open(os.path.join(UN, "reports/straightness_skeleton.json"), "w", encoding="utf-8"),
  L411  main                   print("共 {} 条合格飞线的线段读数 ✓；线段掩膜 -> {}/pred_masks_wires/".format(len(out), UN))

## core/vendor/s5_steps/s253_gap_fill.py （266 行，命中 32 行）
  L3    <模块级>                  import glob
  L35   <模块级>                  SRC = os.path.join(HERE, "chip_samples__20260128")
  L36   <模块级>                  UN = os.path.join(HERE, "outputs", "_masks")
  L37   <模块级>                  PAN = os.path.join(HERE, "outputs")
  L38   <模块级>                  SOLD = os.path.join(HERE, "refs", "solder_table.json")
  L39   <模块级>                  BOARD_FILE = os.path.join(HERE, "refs", "boards.json")
  L40   <模块级>                  BOARDS = []
  L45   <模块级>                  ROOT = os.path.join(HERE, "refs", "annot")
  L46   <模块级>                  ANN = ROOT
  L47   <模块级>                  MASKDIR = UN
  L67   <模块级>                  sys.path.insert(0, os.path.join(HERE, "lib"))
  L88   load                   p = os.path.join(UN, "pred_masks_chan9", k[5:] + "_" + fn) if k.startswith("chan:") else os.path.join(UN, k, fn)
  L89   load                   v = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
  L105  build_W                def build_W(keep, lab2, n2, P_pts):
  L110  build_W                own = [q for q, (x, y) in enumerate(P_pts)
  L121  build_W                x, y = P_pts[q]
  L128  build_W                o2 = [q for q, (x, y) in enumerate(P_pts)
  L135  main                   rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"]
  L143  main                   for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
  L144  main                   grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
  L145  main                   key = grp + "/" + base[-4:]; name = grp + "-" + base[-4:]
  L146  main                   fn = grp + "_" + base[-4:] + ".png"
  L147  main                   g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE)
  L150  main                   m = load(ALL[nm], fn)
  L159  main                   P_pts = pts.get(name, [])
  L160  main                   W = build_W(keep, lab2, n2, P_pts)
  L179  main                   for (px_, py_) in P_pts:                       # ← 本飞线骨架端点附近（≤删盘+8px）的焊点圆盘也纳入区域 ✓
  L184  main                   x, y = P_pts[q]
  L228  main                   for (x, y) in P_pts:
  L256  main                   od = os.path.join(PAN, "state", "gapfill"); os.makedirs(od, exist_ok=True)
  L257  main                   fn2 = os.path.join(od, "补缺_{}.png".format(tag))
  L261  main                   "boards": rep}, open(os.path.join(UN, "reports/gapfill.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

## core/s_report.py （185 行，命中 15 行）
  L32   _blob_of               grp, tail = (board.split("/") if "/" in board else board.rsplit("-", 1))
  L35   _blob_of               m = cv2.imread(os.path.join(run, "masks", k, grp + "_" + tail + ".png"), cv2.IMREAD_GRAYSCALE)
  L53   build                  rep = os.path.join(run, "masks", "reports")
  L54   build                  rows = json.load(open(os.path.join(rep, "straightness_skeleton.json"), encoding="utf-8"))
  L55   build                  reasons = [json.loads(l) for l in open(os.path.join(rep, "wire_reasons.jsonl"), encoding="utf-8")]
  L56   build                  wdir = os.path.join(run, "masks", "pred_masks_wires")
  L72   build                  if os.path.exists(os.path.join(wdir, x["file"])):
  L86   build                  c0 = max(cnt, key=cv2.contourArea)
  L95   build                  grp, tail = (b.split("/") if "/" in b else b.rsplit("-", 1))
  L96   build                  cands = [f for f in sorted(os.listdir(os.path.join(src, grp))) if tail in f]
  L98   build                  g = cv2.cvtColor(cv2.imread(os.path.join(src, grp, cands[0]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
  L99   build                  for f in sorted(os.listdir(wdir)):
  L100  build                  key = by_file.get(f)
  L103  build                  m = cv2.imread(os.path.join(wdir, f), cv2.IMREAD_GRAYSCALE)
  L140  build                  _bpj = json.load(open(os.path.join(run, "masks", "reports", "bare_pad.json"), encoding="utf-8"))

## core/s_stagefig.py （80 行，命中 16 行）
  L4    <模块级>                  import glob
  L18   _stack                 _p = os.path.join(d, f)
  L21   _stack                 m = cv2.imread(_p, 0)
  L29   build                  reas = os.path.join(run, "masks", "reports", "wire_reasons.jsonl")
  L30   build                  rows = [json.loads(l) for l in open(reas, encoding="utf-8")]
  L31   build                  sold = json.load(open(os.path.join(run, "masks", "solder_table.json"), encoding="utf-8"))["rows"]
  L32   build                  out_dir = out_dir or os.path.join(run, "state", "stagefig")
  L36   build                  grp, tail = b.split("/")
  L37   build                  key = grp + "_" + tail
  L38   build                  imgs = [f for f in sorted(os.listdir(os.path.join(src, grp))) if tail in f]
  L41   build                  g = cv2.imread(os.path.join(src, grp, imgs[0]), 0)
  L45   build                  m = cv2.imread(os.path.join(run, "masks", k, key + ".png"), 0)
  L49   build                  ("2_除杂后_判定保留块", _stack(files, os.path.join(run, "masks", "pred_masks_all"))),
  L50   build                  ("3_最终飞线掩膜", _stack(files, os.path.join(run, "masks", "pred_masks_wires")))]
  L78   build                  Image.fromarray(canv).save(os.path.join(out_dir, "三段对照_{}.png".format(b.replace("/", "_"))))
  L79   build                  json.dump(res, open(os.path.join(out_dir, "stagefig.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

## core/s_barepad.py （132 行，命中 19 行）
  L18   _union                 _p = os.path.join(d, f)
  L21   _union                 m = cv2.imread(_p, 0)
  L29   build                  rows = json.load(open(solder_json, encoding="utf-8"))["rows"]
  L30   build                  reas = os.path.join(run, "masks", "reports", "wire_reasons.jsonl")
  L31   build                  rws = [json.loads(l) for l in open(reas, encoding="utf-8")]
  L32   build                  out_dir = out_dir or os.path.join(run, "masks", "pred_masks_barepad")
  L39   build                  grp, tail = b.split("/")
  L40   build                  key = grp + "_" + tail
  L43   build                  m = cv2.imread(os.path.join(run, "masks", k, key + ".png"), 0)
  L51   build                  kept = _union(files, os.path.join(run, "masks", "pred_masks_all"))
  L57   build                  wire_u = _union(files, os.path.join(run, "masks", "pred_masks_wires"))
  L78   build                  cv2.imwrite(os.path.join(out_dir, "{}_bp{:03d}.png".format(b.replace("/", "_"), n_bp)), mm)
  L86   build                  json.dump(res, open(os.path.join(run, "masks", "reports", "bare_pad.json"), "w", encoding="utf-8"),
  L99   crop_sheet             out_png = out_png or os.path.join(run, "state", "report", "裸焊盘_{}个_逐块.png".format(len(res)))
  L103  crop_sheet             b = r["board"]; grp, tail = b.split("/")
  L104  crop_sheet             fs = [x for x in sorted(os.listdir(os.path.join(src, grp))) if tail in x]
  L107  crop_sheet             gm = cv2.imread(os.path.join(src, grp, fs[0]), 0)
  L108  crop_sheet             mp = os.path.join(run, "masks", "pred_masks_barepad", "{}_bp{:03d}.png".format(b.replace("/", "_"), r["idx"]))
  L109  crop_sheet             mm = cv2.imread(mp, 0) if os.path.exists(mp) else None
