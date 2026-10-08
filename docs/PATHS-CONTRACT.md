# 路径契约表（机械抽取，2026-10-07）

规则：只有列在【写】里的路径才允许被别的模块【读】；命名字段必须两边一致。

## run_pipeline.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L44 | run_s1_s3 | 读 | g0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) |
| L57 | run_s1_s3 | 写 | cv2.imwrite(os.path.join(md, k, base), (np.asarray(m) > 0).astype('uint8') * 255) |
| L59 | run_s1_s3 | 写 | json.dump({'rows': rows_all}, open(os.path.join(md, 'solder_table.json'), 'w', encoding='utf-8'), |
| L77 | main | 读 | log = open(os.path.join(run, 'run.log'), 'w', encoding='utf-8') |
| L97 | main | 写 | say('[S4 掩膜分类] ok %.1fs %s' % (time.time() - t0, json.dumps(r4, ensure_ascii=False)[:600])) |
| L131 | _stagefig | 读 | for grp in sorted(os.listdir(src)): |
| L135 | _stagefig | 读 | for f in sorted(os.listdir(d)): |

## core/s4_classify.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L33 | _install_on_the_fly_mask | 读 | cand = [f for f in os.listdir(d) if os.path.splitext(f)[0][-4:] == key] |

## core/s5_straightness.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L63 | run | 读 | cand = [f for f in os.listdir(d) if os.path.splitext(f)[0][-4:] == k] |

## core/s_report2.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L53 | _overlay | 读 | m = cv2.imread(path, 0) |
| L88 | build | 读 | rows = json.load(open(os.path.join(rep, "straightness_skeleton.json"), encoding="utf-8")) |
| L92 | build | 读 | reasons = [json.loads(l) for l in open(os.path.join(rep, "wire_reasons.jsonl"), encoding="utf-8")] |
| L96 | build | 读 | bare = json.load(open(os.path.join(rep, "bare_pad.json"), encoding="utf-8")) |
| L100 | build | 读 | name_map = json.load(open(os.path.join(run, "inputs", "产品名对照.json"), encoding="utf-8")) |
| L104 | build | 读 | have_wire = set(os.listdir(wires_dir)) if os.path.isdir(wires_dir) else set() |
| L151 | build | 读 | gray = cv2.imread(t, 0) |
| L157 | build | 读 | for f in sorted(os.listdir(d)): |
| L159 | build | 读 | gray = cv2.imread(os.path.join(d, f), 0) |
| L210 | build | 写 | im.save(out_png, "PNG") |
| L216 | build | 写 | with open(csv_p, "w", newline="", encoding="utf-8-sig") as fh: |
| L233 | build | 写 | Image.fromarray(canv).save(ov, "PNG") |

## core/s_stagefig.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L21 | _stack | 读 | m = cv2.imread(_p, 0) |
| L30 | build | 读 | rows = [json.loads(l) for l in open(reas, encoding="utf-8")] |
| L31 | build | 读 | sold = json.load(open(os.path.join(run, "masks", "solder_table.json"), encoding="utf-8"))["rows"] |
| L38 | build | 读 | imgs = [f for f in sorted(os.listdir(os.path.join(src, grp))) if tail in f] |
| L41 | build | 读 | g = cv2.imread(os.path.join(src, grp, imgs[0]), 0) |
| L45 | build | 读 | m = cv2.imread(os.path.join(run, "masks", k, key + ".png"), 0) |
| L78 | build | 写 | Image.fromarray(canv).save(os.path.join(out_dir, "三段对照_{}.png".format(b.replace("/", "_")))) |
| L79 | build | 写 | json.dump(res, open(os.path.join(out_dir, "stagefig.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1) |

## core/s_barepad.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L21 | _union | 读 | m = cv2.imread(_p, 0) |
| L29 | build | 读 | rows = json.load(open(solder_json, encoding="utf-8"))["rows"] |
| L31 | build | 读 | rws = [json.loads(l) for l in open(reas, encoding="utf-8")] |
| L43 | build | 读 | m = cv2.imread(os.path.join(run, "masks", k, key + ".png"), 0) |
| L78 | build | 写 | cv2.imwrite(os.path.join(out_dir, "{}_bp{:03d}.png".format(b.replace("/", "_"), n_bp)), mm) |
| L86 | build | 写 | json.dump(res, open(os.path.join(run, "masks", "reports", "bare_pad.json"), "w", encoding="utf-8"), |
| L104 | crop_sheet | 读 | fs = [x for x in sorted(os.listdir(os.path.join(src, grp))) if tail in x] |
| L107 | crop_sheet | 读 | gm = cv2.imread(os.path.join(src, grp, fs[0]), 0) |
| L109 | crop_sheet | 读 | mm = cv2.imread(mp, 0) if os.path.exists(mp) else None |
| L131 | crop_sheet | 写 | Image.fromarray(canv).save(out_png) |

## core/vendor/s4_core.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L22 | load | 读 | v = cv2.imread(p, cv2.IMREAD_GRAYSCALE) |
| L51 | run_reference | 读 | rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"] |
| L58 | run_reference | 读 | for f in sorted(glob.glob(ROOT + "/*/*_m.png")): |
| L62 | run_reference | 读 | g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE) |
| L196 | run_reference | 写 | json.dump(summary, open(os.path.join(UN, "reports/registry.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1) |

## core/vendor/s5_steps/s253_gap_fill.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L89 | load | 读 | v = cv2.imread(p, cv2.IMREAD_GRAYSCALE) |
| L135 | main | 读 | rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"] |
| L143 | main | 读 | for f in sorted(glob.glob(ROOT + "/*/*_m.png")): |
| L147 | main | 读 | g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE) |
| L258 | main | 写 | pg.save(fn2, "PNG", optimize=True) |
| L260 | main | 写 | json.dump({"close_r": CLOSE_R, "bridge": {"maxd": BRIDGE_MAXD, "maxa": BRIDGE_MAXA, "thick": BRIDGE_T}, |
| L261 | main | 读 | "boards": rep}, open(os.path.join(UN, "reports/gapfill.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1) |

## core/vendor/s5_steps/s255_straightness.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L98 | load | 读 | v = cv2.imread(p, cv2.IMREAD_GRAYSCALE) |
| L221 | main | 读 | rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"] |
| L226 | main | 读 | for f in sorted(glob.glob(ROOT + "/*/*_m.png")): |
| L230 | main | 读 | g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE) |
| L307 | main | 读 | with open(os.path.join(UN, "reports/wire_reasons.jsonl"), "a", encoding="utf-8") as _fh: |
| L313 | main | 写 | cv2.imwrite(os.path.join(_dir, _f), (w["mask"] * 255).astype("uint8")) |
| L314 | main | 写 | _fh.write(_json.dumps({"board": key, "idx": _i, "file": _f, "area": int(w["mask"].sum()), |
| L386 | main | 读 | with open(os.path.join(UN, "reports/wire_dropped.jsonl"), "a", encoding="utf-8") as _fd: |
| L387 | main | 写 | _fd.write(_json.dumps({"board": key, "cid": w.get("cid"), "area": int(w["mask"].sum()), |
| L396 | main | 写 | cv2.imwrite(os.path.join(od_w, wf), (m * 255).astype(np.uint8)) |
| L409 | main | 写 | json.dump(out, open(os.path.join(UN, "reports/straightness_skeleton.json"), "w", encoding="utf-8"), |

## core/vendor/s5_steps/s256_panels.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L72 | main | 读 | rows = json.load(open(os.path.join(UN, "reports/straightness_skeleton.json"), encoding="utf-8")) |
| L75 | main | 读 | for e in json.load(open(SOLD, encoding="utf-8"))["rows"]: |
| L87 | main | 读 | g = cv2.imread(os.path.join(SRC, grp, stem + ".jpg"), cv2.IMREAD_GRAYSCALE) |
| L88 | main | 读 | m = cv2.imread(os.path.join(UN, "pred_masks_wires", r["file"]), cv2.IMREAD_GRAYSCALE) |
| L159 | main | 写 | pg.save(fn, "PNG", optimize=True) |

## core/vendor/s5_steps/s257_numbered.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L100 | main | 读 | rows = json.load(open(os.path.join(UN, "reports/straightness_skeleton.json"), encoding="utf-8")) |
| L102 | main | 读 | for e in json.load(open(SOLD, encoding="utf-8"))["rows"]: |
| L109 | main | 读 | rsn = [json.loads(l) for l in open(os.path.join(UN, "reports/wire_reasons.jsonl"), encoding="utf-8")] |
| L126 | main | 读 | for _f in sorted(os.listdir(os.path.join(SRC, grp))):      # 按板号自动匹配原图 ✓（避免硬编码文件名出错 ✗） |
| L128 | main | 读 | g = cv2.imread(os.path.join(SRC, grp, _f), cv2.IMREAD_GRAYSCALE) |
| L138 | main | 读 | m = cv2.imread(os.path.join(UN, "pred_masks_wires", r["file"]), cv2.IMREAD_GRAYSCALE) |
| L174 | main | 读 | _m = cv2.imread(os.path.join(UN, "pred_masks_all", _f), cv2.IMREAD_GRAYSCALE) |
| L222 | main | 写 | cv2.imwrite(fn, pg) |
| L223 | main | 写 | json.dump(ids, open(os.path.join(UN, "reports/wire_ids.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1) |

## core/vendor/s5_steps/s259_stage.py
| 行 | 函数 | 动作 | 语句 |
|---|---|---|---|
| L69 | find_src | 读 | for f in sorted(os.listdir(os.path.join(SRC, grp))): |
| L71 | find_src | 读 | return cv2.imread(os.path.join(SRC, grp, f), cv2.IMREAD_GRAYSCALE) |
| L81 | main | 读 | rows = [json.loads(l) for l in open(os.path.join(UN, "reports/wire_reasons.jsonl"), encoding="utf-8")] |
| L82 | main | 读 | rd = json.load(open(os.path.join(UN, "reports/straightness_skeleton.json"), encoding="utf-8")) |
| L84 | main | 读 | for e in json.load(open(SOLD, encoding="utf-8"))["rows"]: |
| L96 | main | 读 | m = cv2.imread(os.path.join(UN, "pred_masks_all", r["file"]), cv2.IMREAD_GRAYSCALE) |
| L115 | main | 读 | m = cv2.imread(os.path.join(UN, "pred_masks_wires", r["file"]), cv2.IMREAD_GRAYSCALE) |
| L146 | main | 读 | m = cv2.imread(os.path.join(UN, "pred_masks_all", r["file"]), cv2.IMREAD_GRAYSCALE) |
| L187 | main | 写 | cv2.imwrite(fn, pg) |

## 关键契约（由代码推出）
1. 板枚举：S4/S255 都按 `ROOT/*/*_m.png` 枚举，取 `base = 文件名去掉 _m.png`；`key=组名/base[-4:]`、`name=组名-base[-4:]`、`fn=组名_base[-4:].png`。
2. 原图：`SRC/组名/base.jpg`（用完整 base，不是 base[-4:]）。
3. 焊点表：写入时 `sample=组名/尾号` 与 `sample_alt=组名-尾号` 并存；S4/S255 查表用 `name`（连字符形式）⇒ 两边登记必须一致。
4. 掩膜：三成员写在 `UN/<成员>/<fn>`；飞线掩膜写在 `UN/pred_masks_wires/<file>`；判定阶段块在 `UN/pred_masks_all/<file>`。
5. 报告：读 `UN/reports/*.json(l)` 与 `UN/pred_masks_*`，写 `state/report/**`。