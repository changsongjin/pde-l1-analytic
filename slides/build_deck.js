// 进展汇报 PPT。数字全部取自论文和结果 JSON。运行：cd slides && node build_deck.js
// 需要环境变量 PPTX_SKILL（pptx skill 目录）和 NODE_PATH（本目录 node_modules）。中文配图由 make_figures_zh.py 生成。
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const THEME = {
  name: "PDE L1",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: "141B2D", lt1: "FFFFFF", dk2: "1E2A44", lt2: "EEF3FA",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "6250D6", accent6: "52514E", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = { blue: "2A78D6", paleBlue: "DCEAFB", soft: "F3F6FA", grid: "D9DDE3", muted: "6B6A66", okText: "13865E", badText: "C2410C" };
const MATH = "Cambria Math";
const IMG = (f) => path.join(__dirname, "img", f);
const SIZE = { overview: [1828, 908], flowchart: [1908, 795], gate: [2024, 807], inverse: [1551, 801],
  nInv: [3427, 923], nFwd: [1517, 1286], nSga: [3434, 880] }; // n* 由 make_noise_figures_zh.py 生成

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "用 L1 稀疏库求 PDE 解析解";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- 版式 ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 7.1, h: 2.0, fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.1, w: 6.6, h: 1.6, fontSize: 18, color: "C9D6EA", valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8A8984", align: "right" },
});

let sectionName = "";
function section(title) { pres.addSection({ title }); sectionName = title; }
function content(title, sec) {
  if (sec && sec !== sectionName) section(sec);
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sectionName });
  s.addText(title, { placeholder: "title" });
  return s;
}
const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top", ...o });
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
}
function badge(s, x, y, n, color, d = 0.5) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color } });
  s.addText(String(n), { isTextBox: true, x, y, w: d, h: d, margin: 0, align: "center", valign: "middle", fontSize: d > 0.6 ? 22 : 16, bold: true, color: C.background1 });
}
// 贯穿全片的视觉元素：候选项“芯片”，保留的为蓝色、压成 0 的为灰色
function chip(s, x, y, w, h, text, on, fs = 14, dark = false) {
  const fill = on ? (dark ? HEX.blue : "FFFFFF") : (dark ? "2C3A57" : "F0EFEC");
  const line = on ? HEX.blue : (dark ? "4A5873" : "B8B7B2");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: line, width: 1 } });
  s.addText(text, { isTextBox: true, x, y, w, h, margin: 0, align: "center", valign: "middle", fontSize: fs, fontFace: MATH,
    color: on ? (dark ? "FFFFFF" : "141B2D") : (dark ? "8D9AB3" : "8A8984") });
}
const img = (s, key, file, x, y, w, alt) => s.addImage({ path: IMG(file), x, y, w, h: w * SIZE[key][1] / SIZE[key][0], altText: alt });
const tableOpts = (x, y, w, colW) => ({ x, y, w, colW, fontSize: 14, color: C.text1, border: { type: "solid", pt: 0.75, color: HEX.grid }, valign: "middle", margin: [0.04, 0.08, 0.04, 0.08] });
const head = (cells) => cells.map((t) => ({ text: t, options: { bold: true, fill: { color: HEX.paleBlue }, color: C.text2 } }));
const caption = (s, text, x, y, w) => T(s, text, { x, y, w, h: 0.4, fontSize: 13, color: HEX.muted });

// ---------- 1 封面 ----------
section("开场");
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sectionName });
  s.addText("用 L1 稀疏库求 PDE 的解析解", { placeholder: "title" });
  s.addText("先设出很多候选项和系数，让 L1 把大部分系数压成 0\n正问题与反问题 · 进展汇报 · 2026-10-30", { placeholder: "body" });
  const items = ["1", "tanh θ", "tanh² θ", "tanh³ θ", "tanh⁴ θ", "sech θ", "e^(−θ²)", "θ"];
  const coef = ["2", "0", "−2", "0", "0", "0", "0", "0"];
  items.forEach((t, i) => {
    const x = 8.2 + (i % 2) * 2.15, y = 1.45 + Math.floor(i / 2) * 1.05;
    chip(s, x, y, 1.9, 0.62, t, coef[i] !== "0", 16, true);
    s.addText(coef[i], { isTextBox: true, x: x + 1.9 - 0.45, y: y - 0.32, w: 0.45, h: 0.3, margin: 0, align: "right", fontSize: 12, bold: coef[i] !== "0", color: coef[i] !== "0" ? "9CC3F2" : "6F7D97" });
  });
  s.addText("u = 2 − 2tanh²θ = 2sech²(x − 4t)", { isTextBox: true, x: 8.2, y: 5.85, w: 4.05, h: 0.5, margin: 0, align: "center", fontSize: 16, fontFace: MATH, color: "C9D6EA" });
  s.addNotes("这次汇报老师提出的思路：先把候选项和系数都设出来，用 L1 让大部分系数为 0，剩下的就是解析解。右边是 KdV 的例子：8 个候选项里只留下两项，拼起来就是孤子解。");
}

// ---------- 2 目录 ----------
{
  const s = content("汇报内容");
  const parts = [
    ["背景", "老师的想法、相关工作、两个方向"],
    ["方法", "L1 选项规则、两个难点和修法、算法流程"],
    ["结果", "实验设置、通用库结果、门槛与反例、汇总"],
    ["补充实验", "加高斯噪声的误差分析、与遗传算法比较"],
    ["总结", "局限与下一步"],
  ];
  parts.forEach(([h, d], i) => {
    const x = 0.6 + i * 2.45;
    card(s, x, 1.7, 2.25, 3.7, i === 3 ? HEX.paleBlue : HEX.soft, "agenda-" + i);
    T(s, String(i + 1).padStart(2, "0"), { x: x + 0.25, y: 2.0, w: 1.8, h: 1.0, fontSize: 44, bold: true, color: C.accent1 });
    T(s, h, { x: x + 0.25, y: 3.2, w: 1.85, h: 0.6, fontSize: 22, bold: true, color: C.text2 });
    T(s, d, { x: x + 0.25, y: 4.0, w: 1.8, h: 1.8, fontSize: 14 });
  });
  s.addNotes("先讲老师的想法和已有工作，再讲方法里遇到的两个难点和结果；第四部分是按老师意见补的噪声实验和遗传算法比较；最后是局限和下一步。");
}

// ---------- 3 出发点 ----------
{
  const s = content("出发点：把“猜解”换成“选项”", "背景");
  card(s, 0.6, 1.5, 5.6, 4.3, HEX.paleBlue, "advisor-idea");
  T(s, "导师的想法", { x: 0.95, y: 1.8, w: 5.0, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  T(s, "面对一个新问题，只知道 PDE，想知道它的解析解。解由很多项和系数构成，但不知道是哪些项、系数是几。", { x: 0.95, y: 2.5, w: 4.9, h: 1.7, fontSize: 18 });
  T(s, "能不能先把这些项和系数全部设出来，用 L1 norm 让大部分系数为 0？", { x: 0.95, y: 4.4, w: 4.9, h: 1.4, fontSize: 18, italic: true, color: C.accent1 });
  const steps = [
    ["设候选项", "写出很多候选函数，系数全部未知"],
    ["L1 压零", "拟合 PDE 和初边值，L1 让多数系数精确为 0"],
    ["读出解析解", "剩下的几项和系数就是闭式解；代回残差到机器精度才算数"],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.6 + i * 1.6;
    badge(s, 6.9, y, i + 1, C.accent1);
    T(s, h, { x: 7.65, y: y - 0.02, w: 5.0, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
    T(s, d, { x: 7.65, y: y + 0.5, w: 5.0, h: 0.8, fontSize: 16 });
  });
  s.addNotes("三步：设候选项、L1 压零、读出解析解。判断“真的得到解析解”的标准是代回 PDE 后残差降到机器精度，只是残差小不够。");
}

// ---------- 4 相关工作 ----------
{
  const s = content("相关工作：多数做的是“从数据反推方程”", "背景");
  const cols = [
    ["稀疏回归识别方程", "已知数据，求方程", ["SINDy（Brunton 等，PNAS 2016）", "PDE-FIND（Rudy 等，Sci. Adv. 2017）", "Schaeffer 2017：L1 稀疏回归", "PDE-Net（ICML 2018）、DeepMoD"]],
    ["遗传算法识别方程", "已知数据，求方程", ["DLGA-PDE（Xu 等，2020）", "SGA-PDE（Chen 等，PRR 2022）", "均出自张东晓、陈云天组"]],
    ["神经网络求解", "已知方程，求数值解", ["PINN（Raissi 等，JCP 2019）", "Deep Ritz（E、Yu）", "给出数值近似，不给闭式解"]],
  ];
  cols.forEach(([h, sub, items], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.5, 3.8, 3.75, HEX.soft, "related-" + i);
    T(s, h, { x: x + 0.3, y: 1.75, w: 3.3, h: 0.45, fontSize: 19, bold: true, color: C.text2 });
    T(s, sub, { x: x + 0.3, y: 2.25, w: 3.3, h: 0.35, fontSize: 14, color: C.accent1 });
    T(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })), { x: x + 0.3, y: 2.8, w: 3.35, h: 2.3, fontSize: 14, paraSpaceAfter: 6 });
  });
  card(s, 0.6, 5.5, 12.1, 1.2, HEX.paleBlue, "our-position");
  T(s, [
    { text: "我们：", options: { bold: true, color: C.accent1 } },
    { text: "已知方程，用稀疏展开直接找闭式解（正问题）；同一套工具也做反问题，并解释了 PySINDy 公开 KdV 例子里同样出现的“多留 uₓ”现象。" },
  ], { x: 0.9, y: 5.68, w: 11.5, h: 0.9, fontSize: 16, color: C.text2, valign: "middle" });
  s.addNotes("文献主流是从数据反推方程。已知方程、用稀疏展开找闭式解这个方向做得少；神经网络方法只给数值解。");
}

// ---------- 5 两个方向 ----------
{
  const s = content("同一套工具，两个方向", "背景");
  img(s, "overview", "overview_zh.png", 1.3, 1.3, 10.7, "正问题与反问题示意图");
  caption(s, "蓝色：L1 保留的项（系数为重拟合值）；灰色：L1 压成 0 的项。示例都是 KdV。", 1.3, 6.75, 10.7);
  s.addNotes("上面是正问题：已知 KdV 方程和初边值，在 8 个通用候选项里 L1 只留下 1 和 tanh²，系数 2 和 -2，拼起来就是孤子 2sech²(x-4t)。下面是反问题：已知双孤子的采样，L1 只留下 uu_x 和 u_xxx。");
}

// ---------- 6 方法核心 ----------
{
  const s = content("方法核心：只让 L1 决定留哪几项", "方法");
  const cards = [
    ["目标函数", "½‖r(ξ)‖² + λ Σ wⱼ |ξⱼ|", "r = PDE 配点残差 + 边界残差 + 初值残差，三块等权；不给解的内部采样"],
    ["定支撑", "40 个 λ，|ξⱼ| > 10⁻⁵ 过半", "λ 从 λ_max 几何降到 10⁻⁶ λ_max\n阈值不放宽"],
    ["重拟合", "只在支撑上最小二乘", "只读系数，不增删任何一项"],
  ];
  cards.forEach(([h, f, d], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.5, 3.8, 3.4, HEX.soft, "method-card-" + i);
    badge(s, x + 0.3, 1.8, i + 1, C.accent1);
    T(s, h, { x: x + 0.95, y: 1.83, w: 2.7, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
    T(s, f, { x: x + 0.3, y: 2.6, w: 3.3, h: 0.8, fontSize: 20, fontFace: MATH, color: C.accent1 });
    T(s, d, { x: x + 0.3, y: 3.5, w: 3.3, h: 1.3, fontSize: 15 });
  });
  card(s, 0.6, 5.2, 12.1, 1.4, HEX.paleBlue, "solver-note");
  T(s, [
    { text: "精确求解：", options: { bold: true, color: C.accent1 } },
    { text: "每个 L1 子问题都用 LARS 同伦路径求解（KKT 误差 < 5×10⁻⁹），排除“求解器没收敛”这种解释。非线性方程用 Gauss–Newton 线性化，每步解一个精确 L1。" },
  ], { x: 0.9, y: 5.4, w: 11.5, h: 1.0, fontSize: 16, color: C.text2, valign: "middle" });
  s.addNotes("关键约束：哪几项留下只由 L1 决定；重拟合只读系数。");
}

// ---------- 7 难点一 ----------
{
  const s = content("难点一：反问题里普通 L1 总多留一个 uₓ", "方法");
  img(s, "inverse", "inverse_path_zh.png", 0.6, 1.45, 7.2, "普通 L1 沿 λ 路径的 u_x 系数");
  caption(s, "真项进来后，uₓ 系数随 λ 成比例下降，但始终不为 0", 0.6, 5.3, 7.2);
  T(s, "不可表示条件：uₓ 列的值", { x: 8.4, y: 1.5, w: 4.3, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  [["2.77", "KdV 双孤子"], ["1.39", "Burgers，Cole–Hopf"], ["1.52", "Burgers，两行波"]].forEach(([v, l], i) => {
    const y = 2.1 + i * 1.15;
    T(s, v, { x: 8.4, y, w: 1.9, h: 0.9, fontSize: 44, bold: true, color: C.accent2, valign: "middle" });
    T(s, l, { x: 10.35, y: y + 0.2, w: 2.4, h: 0.5, fontSize: 16, valign: "middle" });
  });
  card(s, 8.4, 5.6, 4.3, 1.1, "FDEEE7", "irrep-rule");
  T(s, "需要 < 1 才不会被选；> 1 时任何足够小的 λ 都会选上，与求解器无关", { x: 8.6, y: 5.72, w: 3.95, h: 0.9, fontSize: 14, color: C.text2 });
  s.addNotes("精确求解器复现了旧结果，KKT 误差 1e-10 量级，排除了没收敛。其他错列的不可表示值都小于 1，只有 u_x 大于 1，所以普通 L1 只会多留 u_x。这也解释了为什么旧的 1e-8 残差门槛整条路径都到不了：残差和 λ 成正比。");
}

// ---------- 8 修法 ----------
{
  const s = content("修法：迭代重加权 L1，两轮去掉 uₓ", "方法");
  const flow = [
    ["第 0 轮：普通 L1", "KdV 选 {uuₓ, uₓₓₓ, uₓ, uₓₓ}\nBurgers 选 {uuₓ, uₓₓ, uₓ}"],
    ["改权重", "wⱼ = 1/(|βⱼ| + ε)，β 取上一轮 L1 解\nuₓ 系数小 → 权重大"],
    ["第 1 轮起只剩真项", "KdV：uuₓ (−6)，uₓₓₓ (−1)\nBurgers：uuₓ (−1)，uₓₓ (0.1)"],
  ];
  flow.forEach(([h, d], i) => {
    const y = 1.5 + i * 1.75;
    card(s, 0.6, y, 6.2, 1.45, i === 2 ? HEX.paleBlue : HEX.soft, "irl1-step-" + i);
    badge(s, 0.85, y + 0.25, i + 1, i === 2 ? C.accent1 : C.accent6);
    T(s, h, { x: 1.55, y: y + 0.2, w: 5.1, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
    T(s, d, { x: 1.55, y: y + 0.68, w: 5.1, h: 0.7, fontSize: 15 });
  });
  T(s, "第 0 轮 λ 比例的敏感性", { x: 7.3, y: 1.5, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  caption(s, "每格汇总 4 个 ε，结果与 ε 无关", 7.3, 1.95, 5.4);
  const P = (t) => ({ text: t, options: { color: t === "通过" ? HEX.okText : HEX.badText, bold: true, align: "center" } });
  const rows = [
    head(["λ/λ_max", "KdV", "Burgers CH", "Burgers 两行波"]),
    ["10⁻¹", "失败", "失败", "失败"], ["10⁻²", "失败", "通过", "通过"], ["10⁻³", "通过", "通过", "通过"],
    ["10⁻⁴", "通过", "通过", "通过"], ["10⁻⁵", "通过", "通过", "通过"],
  ].map((r, i) => (i === 0 ? r : [r[0], P(r[1]), P(r[2]), P(r[3])]));
  rows.push(["合计", "12/20", "16/20", "16/20"]);
  s.addTable(rows, { ...tableOpts(7.3, 2.45, 5.4, [1.2, 1.2, 1.4, 1.6]), rowH: 0.42, objectName: "sensitivity" });
  caption(s, "论文固定用 10⁻³ 并附这张表；比例太大时第 0 轮漏掉真项，重加权会锁死错的支撑", 7.3, 5.9, 5.4);
  s.addNotes("权重只来自上一轮 L1 解，不用岭回归：岭回归会把能量分给相关的 u_x，反而让 u_x 权重变小。阈值 1e-5 没有放宽。");
}

// ---------- 9 难点二 ----------
{
  const s = content("难点二：旧的正问题库里本来就放着答案", "方法");
  card(s, 0.6, 1.5, 5.85, 4.45, HEX.soft, "old-library");
  T(s, "之前：针对真解设计的库", { x: 0.9, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  ["2sech²(x−4t)", "c = 1 孤子", "c = 2 孤子", "c = 9 孤子"].forEach((t, i) => chip(s, 0.9 + (i % 2) * 2.7, 2.45 + Math.floor(i / 2) * 0.7, 2.5, 0.52, t, i === 0, 14));
  T(s, [
    { text: "KdV 库里直接有 2sech²(x − 4t) 这一列", options: { bullet: true, breakLine: true } },
    { text: "波速预先离散成几列：c = 1, 2, 4, 9", options: { bullet: true, breakLine: true } },
    { text: "L1 只是在准备好的几列里挑一列", options: { bullet: true } },
  ], { x: 0.9, y: 4.05, w: 5.3, h: 2.2, fontSize: 16, paraSpaceAfter: 10 });
  card(s, 6.85, 1.5, 5.85, 4.45, HEX.paleBlue, "generic-library");
  T(s, "现在：三个方程共用的通用库", { x: 7.15, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  ["1", "tanh θ", "tanh² θ", "tanh³ θ", "tanh⁴ θ", "sech θ", "e^(−θ²)", "θ"].forEach((c, i) =>
    chip(s, 7.15 + (i % 4) * 1.32, 2.45 + Math.floor(i / 4) * 0.7, 1.2, 0.52, c, i < 5, 14));
  T(s, "θ = k(x − x₀ − ct)，(k, c, x₀) 连续搜索", { x: 7.15, y: 3.95, w: 5.3, h: 0.5, fontSize: 16, fontFace: MATH, color: C.accent1 });
  T(s, [
    { text: "前五项：求行波解常用的 tanh 展开法", options: { bullet: true, breakLine: true } },
    { text: "后三项：干扰项", options: { bullet: true, breakLine: true } },
    { text: "没有任何一列等于真解，真解要由几列拼出来", options: { bullet: true } },
  ], { x: 7.15, y: 4.6, w: 5.3, h: 1.7, fontSize: 15, paraSpaceAfter: 8 });
  s.addNotes("这是为了回答“库里本来就放了答案”的质疑。搜索范围是通用量级：k 在 0.1 到 10，c 在 -8 到 8，x0 在整个计算域。");
}

// ---------- 10 流程图 ----------
{
  const s = content("算法流程：三步，L1 只管选项", "方法");
  img(s, "flowchart", "flowchart_zh.png", 0.9, 1.4, 11.5, "算法流程图");
  caption(s, "第一步找波参数（无正则、多起点）；第二步只用 L1 定支撑；第三步在支撑上重拟合。反问题直接从第二步进入。", 0.9, 6.35, 11.5);
  s.addNotes("门槛 1e-8：第一步残差到不了，就判定库里没有精确解，不报支撑。Burgers 原来 300 步迭代不够，所以对最好的起点再延长迭代最多 3000 步，其他规则不变。");
}

// ---------- 11 实验设置 ----------
{
  const s = content("实验设置：10 个问题，全部用精确解", "结果");
  const rows = [
    head(["方程", "正问题（已知方程和初边值）", "反问题（已知解的采样）"]),
    ["热方程 uₜ = uₓₓ", "初值 sin πx，14 列", "两个傅里叶模态"],
    ["波动方程 uₜₜ = uₓₓ", "sin πx cos πt，8 列", "两个傅里叶模态"],
    ["粘性 Burgers", "行波 c = 0.5；另用通用库", "Cole–Hopf 三激波合并，161×41"],
    ["Chafee–Infante", "稳态 tanh(x/√2)；另用通用库", "—"],
    ["KdV", "孤子 2sech²(x − 4t)；另用通用库", "双孤子，90×16"],
    ["平流扩散", "—", "两个高斯，σ₀ = 0.5 和 0.9"],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 1.5, 8.0, [2.1, 3.3, 2.6]), rowH: 0.52, fontSize: 14, objectName: "setup-table" });
  const crit = [
    ["判定选对", "L1 选出的项 = 真项"],
    ["残差门槛", "相对残差 ≤ 10⁻⁸ 才算库里有精确解"],
    ["运行环境", "Python 3.12.10 · NumPy 2.2.5\n双精度，本机运行"],
  ];
  crit.forEach(([h, d], i) => {
    const y = 1.5 + i * 1.6;
    card(s, 9.0, y, 3.7, 1.35, HEX.soft, "criterion-" + i);
    T(s, h, { x: 9.25, y: y + 0.18, w: 3.2, h: 0.4, fontSize: 17, bold: true, color: C.accent1 });
    T(s, d, { x: 9.25, y: y + 0.65, w: 3.25, h: 0.6, fontSize: 14 });
  });
  caption(s, "正问题不用解的内部采样；反问题的导数全部用解析式计算，不用差分。", 0.6, 5.35, 8.0);
  s.addNotes("热方程和波动方程的反问题必须用两个模态：单个模态满足更短的等价方程。Burgers 旧的两行波数据设计阵秩亏，所以主结果改用满秩的 Cole–Hopf 数据，旧数据放在论文附录。");
}

// ---------- 12 通用库结果 ----------
{
  const s = content("结果：同一个库拼出三个方程的解析解", "结果");
  const rows = [
    head(["方程", "L1 选出的项", "拼出的解析解", "(k, c, x₀)", "起点命中", "种子"]),
    ["KdV", "1, tanh²θ", "2 − 2tanh²θ = 2sech²(x − 4t)", "(1, 4, 0)", "13/40", "5/5"],
    ["Burgers", "1, tanh θ", "0.5 − 0.5 tanh θ", "(2.5, 0.5, 0.3)", "1/40", "5/5"],
    ["Chafee–Infante", "tanh θ", "tanh(x/√2)", "(1/√2, 0, 0)", "10/40", "5/5"],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 1.5, 12.1, [2.0, 1.9, 3.7, 1.9, 1.3, 1.3]), fontSize: 17, rowH: 0.6, objectName: "generic-results" });
  [["≤ 10⁻¹⁵", "波参数误差"], ["3 / 3", "λ 比例 10⁻²～10⁻⁴ 都通过"], ["≈ 36 分钟", "整体运行时间（本机）"]].forEach(([v, l], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.3, 3.8, 2.1, HEX.soft, "stat-" + i);
    T(s, v, { x: x + 0.3, y: 4.55, w: 3.3, h: 1.0, fontSize: 40, bold: true, color: C.accent1, valign: "middle" });
    T(s, l, { x: x + 0.3, y: 5.65, w: 3.3, h: 0.55, fontSize: 15, color: C.text2 });
  });
  s.addNotes("三个正例在 5 个随机种子下都选对；重拟合后波速、宽度、位置都恢复到舍入误差。Burgers 的起点命中率最低，40 个里只有 1 个。");
}

// ---------- 13 门槛 ----------
{
  const s = content("库里没有答案时，方法会说“没找到”", "结果");
  img(s, "gate", "gate_zh.png", 0.6, 1.5, 8.1, "第一步 40 个起点残差与门槛");
  caption(s, "每个情形 40 个随机起点的残差（排序）；虚线是门槛 10⁻⁸", 0.6, 4.85, 8.1);
  const items = [
    ["正例", "最好的起点都 < 2×10⁻¹⁰", C.accent3],
    ["反例 1：删掉 tanh²", "最好残差 0.178，拦下", C.accent2],
    ["反例 2：KdV 双孤子", "最好残差 0.306，拦下", C.accent2],
  ];
  items.forEach(([h, d, col], i) => {
    const y = 1.6 + i * 1.4;
    s.addShape(pres.shapes.OVAL, { x: 9.1, y: y + 0.08, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col } });
    T(s, h, { x: 9.6, y, w: 3.2, h: 0.45, fontSize: 17, bold: true, color: C.text2 });
    T(s, d, { x: 9.6, y: y + 0.5, w: 3.2, h: 0.5, fontSize: 15 });
  });
  card(s, 9.1, 5.85, 3.6, 0.95, HEX.paleBlue, "gate-note");
  T(s, "残差相差 7 个数量级\n不会硬凑出错的解", { x: 9.3, y: 5.93, w: 3.25, h: 0.8, fontSize: 15, color: C.text2, valign: "middle" });
  s.addNotes("这两个反例是专门设计的：删掉必需的 tanh² 列，或者用单一行波库去拟合双孤子。门槛把“找到了”和“只是拟合得不错”分开。");
}

// ---------- 14 x0 ----------
{
  const s = content("一个发现：位置参数要靠稀疏性才能定准", "结果");
  T(s, "tanh 库能近似“吸收”一个小平移", { x: 0.6, y: 1.5, w: 6.0, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  card(s, 0.6, 2.1, 6.0, 0.75, HEX.paleBlue, "shift-formula");
  T(s, "tanh(θ + ε) ≈ tanh θ + ε(1 − tanh²θ)", { x: 0.85, y: 2.15, w: 5.6, h: 0.65, fontSize: 20, fontFace: MATH, color: C.accent1, valign: "middle" });
  T(s, [
    { text: "x₀ 偏一点，常数项和 tanh² 项各补一点就能抵掉大部分误差", options: { bullet: true, breakLine: true } },
    { text: "种子 3：x₀ = 0.2895（真值 0.3），常数项 0.513、tanh² 项 −0.013，和公式吻合", options: { bullet: true, breakLine: true } },
    { text: "残差沿 x₀ 几乎是平的，优化 300 步走不到门槛", options: { bullet: true, breakLine: true } },
    { text: "L1 选出 {1, tanh θ} 后在支撑上重拟合，x₀ 精确回到 0.3（种子 0）", options: { bullet: true } },
  ], { x: 0.6, y: 3.1, w: 6.0, h: 3.6, fontSize: 15, paraSpaceAfter: 10 });
  T(s, "Burgers：延长迭代前后的最好残差", { x: 7.1, y: 1.5, w: 5.6, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
  const bad = (t) => ({ text: t, options: { color: HEX.badText, bold: true } });
  const rows = [
    head(["种子", "原来（300 步）", "延长后", "支撑"]),
    ["0", "1.4×10⁻¹⁰", "1.3×10⁻¹⁰", "{1, tanh}"], ["1", "3.4×10⁻⁹", "3.7×10⁻¹¹", "{1, tanh}"],
    ["2", "1.3×10⁻¹⁰", "1.3×10⁻¹⁰", "{1, tanh}"], ["3", bad("1.9×10⁻⁸"), "4.2×10⁻¹¹", "{1, tanh}"],
    ["4", bad("2.5×10⁻⁸"), "4.3×10⁻¹¹", "{1, tanh}"],
  ];
  s.addTable(rows, { ...tableOpts(7.1, 2.15, 5.6, [0.8, 1.7, 1.5, 1.6]), rowH: 0.48, objectName: "seed-table" });
  caption(s, "种子 3、4 原来卡在门槛上方；只延长迭代、不改其他规则后 5/5 通过", 7.1, 5.25, 5.6);
  s.addNotes("这不是落进错误的局部最优：k 和 c 都对，只有 x0 偏了约 0.01。只用最小二乘时 x0 定不准，稀疏的支撑把它唯一定了下来。重拟合目前只在种子 0 上跑过。");
}

// ---------- 15 汇总 ----------
{
  const s = content("汇总：正、反问题各 5 个，全部选对", "结果");
  const groups = [
    ["正问题（已知 PDE，求解析解）", ["热方程", "波动方程", "粘性 Burgers", "Chafee–Infante", "KdV"], "另有通用库：三个方程共用一个候选库"],
    ["反问题（已知解的采样，求方程）", ["热方程（双模态）", "波动方程（双模态）", "平流扩散", "粘性 Burgers", "KdV 双孤子"], "Burgers 和 KdV 双孤子用重加权 L1"],
  ];
  groups.forEach(([h, items, note], g) => {
    const x = 0.6 + g * 6.25;
    card(s, x, 1.5, 5.85, 5.2, g === 0 ? HEX.paleBlue : HEX.soft, "summary-" + g);
    T(s, h, { x: x + 0.3, y: 1.75, w: 5.3, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
    items.forEach((it, i) => {
      const y = 2.5 + i * 0.62;
      s.addShape(pres.shapes.OVAL, { x: x + 0.35, y: y + 0.07, w: 0.34, h: 0.34, fill: { color: HEX.okText }, line: { color: HEX.okText } });
      s.addText("✓", { isTextBox: true, x: x + 0.35, y: y + 0.07, w: 0.34, h: 0.34, margin: 0, align: "center", valign: "middle", fontSize: 12, bold: true, color: C.background1 });
      T(s, it, { x: x + 0.9, y, w: 4.6, h: 0.48, fontSize: 16, valign: "middle" });
    });
    T(s, note, { x: x + 0.3, y: 5.75, w: 5.3, h: 0.8, fontSize: 13, color: HEX.muted });
  });
  s.addNotes("选出的项都等于真项，重拟合系数和真值一致到 1e-15。数据全部是精确解，没有噪声。");
}

// ---------- 16 补充实验：加噪声要改两处 ----------
{
  const s = content("加噪声后，反问题改了两处", "补充实验");
  const cards = [
    ["弱形式：不对噪声数据求导", "∫φ ∂ₓᵏ(uᵐ) = (−1)ᵏ ∫(∂ₓᵏφ) uᵐ", "φ = (1 − s²)⁶ 的“小鼓包”，导数解析计算；积分把噪声平均掉。直接差分求导时 1% 噪声 uₓₓ 误差已达 45%"],
    ["统一判据：只比谁能产生正确组合", "L(S) = ‖A(w_S − w_LS)‖/‖A w_LS‖ + |S|/p", "前一项：少了真项会变大；后一项：多了假项会变大。所有方法共用，比较才公平。BIC 在无噪声时会多选项，已弃用"],
  ];
  cards.forEach(([h, f, d], i) => {
    const x = 0.6 + i * 6.15;
    card(s, x, 1.5, 5.95, 3.5, i === 0 ? HEX.paleBlue : HEX.soft, "noise-change-" + i);
    badge(s, x + 0.3, 1.8, i + 1, C.accent1);
    T(s, h, { x: x + 0.95, y: 1.83, w: 4.8, h: 0.45, fontSize: 19, bold: true, color: C.text2 });
    T(s, f, { x: x + 0.3, y: 2.6, w: 5.4, h: 0.7, fontSize: 19, fontFace: MATH, color: C.accent1 });
    T(s, d, { x: x + 0.3, y: 3.45, w: 5.4, h: 1.4, fontSize: 16 });
  });
  card(s, 0.6, 5.3, 12.1, 1.05, "FDEEE7", "noise-def");
  T(s, [
    { text: "噪声：", options: { bold: true, color: C.accent2 } },
    { text: "u + σ·std(u)·N(0,1)，σ 取 0～10%；反问题 5 个方程 × 9 个噪声 × 20 个种子，正问题初边值加噪声 0.1%～10% × 5 个种子。" },
  ], { x: 0.9, y: 5.4, w: 11.5, h: 0.85, fontSize: 16, color: C.text2, valign: "middle" });
  s.addNotes("候选库仍然是 13 列 ∂ₓᵏ(uᵐ)，所以 Burgers 的 uuₓ 写成 ½(u²)ₓ。选项仍由 L1 / 加权 L1 的 λ 路径产生，判据只在路径上的组合里挑。");
}

// ---------- 17 反问题噪声结果 ----------
{
  const s = content("反问题：加权 L1 到 5% 噪声全部选对", "补充实验");
  img(s, "nInv", "noise_inverse_zh.png", 0.6, 1.35, 12.1, "反问题各方法选对率随噪声变化");
  [["5 / 5", "个方程到 5% 噪声全部选对（加权 L1）"], ["0.12%", "Burgers 1% 噪声时的系数误差；2～3% 以内与噪声成正比"], ["10%", "热方程、Burgers、KdV 连穷举都错：数据信息不够"]].forEach(([v, l], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.85, 3.8, 1.85, HEX.soft, "inv-stat-" + i);
    T(s, v, { x: x + 0.3, y: 5.0, w: 3.3, h: 0.8, fontSize: 34, bold: true, color: C.accent1, valign: "middle" });
    T(s, l, { x: x + 0.3, y: 5.85, w: 3.3, h: 0.75, fontSize: 14, color: C.text2 });
  });
  s.addNotes("普通 L1 在 Burgers（有噪声）和 KdV（任何噪声）上一直是 0，总多留 uₓ，和前面不可表示条件的分析一致。STRidge 在 Burgers 2～3% 就开始掉。7% 时加权 L1 在 Burgers、KdV 上降到 0.75，遗传算法 0.90。");
}

// ---------- 18 正问题噪声结果 ----------
{
  const s = content("正问题：真解全部找对，残差门槛需要收紧", "补充实验");
  img(s, "nFwd", "noise_forward_zh.png", 0.8, 1.3, 5.9, "正问题残差与噪声水平之比");
  caption(s, "有噪声时残差到不了 10⁻⁸，门槛改为“噪声带来的残差 × 3”（差异原理，假定噪声大小已知）", 0.6, 6.4, 7.4);
  const ok = (t) => ({ text: t, options: { color: HEX.okText, bold: true, align: "center" } });
  const bad = (t) => ({ text: t, options: { color: HEX.badText, bold: true, align: "center" } });
  const rows = [
    head(["情形", "1%", "5%", "10%"]),
    ["3 个真解", ok("5/5 对"), ok("5/5 对"), ok("5/5 对")],
    ["删 tanh²", ok("拒绝"), bad("放过"), bad("放过")],
    ["双孤子", ok("拒绝"), ok("拒绝"), bad("放过")],
  ];
  s.addTable(rows, { ...tableOpts(7.6, 1.5, 5.1, [1.8, 1.1, 1.1, 1.1]), rowH: 0.5, objectName: "fwd-noise-table" });
  T(s, "10% 噪声时 KdV 系数误差 0.5%；系数和波参数误差都与噪声成正比", { x: 7.6, y: 3.7, w: 5.1, h: 0.7, fontSize: 15 });
  card(s, 7.6, 4.6, 5.1, 1.65, HEX.paleBlue, "gate-decision");
  T(s, [
    { text: "待定：门槛倍数\n", options: { bold: true, color: C.accent1 } },
    { text: "真解残差稳定在噪声的 0.88～1.10 倍；改成 1.5 倍后只剩“删 tanh²”在 10% 时放过 2/5，真解仍全部通过" },
  ], { x: 7.85, y: 4.72, w: 4.65, h: 1.45, fontSize: 14, color: C.text2 });
  s.addNotes("选项规则也改用反问题的判据：论文原来的“过半”规则在噪声下会失效，Burgers 5% 时 0/5。1.5 倍是用已存残差事后算的，还没重跑。");
}

// ---------- 19 与自写遗传算法 ----------
{
  const s = content("与遗传算法比：准确率打平，L1 便宜约 30 倍", "补充实验");
  T(s, "自写遗传算法：同一个判据，只比搜索方式", { x: 0.6, y: 1.45, w: 7.0, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  T(s, "每个个体是一串 0/1（每位对应库里一项）；种群 40，跑 60 代；锦标赛选择、均匀交叉、按位变异，保留最好的 2 个。", { x: 0.6, y: 1.95, w: 7.0, h: 0.9, fontSize: 15 });
  const rows = [
    head(["", "加权 L1", "自写遗传算法"]),
    ["5% 噪声以内", "5 个方程全对", "KdV 偶尔卡在 {uₓ}（1～2/20）"],
    ["7% 噪声 Burgers / KdV", "0.75 / 0.75", "0.90 / 0.90"],
    ["每次评估判据次数", "约 15", "约 450"],
    ["耗时（中位数）", "0.003 s", "0.09 s"],
    ["结果", "固定", "随种子变化"],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 3.0, 7.0, [2.4, 1.9, 2.7]), rowH: 0.5, objectName: "ga-table" });
  card(s, 8.0, 1.5, 4.7, 5.2, HEX.paleBlue, "ga-takeaway");
  T(s, "怎么看", { x: 8.3, y: 1.75, w: 4.1, h: 0.45, fontSize: 19, bold: true, color: C.accent1 });
  T(s, [
    { text: "低噪声下两者一样准，L1 计算量小 30 倍、结果确定", options: { bullet: true, breakLine: true } },
    { text: "高噪声或库里项很相似时，遗传算法搜得更广，更接近穷举", options: { bullet: true, breakLine: true } },
    { text: "库只有 13 项时穷举也可行；库变大后穷举不可行，L1 的优势会更明显（尚未用大库验证）", options: { bullet: true, breakLine: true } },
    { text: "建议：L1 定支撑为主，遗传算法作对照", options: { bullet: true, bold: true } },
  ], { x: 8.3, y: 2.35, w: 4.15, h: 4.2, fontSize: 15, paraSpaceAfter: 10, color: C.text2 });
  s.addNotes("穷举是所有不超过 6 项的组合，共 4096 个，作为判据的上界。遗传算法卡在 {uₓ} 是搜索失败：同一判据下穷举是对的。");
}

// ---------- 20 与 SGA-PDE ----------
{
  const s = content("与 SGA-PDE 比：有噪声时它全错，我们 1% 噪声全对", "补充实验");
  img(s, "nSga", "noise_sga_zh.png", 0.6, 1.35, 12.1, "与 SGA-PDE 的选对率对比");
  caption(s, "SGA-PDE 论文的 5 个方程全部跑了；C–I = Chafee–Infante；divide* = uₜ = −uₓ/x + 0.25uₓₓ，需在库里加 4 个含 x 的项", 0.6, 4.5, 12.1);
  const notes = [
    ["SGA-PDE 是什么", "不预设候选库，用遗传算法进化符号树，有限差分求导。默认参数：种群 20、深度 4、100 代，每次限时 30 分钟"],
    ["我们的局限", "uₓ/x 不在基础库里时所有方法都找不到；1% 噪声的 divide* 上只有遗传算法和穷举对，加权 L1 错"],
    ["公平性", "SGA 论文报告这些方程都能找到，我们未复现：关了它的神经网络去噪，30 分钟内多数只跑到 24～94 代"],
  ];
  notes.forEach(([h, d], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 5.0, 3.8, 1.75, i === 2 ? "FDEEE7" : HEX.soft, "sga-note-" + i);
    T(s, h, { x: x + 0.25, y: 5.1, w: 3.3, h: 0.4, fontSize: 15, bold: true, color: i === 2 ? C.accent2 : C.accent1 });
    T(s, d, { x: x + 0.25, y: 5.5, w: 3.35, h: 1.2, fontSize: 12 });
  });
  s.addNotes("同一份带噪声数据同时给两边。SGA 无噪声时 Burgers、Chafee–Infante 对，两组 KdV 都没找到；1% 和 5% 噪声下 0/36。5% 时我们失败的几组网格较小，连穷举也错；大网格上 5% 全对。论文里写成“不去噪、30 分钟预算下”的比较。");
}

// ---------- 21 局限与下一步 ----------
{
  section("总结");
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sectionName });
  s.addText("局限与下一步", { isTextBox: true, x: 0.8, y: 0.8, w: 11.7, h: 0.9, fontSize: 40, bold: true, color: C.background1, margin: 0 });
  const cols = [
    ["目前的局限", ["候选库必须包含真项（uₓ/x 需另加）", "通用库只覆盖单一行波，双孤子需要另设库", "正问题噪声门槛假定噪声大小已知", "高噪声或库项相似时，加权 L1 不如遗传算法"]],
    ["下一步", ["确定正问题噪声门槛倍数（3 或 1.5）", "论文加入噪声与遗传算法比较两节", "设计多波（ln F 或两个波变量）的候选库", "确定作者、目标会议和页数"]],
  ];
  cols.forEach(([h, items], i) => {
    const x = 0.8 + i * 6.1;
    card(s, x, 2.05, 5.7, 4.8, "26344F", "summary-col-" + i);
    T(s, h, { x: x + 0.35, y: 2.3, w: 5.0, h: 0.6, fontSize: 26, bold: true, color: i === 0 ? "F6B08F" : "9CC3F2" });
    T(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
      { x: x + 0.35, y: 3.1, w: 5.0, h: 3.6, fontSize: 19, color: "E6ECF5", paraSpaceAfter: 16 });
  });
  s.addNotes("论文初稿已写好，英文、NeurIPS 2026 模板、10 页，代码和结果都在 GitHub 的 exp/generic-library 分支。");
}

(async () => {
  const out = path.join(__dirname, "PDE_L1_进展汇报.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();
