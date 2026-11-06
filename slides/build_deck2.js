// 汇报 PPT 第二版：重点讲算法的迭代过程、核心公式、与 SGA-PDE 的对比和方法优劣。
// 数字全部取自结果 JSON。旧版 build_deck.js 保留不动。运行：cd slides && node build_deck2.js
// 需要环境变量 PPTX_SKILL（pptx skill 目录）和 NODE_PATH（本目录 node_modules）。
// 配图：make_figures_zh.py、make_noise_figures_zh.py、make_deck2_figures_zh.py。
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
const HEX = { blue: "2A78D6", paleBlue: "DCEAFB", soft: "F3F6FA", grid: "D9DDE3", muted: "6B6A66", okText: "13865E", badText: "C2410C", warm: "FDEEE7", amber: "B7791F" };
const MATH = "Cambria Math";
const IMG = (f) => path.join(__dirname, "img", f);
const SIZE = {
  overview: [1828, 908], flowchart: [1908, 795], gate: [2024, 807], inverse: [1551, 801],
  nInv: [3427, 923], nFwd: [1517, 1286], nSga: [3434, 880], cost: [2057, 989], coef: [1555, 1035], robust: [1791, 816],
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "用 L1 稀疏库求 PDE 解析解与控制方程";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- 版式与小部件（与 build_deck.js 相同） ----------
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
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 28, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
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
  s.addText(String(n), { isTextBox: true, x, y, w: d, h: d, margin: 0, align: "center", valign: "middle", fontSize: d > 0.6 ? 22 : 15, bold: true, color: C.background1 });
}
function chip(s, x, y, w, h, text, on, fs = 14, dark = false) {
  const fill = on ? (dark ? HEX.blue : "FFFFFF") : (dark ? "2C3A57" : "F0EFEC");
  const line = on ? HEX.blue : (dark ? "4A5873" : "B8B7B2");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: line, width: 1 } });
  s.addText(text, { isTextBox: true, x, y, w, h, margin: 0, align: "center", valign: "middle", fontSize: fs, fontFace: MATH,
    color: on ? (dark ? "FFFFFF" : "141B2D") : (dark ? "8D9AB3" : "8A8984") });
}
// 公式条：浅蓝底、Cambria Math、居中
function formula(s, text, x, y, w, h = 0.75, fs = 20) {
  card(s, x, y, w, h, HEX.paleBlue);
  T(s, text, { x: x + 0.2, y, w: w - 0.4, h, fontSize: fs, fontFace: MATH, color: C.text2, align: "center", valign: "middle" });
}
const img = (s, key, file, x, y, w, alt) => s.addImage({ path: IMG(file), x, y, w, h: w * SIZE[key][1] / SIZE[key][0], altText: alt });
const tableOpts = (x, y, w, colW) => ({ x, y, w, colW, fontSize: 13, color: C.text1, border: { type: "solid", pt: 0.75, color: HEX.grid }, valign: "middle", margin: [0.04, 0.08, 0.04, 0.08] });
const head = (cells) => cells.map((t) => ({ text: t, options: { bold: true, fill: { color: HEX.paleBlue }, color: C.text2 } }));
const caption = (s, text, x, y, w) => T(s, text, { x, y, w, h: 0.4, fontSize: 12, color: HEX.muted });
const bullets = (items) => items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } }));
const ok = (t) => ({ text: t, options: { color: HEX.okText, bold: true } });
const bad = (t) => ({ text: t, options: { color: HEX.badText, bold: true } });
const mid = (t) => ({ text: t, options: { color: HEX.amber, bold: true } });

// ================= 开场 =================
section("开场");
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sectionName });
  s.addText("用 L1 稀疏库求 PDE 的解析解与控制方程", { placeholder: "title" });
  s.addText("算法迭代 · 核心公式 · 噪声分析 · 与 SGA-PDE 对比\n进展汇报 · 2026-10-30", { placeholder: "body" });
  const items = ["1", "tanh θ", "tanh² θ", "tanh³ θ", "tanh⁴ θ", "sech θ", "e^(−θ²)", "θ"];
  const coef = ["2", "0", "−2", "0", "0", "0", "0", "0"];
  items.forEach((t, i) => {
    const x = 8.2 + (i % 2) * 2.15, y = 1.45 + Math.floor(i / 2) * 1.05;
    chip(s, x, y, 1.9, 0.62, t, coef[i] !== "0", 16, true);
    s.addText(coef[i], { isTextBox: true, x: x + 1.9 - 0.45, y: y - 0.32, w: 0.45, h: 0.3, margin: 0, align: "right", fontSize: 12, bold: coef[i] !== "0", color: coef[i] !== "0" ? "9CC3F2" : "6F7D97" });
  });
  s.addText("u = 2 − 2tanh²θ = 2sech²(x − 4t)", { isTextBox: true, x: 8.2, y: 5.85, w: 4.05, h: 0.5, margin: 0, align: "center", fontSize: 16, fontFace: MATH, color: "C9D6EA" });
  s.addNotes("这一版重点讲算法是怎么一步步迭代出来的、每一步的核心公式，以及按老师意见补的噪声实验和与遗传算法 SGA-PDE 的对比。");
}
{
  const s = content("汇报内容");
  const parts = [
    ["背景", "老师的思路、两个方向、相关工作"],
    ["算法与迭代", "五次迭代：每次遇到什么问题、怎么改、核心公式"],
    ["实验结果", "无噪声、反例门槛、加高斯噪声的误差分析"],
    ["对比", "自写遗传算法、SGA-PDE、各方法优劣"],
    ["总结", "结论、局限与下一步"],
  ];
  parts.forEach(([h, d], i) => {
    const x = 0.6 + i * 2.45;
    card(s, x, 1.7, 2.25, 3.9, i === 1 || i === 3 ? HEX.paleBlue : HEX.soft, "agenda-" + i);
    T(s, String(i + 1).padStart(2, "0"), { x: x + 0.25, y: 2.0, w: 1.8, h: 1.0, fontSize: 44, bold: true, color: C.accent1 });
    T(s, h, { x: x + 0.25, y: 3.2, w: 1.85, h: 0.6, fontSize: 22, bold: true, color: C.text2 });
    T(s, d, { x: x + 0.25, y: 4.0, w: 1.8, h: 1.5, fontSize: 14 });
  });
  s.addNotes("第二、四部分是这次的重点。");
}

// ================= 背景 =================
{
  const s = content("思路：把“猜解 / 猜方程”变成“从候选库里选项”", "背景");
  T(s, "先列很多候选项，每项配一个未知系数，用 L1 把绝大多数系数压成 0，剩下的几项就是答案。", { x: 0.6, y: 1.45, w: 12.1, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
  const rows = [
    head(["方向", "已知", "要找", "候选库里放什么", "例子"]),
    ["正问题：求解析解", "方程 + 初值 / 边界数据", "解 u(x,t) 的表达式", "函数项 1、tanh θ、tanh² θ、sech θ …", "KdV → 2 − 2tanh²θ"],
    ["反问题：求方程", "解的数据 u(x,t)", "控制方程 uₜ = …", "导数项 u、uₓ、uₓₓ、uuₓ、u³ …", "KdV → uₜ = −6uuₓ − uₓₓₓ"],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 2.2, 12.1, [2.2, 2.3, 2.1, 3.3, 2.2]), fontSize: 15, rowH: 0.62, objectName: "two-directions" });
  img(s, "overview", "overview_zh.png", 2.4, 4.3, 5.6, "正问题与反问题示意图");
  card(s, 8.4, 4.4, 4.3, 2.4, HEX.soft, "judge");
  T(s, "怎么算“真的找到了”", { x: 8.65, y: 4.55, w: 3.9, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  T(s, bullets(["选出的项 = 真项，一项不多一项不少", "代回方程，残差降到 10⁻⁸ 以下（无噪声）", "多个随机种子都成立"]), { x: 8.65, y: 5.05, w: 3.9, h: 1.7, fontSize: 14, paraSpaceAfter: 6 });
  s.addNotes("两个方向用同一套“候选库 + L1 稀疏选择”工具，只是候选库里放的东西不同。");
}
{
  const s = content("相关工作：三类方法，我们属于“固定候选库”", "背景");
  const cols = [
    ["固定候选库", "SINDy、PDE-FIND、本文", "先列好候选项，再做稀疏回归选项", "快、可解释；真项必须在库里", true],
    ["可扩展候选库", "DLGA-PDE（Xu 等 2020）", "遗传算法往库里加新项，再稀疏回归", "库能长大；搜索变慢", false],
    ["开放形式", "SGA-PDE（Chen 等，PRR 2022）", "遗传算法直接进化符号树，不预设库", "能发现库外项；慢、结果随机", false],
  ];
  cols.forEach(([h, who, how, pc, ours], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.5, 3.8, 3.9, ours ? HEX.paleBlue : HEX.soft, "family-" + i);
    T(s, h, { x: x + 0.3, y: 1.75, w: 3.3, h: 0.5, fontSize: 21, bold: true, color: C.text2 });
    T(s, who, { x: x + 0.3, y: 2.35, w: 3.3, h: 0.45, fontSize: 14, color: C.accent1 });
    T(s, how, { x: x + 0.3, y: 2.95, w: 3.3, h: 1.1, fontSize: 15 });
    T(s, pc, { x: x + 0.3, y: 4.2, w: 3.3, h: 1.0, fontSize: 14, color: HEX.muted });
  });
  card(s, 0.6, 5.65, 12.1, 1.05, HEX.warm, "our-position");
  T(s, [
    { text: "我们的不同：", options: { bold: true, color: C.accent2 } },
    { text: "同一套工具同时做正问题（已知方程求闭式解，文献少）和反问题；用精确 L1 路径解释并修掉了普通 L1 “多留 uₓ”的现象。" },
  ], { x: 0.9, y: 5.75, w: 11.5, h: 0.85, fontSize: 16, color: C.text2, valign: "middle" });
  s.addNotes("三类方法的划分参考陈云天老师《AI 应用于科学》讲义。我们和 SGA-PDE 分属两端：一个快而受限于库，一个灵活但慢。");
}

// ================= 算法与迭代 =================
{
  const s = content("算法演进：五次迭代，每次解决一个具体问题", "算法与迭代");
  const steps = [
    ["v0", "离散库 + 普通 L1", "问题：反问题总多留 uₓ", "精确 LARS 复现，排除“没收敛”"],
    ["v1", "加权 L1（IRL1）", "问题：uₓ 去不掉", "按上一轮系数加权，两轮去掉"],
    ["v2", "通用库 + 连续波参数", "问题：旧库里本来就放了答案", "tanh 展开库，(k, c, x₀) 连续搜"],
    ["v3", "多起点 LM + 延长迭代", "问题：Burgers 种子只过 3/5", "最好起点再迭代 3000 步，5/5"],
    ["v4", "弱形式 + 统一判据", "问题：加噪声后求导误差 45%", "积分代替求导；MSTLS 判据选项"],
  ];
  const w = 2.3, gap = 0.15;
  steps.forEach(([tag, h, p, f], i) => {
    const x = 0.6 + i * (w + gap);
    card(s, x, 1.55, w, 4.4, i === 4 ? HEX.paleBlue : HEX.soft, "evo-" + i);
    T(s, tag, { x: x + 0.2, y: 1.7, w: 1.0, h: 0.5, fontSize: 24, bold: true, color: C.accent1 });
    T(s, h, { x: x + 0.2, y: 2.3, w: w - 0.35, h: 0.9, fontSize: 16, bold: true, color: C.text2 });
    T(s, p, { x: x + 0.2, y: 3.3, w: w - 0.35, h: 1.1, fontSize: 13, color: HEX.badText });
    T(s, f, { x: x + 0.2, y: 4.55, w: w - 0.35, h: 1.2, fontSize: 13, color: HEX.okText });
    if (i < steps.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + w - 0.02, y: 3.6, w: gap + 0.04, h: 0.25, fill: { color: "B8B7B2" }, line: { color: "B8B7B2" } });
  });
  T(s, [
    { text: "不变的约束：", options: { bold: true, color: C.text2 } },
    { text: "哪几项留下只由 L1 决定；支撑阈值 10⁻⁵ 不放宽；重拟合只读系数、不增删项。" },
  ], { x: 0.6, y: 6.2, w: 12.1, h: 0.5, fontSize: 15 });
  s.addNotes("红字是那一版遇到的问题，绿字是下一版的修法。后面几页逐个展开每一步的公式。");
}
{
  const s = content("核心公式①：L1 稀疏回归（Lasso）", "算法与迭代");
  formula(s, "min_ξ  ½‖Aξ − b‖₂²  +  λ Σⱼ |ξⱼ|", 0.6, 1.5, 12.1, 0.85, 26);
  const cols = [
    ["A、b、ξ 是什么", ["A 的每一列 = 一个候选项在所有网格点上的值", "b = 要拟合的量（反问题里是 uₜ）", "ξ = 每个候选项的系数，绝大多数应为 0"]],
    ["λ 的作用：罚款单价", ["λ 大：每留一项都要多付罚款 → 留下的项少", "λ 小：罚款便宜 → 为了误差小会多用项", "λ 从大往小扫，项逐个“进场”= λ 路径"]],
    ["为什么系数能精确为 0", ["|ξ| 在 0 处有尖角，最优点会卡在 0 上", "平方罚（岭回归）没有尖角，只会变小", "所以 L1 能“选项”，岭回归不能"]],
  ];
  cols.forEach(([h, items], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 2.65, 3.8, 3.4, HEX.soft, "lasso-" + i);
    T(s, h, { x: x + 0.3, y: 2.85, w: 3.3, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
    T(s, bullets(items), { x: x + 0.3, y: 3.45, w: 3.3, h: 2.5, fontSize: 14, paraSpaceAfter: 8 });
  });
  caption(s, "正问题里残差 r(ξ) 非线性，用 Gauss–Newton 线性化，每步解一个上面形式的 L1 子问题", 0.6, 6.3, 12.1);
  s.addNotes("Lasso 是 Tibshirani 1996 提出的。这里的 A 在反问题里是候选导数项，在正问题里是候选函数项对系数的雅可比。");
}
{
  const s = content("核心公式②：LARS 精确走完整条 λ 路径", "算法与迭代");
  formula(s, "|Aⱼᵀ(b − Aξ)| ≤ λ，且 ξⱼ ≠ 0 时取等号", 0.6, 1.5, 6.4, 0.75, 19);
  T(s, "最优解的条件（KKT）：在场的项与残差的相关度都等于 λ，场外的都不超过 λ。", { x: 0.6, y: 2.4, w: 6.4, h: 0.7, fontSize: 14, color: HEX.muted });
  const steps = [
    "λ 很大时系数全为 0；与残差最相关的列第一个进场",
    "在场系数沿“等角方向”一起变：各列与残差的相关度同速下降",
    "场外某列追平 → 进场；在场系数碰到 0 → 退场",
    "在这个转折点重算方向，直到 λ 降到下限",
  ];
  steps.forEach((t, i) => {
    const y = 3.2 + i * 0.8;
    badge(s, 0.6, y, i + 1, C.accent1, 0.45);
    T(s, t, { x: 1.25, y: y + 0.03, w: 5.75, h: 0.7, fontSize: 15 });
  });
  img(s, "inverse", "inverse_path_zh.png", 7.4, 1.45, 5.3, "普通 L1 沿 λ 路径的 u_x 系数");
  card(s, 7.4, 4.45, 5.3, 2.25, HEX.paleBlue, "lars-why");
  T(s, "为什么用它", { x: 7.65, y: 4.6, w: 4.8, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  T(s, bullets(["路径是分段直线，只有十几个转折点，算出来就是精确解（KKT 误差 < 5×10⁻⁹）", "排除“求解器没收敛”的解释", "每段的非零项就是一个候选组合（约 15 个）"]), { x: 7.65, y: 5.05, w: 4.85, h: 1.6, fontSize: 13, paraSpaceAfter: 4 });
  s.addNotes("LARS 是 Efron 等 2004 年的最小角回归。右图是普通 L1 在 KdV 上 uₓ 系数沿路径的变化：随 λ 变小而变小，但始终不为 0。");
}
{
  const s = content("迭代一 → 问题：普通 L1 总多留一个 uₓ", "算法与迭代");
  formula(s, "‖ A_Sᶜᵀ A_S (A_Sᵀ A_S)⁻¹ sign(ξ_S) ‖_∞  <  1", 0.6, 1.5, 7.2, 0.8, 22);
  T(s, "不可表示条件：S = 真项，Sᶜ = 错项。错列不能太像“真项的组合”，否则 L1 一定会选它。", { x: 0.6, y: 2.45, w: 7.2, h: 0.8, fontSize: 15 });
  card(s, 0.6, 3.4, 7.2, 3.3, HEX.soft, "irrep-why");
  T(s, "通俗地说", { x: 0.9, y: 3.6, w: 6.6, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
  T(s, bullets([
    "行波数据里，uₓ 这一列和真项（如 uuₓ）的组合长得非常像",
    "L1 发现：用 uₓ 顶替一部分真项，罚款更便宜、误差差不多",
    "于是不管 λ 取多小，uₓ 都会被留下；这是 L1 本身的局限，不是程序错误",
    "uₓ 系数与 λ 成正比地变小，所以旧的 10⁻⁸ 残差门槛整条路径都到不了",
  ]), { x: 0.9, y: 4.1, w: 6.7, h: 2.5, fontSize: 14, paraSpaceAfter: 6 });
  T(s, "uₓ 那一列的指标（> 1 就会被选）", { x: 8.3, y: 1.5, w: 4.4, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
  [["2.77", "KdV 双孤子"], ["1.39", "Burgers，Cole–Hopf"], ["1.52", "Burgers，两行波"]].forEach(([v, l], i) => {
    const y = 2.1 + i * 1.1;
    T(s, v, { x: 8.3, y, w: 1.9, h: 0.9, fontSize: 42, bold: true, color: C.accent2, valign: "middle" });
    T(s, l, { x: 10.3, y: y + 0.2, w: 2.4, h: 0.5, fontSize: 15, valign: "middle" });
  });
  card(s, 8.3, 5.55, 4.4, 1.15, HEX.warm, "irrep-other");
  T(s, "其余错列的指标都 < 1，正好解释了为什么只多留 uₓ 一项", { x: 8.5, y: 5.65, w: 4.0, h: 0.95, fontSize: 14, color: C.text2, valign: "middle" });
  s.addNotes("不可表示条件来自 Zhao & Yu 2006 等。这个指标只取决于数据设计阵，与 λ、与求解器都无关。");
}
{
  const s = content("迭代二：加权 L1（IRL1），两轮去掉 uₓ", "算法与迭代");
  formula(s, "min_ξ ½‖Aξ − b‖² + λ Σⱼ wⱼ|ξⱼ|，   wⱼ⁽ᵗ⁺¹⁾ = 1 / (|ξⱼ⁽ᵗ⁾| + ε)", 0.6, 1.5, 12.1, 0.8, 21);
  const flow = [
    ["第 0 轮：普通 L1（λ = 10⁻³ λ_max）", "KdV 选 {uuₓ, uₓₓₓ, uₓ, uₓₓ}；Burgers 选 {uuₓ, uₓₓ, uₓ}"],
    ["改权重", "系数大的项罚得轻，系数小的项（uₓ）罚得重"],
    ["第 1 轮起只剩真项，支撑不再变化就停", "KdV：uuₓ (−6)、uₓₓₓ (−1)；Burgers：uuₓ (−1)、uₓₓ (0.1)"],
  ];
  flow.forEach(([h, d], i) => {
    const y = 2.6 + i * 1.35;
    card(s, 0.6, y, 6.6, 1.15, i === 2 ? HEX.paleBlue : HEX.soft, "irl1-" + i);
    badge(s, 0.85, y + 0.32, i + 1, i === 2 ? C.accent1 : C.accent6, 0.45);
    T(s, h, { x: 1.5, y: y + 0.12, w: 5.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
    T(s, d, { x: 1.5, y: y + 0.55, w: 5.6, h: 0.5, fontSize: 13 });
  });
  T(s, "第 0 轮 λ 比例的敏感性（每格 4 个 ε）", { x: 7.6, y: 2.6, w: 5.1, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  const P = (t) => (t === "通过" ? ok(t) : bad(t));
  const rows = [
    head(["λ/λ_max", "KdV", "Burgers CH", "两行波"]),
    ["10⁻¹", P("失败"), P("失败"), P("失败")], ["10⁻²", P("失败"), P("通过"), P("通过")], ["10⁻³", P("通过"), P("通过"), P("通过")],
    ["10⁻⁴", P("通过"), P("通过"), P("通过")], ["10⁻⁵", P("通过"), P("通过"), P("通过")],
  ];
  s.addTable(rows, { ...tableOpts(7.6, 3.1, 5.1, [1.2, 1.2, 1.45, 1.25]), rowH: 0.42, objectName: "sens" });
  caption(s, "比例太大时第 0 轮漏掉真项，重加权会锁死错的支撑；论文固定 10⁻³", 7.6, 5.75, 5.1);
  s.addNotes("权重只来自上一轮 L1 解，不用岭回归：岭回归会把能量分给相关的 uₓ，反而让 uₓ 权重变小。加权 L1 仍然是 L1：把每列除以权重就变回普通 L1，照样用 LARS 求解。");
}
{
  const s = content("迭代三：正问题改用通用库，库里不直接放答案", "算法与迭代");
  formula(s, "u(x,t) ≈ Σⱼ ξⱼ fⱼ(θ)，   θ = k (x − ct − x₀)", 0.6, 1.5, 12.1, 0.8, 24);
  T(s, "旧库：针对真解设计", { x: 0.6, y: 2.6, w: 5.8, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  ["2sech²(x−4t)", "c = 1 孤子", "c = 2 孤子", "c = 9 孤子"].forEach((t, i) => chip(s, 0.6 + (i % 2) * 2.9, 3.2 + Math.floor(i / 2) * 0.68, 2.7, 0.52, t, i === 0, 14));
  T(s, bullets(["库里直接有真解那一列", "波速预先离散成几列", "L1 只是在几列里挑一列"]), { x: 0.6, y: 4.65, w: 5.8, h: 1.6, fontSize: 15, paraSpaceAfter: 6 });
  T(s, "新库：三个方程共用", { x: 6.9, y: 2.6, w: 5.8, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  ["1", "tanh θ", "tanh² θ", "tanh³ θ", "tanh⁴ θ", "sech θ", "e^(−θ²)", "θ"].forEach((c, i) => chip(s, 6.9 + (i % 4) * 1.45, 3.2 + Math.floor(i / 4) * 0.68, 1.3, 0.52, c, i < 5, 14));
  T(s, bullets(["前五项：tanh 展开法常用项；后三项：干扰项", "k、c、x₀ 不告诉程序，连续搜索", "没有任何一列等于真解，要几列拼出来"]), { x: 6.9, y: 4.65, w: 5.8, h: 1.6, fontSize: 15, paraSpaceAfter: 6 });
  formula(s, "min  ½‖r(ξ, k, c, x₀)‖² + λ Σⱼ wⱼ|ξⱼ|，  r = [PDE 配点残差；边界误差；初值误差]", 0.6, 6.15, 12.1, 0.6, 15);
  s.addNotes("这是为了回答“库里本来就放了答案”的质疑。搜索范围：k 在 0.1 到 10，c 在 −8 到 8，x₀ 在整个计算域。三块残差等权，不用解的内部采样。");
}
{
  const s = content("核心公式③：多起点 Levenberg–Marquardt 找波参数", "算法与迭代");
  formula(s, "(JᵀJ + μ·diag(JᵀJ)) δ = −Jᵀ r，   z ← z + δ", 0.6, 1.5, 7.0, 0.8, 21);
  T(s, "z = (ξ, k, c, x₀)，J = 残差对 z 的雅可比", { x: 0.6, y: 2.4, w: 7.0, h: 0.4, fontSize: 14, color: HEX.muted });
  card(s, 0.6, 2.95, 7.0, 2.2, HEX.soft, "lm-idea");
  T(s, "通俗地说：在山谷里找最低点", { x: 0.9, y: 3.1, w: 6.4, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  T(s, bullets(["μ 小 ≈ 高斯–牛顿：按地形曲率直接“跳”到估计的最低点，近处很快", "μ 大 ≈ 梯度下降：顺最陡方向小步走，稳但慢", "这一步误差变小就减小 μ（更敢跳），变大就增大 μ（退回小步）"]), { x: 0.9, y: 3.6, w: 6.5, h: 1.5, fontSize: 14, paraSpaceAfter: 4 });
  card(s, 0.6, 5.35, 7.0, 1.35, HEX.warm, "lm-weak");
  T(s, "弱点：只找附近的低谷 → 从 40 个随机起点各跑一遍取最好；v3 再对最好的起点继续迭代最多 3000 步", { x: 0.85, y: 5.45, w: 6.5, h: 1.15, fontSize: 14, color: C.text2, valign: "middle" });
  T(s, "Burgers：延长迭代前后的最好残差", { x: 8.0, y: 1.5, w: 4.7, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
  const rows = [
    head(["种子", "300 步", "延长后"]),
    ["0", "1.4×10⁻¹⁰", "1.3×10⁻¹⁰"], ["1", "3.4×10⁻⁹", "3.7×10⁻¹¹"], ["2", "1.3×10⁻¹⁰", "1.3×10⁻¹⁰"],
    ["3", bad("1.9×10⁻⁸"), ok("4.2×10⁻¹¹")], ["4", bad("2.5×10⁻⁸"), ok("4.3×10⁻¹¹")],
  ];
  s.addTable(rows, { ...tableOpts(8.0, 2.05, 4.7, [1.0, 1.85, 1.85]), rowH: 0.46, objectName: "seed" });
  caption(s, "门槛 10⁻⁸：种子 3、4 原来过不了，延长后 5/5 通过；其他规则一概不变", 8.0, 5.0, 4.7);
  s.addNotes("“种子”指随机数编号，换一个种子就换一批 40 个随机起点。Burgers 起点命中率最低，40 个里只有 1～2 个。");
}
{
  const s = content("正问题完整流程：先过门槛，再让 L1 选项", "算法与迭代");
  img(s, "flowchart", "flowchart_zh.png", 0.9, 1.4, 11.5, "算法流程图");
  caption(s, "第一步：全库多起点 LM（不加 L1）；门槛：残差 ≤ 10⁻⁸ 才算库里拼得出解，否则拒绝；第二步：固定 (k, c, x₀) 走加权 L1 路径定支撑；第三步：只在支撑上重拟合。反问题直接从第二步进入。", 0.9, 6.25, 11.5);
  s.addNotes("门槛保证方法不会硬凑出一个错误答案：库不够用时它会明确说“没找到”。");
}
{
  const s = content("迭代四：加噪声后改用弱形式，不对噪声数据求导", "算法与迭代");
  formula(s, "∫ φ ∂ₓᵏ(uᵐ) dx dt  =  (−1)ᵏ ∫ (∂ₓᵏφ) uᵐ dx dt，   φ = (1 − s²)⁶", 0.6, 1.5, 12.1, 0.85, 22);
  const cards = [
    ["问题", "直接差分求导会放大噪声：1% 噪声时 uₓₓ 误差已达 45%，BIC 在无噪声时也多选项", HEX.warm],
    ["做法", "拿一个光滑“小鼓包” φ 乘方程两边再积分，分部积分把导数挪到 φ 上；φ 放在很多位置，每个位置是方程组一行", HEX.soft],
    ["为什么有效", "φ 的导数用公式算，没有误差；u 不用求导，只参与积分，正负噪声互相抵消", HEX.paleBlue],
  ];
  cards.forEach(([h, d, fill], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 2.65, 3.8, 2.6, fill, "weak-" + i);
    T(s, h, { x: x + 0.3, y: 2.85, w: 3.3, h: 0.45, fontSize: 18, bold: true, color: i === 0 ? C.accent2 : C.text2 });
    T(s, d, { x: x + 0.3, y: 3.4, w: 3.3, h: 1.8, fontSize: 14 });
  });
  card(s, 0.6, 5.5, 12.1, 1.2, HEX.soft, "weak-lib");
  T(s, [
    { text: "候选库随之改写成 ∂ₓᵏ(uᵐ)（13 列）：", options: { bold: true, color: C.text2 } },
    { text: "只有“某个东西的导数”才能分部积分，所以 uuₓ 写成 ½(u²)ₓ，KdV 的 6uuₓ 写成 3(u²)ₓ，意思完全一样。" },
  ], { x: 0.9, y: 5.6, w: 11.5, h: 1.0, fontSize: 15, valign: "middle" });
  s.addNotes("弱形式参考 Messenger & Bortz 2021 的 weak SINDy。测试函数在 x、t 两个方向都是 (1 − s²)⁶，积分用 FFT 相关一次算完所有位置。");
}
{
  const s = content("迭代五：统一判据，只比“谁能产生正确的候选组合”", "算法与迭代");
  formula(s, "L(S) = ‖A(w_S − w_LS)‖ / ‖A w_LS‖  +  |S| / p", 0.6, 1.5, 12.1, 0.85, 24);
  const cols = [
    ["第一项：拟合变差了多少", "只用组合 S 里的项，比用全部项拟合差了多少。少了真项，这一项会变大。"],
    ["第二项：用了多少项", "S 占库的比例（p = 13）。多了假项，这一项会变大。"],
    ["为什么统一", "L1、加权 L1、STRidge、遗传算法、穷举都用它挑最终组合，差别只在怎么产生候选，比较才公平。"],
  ];
  cols.forEach(([h, d], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 2.65, 3.8, 2.2, i === 2 ? HEX.paleBlue : HEX.soft, "crit-" + i);
    T(s, h, { x: x + 0.3, y: 2.85, w: 3.3, h: 0.45, fontSize: 17, bold: true, color: C.text2 });
    T(s, d, { x: x + 0.3, y: 3.4, w: 3.3, h: 1.4, fontSize: 14 });
  });
  T(s, "正问题有噪声时，门槛也要跟着改（差异原理，假定噪声大小 σ 已知）：", { x: 0.6, y: 5.15, w: 12.1, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  formula(s, "‖r‖ / ‖数据‖  ≤  3 × √2 · σ · std(u_ic) / ‖数据‖", 0.6, 5.65, 12.1, 0.75, 20);
  s.addNotes("判据来自 weak SINDy 的 MSTLS。倍数 3 在高噪声下偏松，后面结果页会讲。");
}

// ================= 实验结果 =================
{
  const s = content("无噪声结果：正、反问题各 5 个，全部选对", "实验结果");
  const rows = [
    head(["方程", "正问题：L1 选出的解析解", "反问题：L1 选出的方程"]),
    ["热方程", ok("e^(−π²t) sin πx"), ok("uₜ = uₓₓ")],
    ["波动方程", ok("cos πt sin πx"), ok("uₜₜ = uₓₓ")],
    ["粘性 Burgers", ok("0.5 − 0.5 tanh θ（通用库）"), ok("uₜ = −uuₓ + 0.1uₓₓ（加权 L1）")],
    ["Chafee–Infante", ok("tanh(x/√2)（通用库）"), "—"],
    ["KdV", ok("2 − 2tanh²θ（通用库）"), ok("uₜ = −6uuₓ − uₓₓₓ（加权 L1）")],
    ["平流扩散", "—", ok("uₜ = −0.4uₓ + 0.05uₓₓ")],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 1.5, 8.2, [2.0, 3.2, 3.0]), fontSize: 14, rowH: 0.55, objectName: "clean" });
  [["≤ 10⁻¹⁵", "通用库重拟合后波参数误差"], ["5 / 5", "每个通用库正例在 5 个随机种子下都选对"], ["10⁻⁸", "残差门槛；正例最好起点都 < 2×10⁻¹⁰"]].forEach(([v, l], i) => {
    const y = 1.5 + i * 1.7;
    card(s, 9.2, y, 3.5, 1.5, HEX.soft, "clean-stat-" + i);
    T(s, v, { x: 9.45, y: y + 0.12, w: 3.0, h: 0.75, fontSize: 32, bold: true, color: C.accent1, valign: "middle" });
    T(s, l, { x: 9.45, y: y + 0.9, w: 3.05, h: 0.5, fontSize: 13, color: C.text2 });
  });
  caption(s, "数据全部是精确解；反问题的热方程、波动方程用两个模态（单模态满足更短的等价方程）", 0.6, 5.6, 8.2);
  s.addNotes("这一页是噪声实验之前的结果，论文初稿里已有。");
}
{
  const s = content("库里没有答案时，方法会明确拒绝", "实验结果");
  img(s, "gate", "gate_zh.png", 0.6, 1.5, 8.1, "第一步 40 个起点残差与门槛");
  caption(s, "每个情形 40 个随机起点的残差（排序）；虚线是门槛 10⁻⁸", 0.6, 4.85, 8.1);
  const items = [
    ["正例", "最好的起点都 < 2×10⁻¹⁰", C.accent3],
    ["反例 1：库里删掉 tanh²", "最好残差 0.178，拒绝", C.accent2],
    ["反例 2：KdV 双孤子", "不是单一行波，残差 0.306，拒绝", C.accent2],
  ];
  items.forEach(([h, d, col], i) => {
    const y = 1.6 + i * 1.4;
    s.addShape(pres.shapes.OVAL, { x: 9.1, y: y + 0.08, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col } });
    T(s, h, { x: 9.6, y, w: 3.2, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
    T(s, d, { x: 9.6, y: y + 0.5, w: 3.2, h: 0.6, fontSize: 14 });
  });
  card(s, 9.1, 5.85, 3.6, 0.95, HEX.paleBlue, "gate-note");
  T(s, "正、反例残差相差 7 个数量级", { x: 9.3, y: 5.93, w: 3.25, h: 0.8, fontSize: 15, color: C.text2, valign: "middle" });
  s.addNotes("这两个反例是专门设计来证明方法不会硬凑出错误答案的。");
}
{
  const s = content("反问题加高斯噪声：加权 L1 到 5% 全部选对", "实验结果");
  img(s, "nInv", "noise_inverse_zh.png", 0.6, 1.35, 12.1, "反问题各方法选对率随噪声变化");
  caption(s, "噪声 = σ·std(u)·N(0,1)；5 个方程 × 9 个噪声水平 × 20 个种子；所有方法共用弱形式和同一判据", 0.6, 4.65, 12.1);
  const st = [["5 / 5", "个方程到 5% 噪声全部选对（加权 L1）"], ["0 / 20", "普通 L1 在 KdV 上任何噪声都多留 uₓ"], ["1～3%", "STRidge 在 Burgers 上开始出错的噪声"], ["10%", "热方程、Burgers、KdV 连穷举都错：数据信息不够"]];
  st.forEach(([v, l], i) => {
    const x = 0.6 + i * 3.06;
    card(s, x, 5.1, 2.85, 1.6, HEX.soft, "inv-stat-" + i);
    T(s, v, { x: x + 0.25, y: 5.2, w: 2.4, h: 0.7, fontSize: 28, bold: true, color: i === 0 ? C.accent1 : C.text2, valign: "middle" });
    T(s, l, { x: x + 0.25, y: 5.9, w: 2.45, h: 0.75, fontSize: 12, color: C.text2 });
  });
  s.addNotes("7% 时加权 L1 在 Burgers、KdV 上降到 0.75，遗传算法 0.90。");
}
{
  const s = content("误差分析：系数误差与噪声成正比；各方法能承受的噪声", "实验结果");
  img(s, "coef", "coef_err_zh.png", 0.6, 1.4, 5.6, "加权 L1 系数误差随噪声");
  caption(s, "加权 L1 系数相对误差（中位数）。2～3% 以内贴着斜率 1，之后增长更快；KdV 有约 3×10⁻⁴ 的积分误差下限", 0.6, 5.2, 5.6);
  T(s, "选对率 ≥ 0.95 时能承受的最大噪声", { x: 6.6, y: 1.4, w: 6.1, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
  img(s, "robust", "robust_zh.png", 6.6, 1.95, 6.1, "各方法能承受的最大噪声");
  caption(s, "加权 L1 与穷举（判据上界）基本一致；遗传算法在 KdV 上偶尔卡住；普通 L1 和 STRidge 明显更早失败", 6.6, 4.85, 6.1);
  card(s, 0.6, 5.95, 12.1, 0.8, HEX.paleBlue, "err-takeaway");
  T(s, "举例：Burgers 1% 噪声时系数误差 0.12%，5% 时 1.3%", { x: 0.9, y: 6.0, w: 11.5, h: 0.7, fontSize: 15, color: C.text2, valign: "middle" });
  s.addNotes("热图从 0 噪声开始往上数，选对率一直保持 ≥ 0.95 的最大噪声。“仅无噪声”指 0.1% 噪声就开始错。");
}
{
  const s = content("正问题加噪声：真解全部找对，门槛倍数需要收紧", "实验结果");
  img(s, "nFwd", "noise_forward_zh.png", 0.8, 1.3, 5.9, "正问题残差与噪声水平之比");
  const rows = [
    head(["情形", "1%", "5%", "10%"]),
    ["3 个真解", ok("5/5 对"), ok("5/5 对"), ok("5/5 对")],
    ["删 tanh²", ok("拒绝"), bad("放过"), bad("放过")],
    ["双孤子", ok("拒绝"), ok("拒绝"), bad("放过")],
  ];
  s.addTable(rows, { ...tableOpts(7.6, 1.5, 5.1, [1.8, 1.1, 1.1, 1.1]), rowH: 0.5, objectName: "fwd-noise" });
  T(s, "初边值数据加噪声 0.1%～10%，5 个种子。10% 噪声时 KdV 系数误差 0.5%；系数和波参数误差都与噪声成正比", { x: 7.6, y: 3.65, w: 5.1, h: 0.9, fontSize: 14 });
  card(s, 7.6, 4.7, 5.1, 1.65, HEX.paleBlue, "gate-decision");
  T(s, [
    { text: "待定：门槛倍数 3 还是 1.5\n", options: { bold: true, color: C.accent1 } },
    { text: "真解残差稳定在噪声的 0.88～1.10 倍；改成 1.5 倍后只剩“删 tanh²”在 10% 时放过 2/5，真解仍全部通过" },
  ], { x: 7.85, y: 4.8, w: 4.65, h: 1.45, fontSize: 14, color: C.text2 });
  caption(s, "论文原来的“过半”选项规则在噪声下会失效（Burgers 5% 时 0/5），改用统一判据后全部选对", 0.6, 6.45, 12.1);
  s.addNotes("1.5 倍是用已存残差事后算的，还没重跑。");
}

// ================= 对比 =================
{
  const s = content("对比一：自写遗传算法（同一判据，只比搜索方式）", "对比");
  T(s, "编码：每个个体是一串 0/1，每位对应库里一项，1 = 选", { x: 0.6, y: 1.45, w: 6.6, h: 0.5, fontSize: 15, bold: true, color: C.text2 });
  ["0", "1", "0", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0"].forEach((b, i) => chip(s, 0.6 + i * 0.5, 2.05, 0.44, 0.48, b, b === "1", 14));
  const steps = [["适应度", "判据 L(S)，越小越好"], ["选择", "3 个里挑最好的（锦标赛）"], ["交叉", "两个父代每位随机取一个"], ["变异", "每位以 1/13 概率翻转"], ["精英", "最好的 2 个直接保留"]];
  steps.forEach(([h, d], i) => {
    const y = 2.85 + i * 0.62;
    badge(s, 0.6, y, i + 1, C.accent4, 0.42);
    T(s, h, { x: 1.2, y: y + 0.04, w: 1.2, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
    T(s, d, { x: 2.4, y: y + 0.04, w: 4.7, h: 0.4, fontSize: 14 });
  });
  caption(s, "种群 40，跑 60 代", 0.6, 6.05, 6.6);
  const rows = [
    head(["", "加权 L1", "自写遗传算法"]),
    ["5% 噪声以内", ok("5 个方程全对"), mid("KdV 偶尔卡在 {uₓ}")],
    ["7% 噪声 Burgers / KdV", mid("0.75 / 0.75"), ok("0.90 / 0.90")],
    ["扩展库 PDE_divide 1%", bad("0/3"), ok("3/3")],
    ["每次评估判据次数", ok("约 15"), "约 450"],
    ["耗时（中位数）", ok("2.7 ms"), "92 ms"],
    ["结果是否随机", ok("固定"), "随种子变"],
  ];
  s.addTable(rows, { ...tableOpts(7.5, 1.5, 5.2, [2.1, 1.5, 1.6]), rowH: 0.52, objectName: "ga" });
  caption(s, "低噪声打平；高噪声或库项很相似时遗传算法更稳，代价是约 30 倍计算量", 7.5, 5.35, 5.2);
  s.addNotes("遗传算法卡在 {uₓ} 是搜索失败：同一判据下穷举是对的。PDE_divide 的扩展库里含 x 的几列彼此很像，正是 L1 不可表示条件难满足的情形。");
}
{
  const s = content("对比二：SGA-PDE 和我们的方法在做法上的差别", "对比");
  const rows = [
    head(["", "SGA-PDE（Chen 等，PRR 2022）", "本文（加权 L1）"]),
    ["候选项从哪来", "遗传算法进化符号树，如 u·(u d x)", "预先列好的固定候选库（13 列）"],
    ["导数怎么算", "有限差分直接对数据求导", "弱形式：积分代替求导"],
    ["怎么选项", "交叉 + 变异 + 替换，100 代", "加权 L1 的 λ 路径（LARS 精确解）"],
    ["每个候选怎么评分", "STRidge 拟合系数，AIC（误差 + 项数罚）", "MSTLS 判据 L(S)"],
    ["能否找到库外项", ok("能，结构自由"), bad("不能，真项必须在库里")],
    ["计算量 / 随机性", "约 30 分钟 / 结果随种子变", "约 3 毫秒 / 结果固定"],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 1.5, 12.1, [2.4, 5.0, 4.7]), fontSize: 15, rowH: 0.62, objectName: "sga-vs" });
  card(s, 0.6, 6.0, 12.1, 0.75, HEX.soft, "sga-setup");
  T(s, "对比设置：同一份带噪声数据同时给两边；SGA-PDE 用论文默认参数（种群 20、深度 4、宽度 5、100 代），每次限时 30 分钟，代码未改", { x: 0.85, y: 6.05, w: 11.6, h: 0.65, fontSize: 14, color: C.text2, valign: "middle" });
  s.addNotes("SGA-PDE 选出的项是符号式子，我们把每一项与标准项比余弦相似度，≥ 0.9 归为那一项，归类后与真项一致才算对。");
}
{
  const s = content("对比二结果：SGA-PDE 论文的 5 个方程，有噪声时它全错", "对比");
  img(s, "nSga", "noise_sga_zh.png", 0.6, 1.35, 12.1, "与 SGA-PDE 的选对率对比");
  caption(s, "C–I = Chafee–Infante；compound：uₜ = (uuₓ)ₓ；divide*：uₜ = −uₓ/x + 0.25uₓₓ，需在库里加 4 个含 x 的项", 0.6, 4.5, 12.1);
  const rows = [
    head(["噪声", "SGA-PDE", "加权 L1", "说明"]),
    ["0%", "3/7", "7/7", "SGA 两组 KdV、compound、divide 未找到"],
    ["1%", bad("0/21"), ok("18/21"), "我们错的 3 次都是 divide*（遗传算法和穷举对）"],
    ["5%", bad("0/21"), mid("6/21"), "我们失败的是小网格数据，穷举也错"],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 4.95, 8.2, [0.9, 1.3, 1.3, 4.7]), rowH: 0.42, fontSize: 13, objectName: "sga-res" });
  card(s, 9.1, 4.95, 3.6, 1.75, HEX.warm, "fair");
  T(s, "公平性", { x: 9.3, y: 5.05, w: 3.2, h: 0.35, fontSize: 15, bold: true, color: C.accent2 });
  T(s, "SGA 论文报告这些方程都能找到，我们未复现：关了它的神经网络去噪；30 分钟内多数只跑到 24～94 代", { x: 9.3, y: 5.45, w: 3.25, h: 1.2, fontSize: 12, color: C.text2 });
  s.addNotes("结论应表述为“在不去噪、30 分钟预算下”的比较，不能写成 SGA-PDE 做不到。");
}
{
  const s = content("计算代价：加权 L1 比遗传类方法快 1～6 个数量级", "对比");
  img(s, "cost", "cost_zh.png", 0.6, 1.4, 7.6, "各方法单次求解耗时");
  caption(s, "前五个方法：noise_inverse 全部实验的中位数；SGA-PDE：其数据集上 49 次运行的中位数，多数触到 30 分钟限时", 0.6, 5.1, 7.6);
  card(s, 8.6, 1.5, 4.1, 5.2, HEX.paleBlue, "cost-why");
  T(s, "为什么差这么多", { x: 8.9, y: 1.7, w: 3.6, h: 0.45, fontSize: 17, bold: true, color: C.accent1 });
  T(s, bullets([
    "加权 L1：LARS 一次给出整条路径，只评估约 15 个组合",
    "遗传算法：每代 40 个个体 × 60 代，约 450 次评估",
    "穷举：13 项里取 ≤ 6 项共 4096 种；库变大后组合数爆炸",
    "SGA-PDE：每个个体是一棵符号树，要重新求导、拟合，100 代",
  ]), { x: 8.9, y: 2.3, w: 3.6, h: 4.2, fontSize: 14, paraSpaceAfter: 10, color: C.text2 });
  s.addNotes("库只有 13 项时穷举也只要 0.18 秒；库扩到几十、上百项后穷举不可行，L1 的优势会更明显，这一点还没用大库验证。");
}
{
  const s = content("方法优劣总览", "对比");
  const rows = [
    head(["方法", "需预设候选库", "无噪声", "抗噪声", "计算量", "结果确定", "能发现库外项"]),
    ["普通 L1", "需要", bad("多留 uₓ"), bad("差"), ok("极小"), ok("是"), bad("否")],
    [{ text: "加权 L1（本文）", options: { bold: true, color: C.accent1 } }, "需要", ok("全对"), ok("到 5%"), ok("极小（~15 次）"), ok("是"), bad("否")],
    ["STRidge", "需要", ok("全对"), mid("1～3% 起掉"), ok("极小"), ok("是"), bad("否")],
    ["自写遗传算法", "需要", mid("KdV 偶尔卡住"), ok("到 5%，7% 略好"), mid("中（~450 次）"), mid("否"), bad("否")],
    ["穷举（判据上界）", "需要", ok("全对"), ok("到 5～7%"), bad("组合数爆炸"), ok("是"), bad("否")],
    ["SGA-PDE", ok("不需要"), mid("3/7"), bad("1% 起全错*"), bad("约 30 分钟"), mid("否"), ok("能")],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 1.5, 12.1, [2.3, 1.6, 1.6, 1.9, 1.8, 1.3, 1.6]), fontSize: 14, rowH: 0.56, objectName: "matrix" });
  caption(s, "* 未开 SGA-PDE 的神经网络去噪、30 分钟限时；绿色 = 优势，橙色 = 一般，红色 = 劣势", 0.6, 5.55, 12.1);
  card(s, 0.6, 6.0, 12.1, 0.8, HEX.paleBlue, "matrix-take");
  T(s, [
    { text: "建议：", options: { bold: true, color: C.accent1 } },
    { text: "加权 L1 定支撑为主（快、确定、抗噪到 5%）；遗传算法作对照和兜底；库外项是固定库方法的共同局限。" },
  ], { x: 0.9, y: 6.05, w: 11.5, h: 0.7, fontSize: 15, color: C.text2, valign: "middle" });
  s.addNotes("这张表是全片的结论页。");
}

// ================= 总结 =================
{
  section("总结");
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sectionName });
  s.addText("结论、局限与下一步", { isTextBox: true, x: 0.8, y: 0.6, w: 11.7, h: 0.9, fontSize: 38, bold: true, color: C.background1, margin: 0 });
  const cols = [
    ["结论", "9CC3F2", ["L1 稀疏选择能同时求解析解和控制方程", "加权 L1 修掉了普通 L1 多留 uₓ 的问题", "弱形式 + 统一判据后，反问题抗噪到 5%", "比遗传类方法快 1～6 个数量级"]],
    ["局限", "F6B08F", ["真项必须在候选库里（uₓ/x 需另加）", "正问题只覆盖单一行波", "噪声门槛假定噪声大小已知", "高噪声或库项相似时不如遗传算法"]],
    ["下一步", "9CC3F2", ["定正问题噪声门槛倍数（3 或 1.5）", "论文加入噪声与遗传算法比较两节", "设计多波候选库", "确定作者、目标会议和页数"]],
  ];
  cols.forEach(([h, col, items], i) => {
    const x = 0.8 + i * 4.0;
    card(s, x, 1.8, 3.75, 5.0, "26344F", "end-col-" + i);
    T(s, h, { x: x + 0.3, y: 2.05, w: 3.2, h: 0.6, fontSize: 24, bold: true, color: col });
    T(s, bullets(items), { x: x + 0.3, y: 2.85, w: 3.2, h: 3.8, fontSize: 16, color: "E6ECF5", paraSpaceAfter: 14 });
  });
  s.addNotes("代码和结果在 GitHub 的 exp/generic-library 分支；噪声和遗传算法部分还没提交。");
}

(async () => {
  const out = path.join(__dirname, "PDE_L1_汇报_v2.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();
