// 进展汇报 PPT。数字全部取自论文和结果 JSON。运行：cd slides && node build_deck.js
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
const HEX = { blue: "2A78D6", paleBlue: "DCEAFB", ink: "141B2D", ink2: "52514E", grid: "D9DDE3", warn: "EB6834", ok: "1BAF7A" };
const MATH = "Cambria Math";
const IMG = (f) => path.join(__dirname, "img", f);

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
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.4, fontSize: 18, color: "C9D6EA", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 30, bold: true, color: C.text2, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8A8984", align: "right" },
});

let sectionName = "";
function content(title, section) {
  if (section && section !== sectionName) { pres.addSection({ title: section }); sectionName = section; }
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sectionName });
  s.addText(title, { placeholder: "title" });
  return s;
}
const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top", ...o });
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
}
function badge(s, x, y, n, color) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color }, line: { color } });
  s.addText(String(n), { isTextBox: true, x, y, w: 0.5, h: 0.5, margin: 0, align: "center", valign: "middle", fontSize: 16, bold: true, color: C.background1 });
}
const tableOpts = (x, y, w, colW) => ({ x, y, w, colW, fontSize: 13, color: C.text1, border: { type: "solid", pt: 0.75, color: HEX.grid }, valign: "middle", margin: [0.04, 0.08, 0.04, 0.08] });
const head = (cells) => cells.map((t) => ({ text: t, options: { bold: true, fill: { color: HEX.paleBlue }, color: C.text2 } }));

// ---------- 1 标题 ----------
pres.addSection({ title: "开场" }); sectionName = "开场";
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "开场" });
  s.addText("用 L1 稀疏库求 PDE 的解析解", { placeholder: "title" });
  s.addText("先设出很多候选项和系数，让 L1 把大部分系数压成 0\n正问题与反问题 · 进展汇报 · 2026-10-29", { placeholder: "body" });
  s.addNotes("这次汇报老师提出的思路：先把候选项和系数都设出来，用 L1 让大部分系数为 0，剩下的就是解析解。我们在正问题和反问题两个方向各做了 5 个方程，全部选对。");
}

// ---------- 2 出发点 ----------
{
  const s = content("出发点：把“猜解”换成“选项”", "背景");
  card(s, 0.6, 1.5, 5.6, 4.9, HEX.paleBlue, "advisor-idea");
  T(s, "导师的想法", { x: 0.95, y: 1.8, w: 5.0, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  T(s, "面对一个新问题，只知道 PDE，想知道它的解析解。解由很多项和系数构成，但不知道是哪些项、系数是几。", { x: 0.95, y: 2.45, w: 4.9, h: 1.6, fontSize: 16 });
  T(s, "能不能先把这些项和系数全部设出来，用 L1 norm 让大部分系数为 0？", { x: 0.95, y: 4.15, w: 4.9, h: 1.2, fontSize: 16, italic: true, color: C.accent1 });
  const steps = [
    ["设候选项", "写出很多候选函数，系数全部未知"],
    ["L1 压零", "拟合 PDE 和初边值，L1 让多数系数精确为 0"],
    ["读出解析解", "剩下的几项和系数就是闭式解；代回残差到机器精度才算数"],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.65 + i * 1.6;
    badge(s, 6.9, y, i + 1, C.accent1);
    T(s, h, { x: 7.65, y: y - 0.02, w: 5.0, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
    T(s, d, { x: 7.65, y: y + 0.5, w: 5.0, h: 0.8, fontSize: 15, color: C.text1 });
  });
  s.addNotes("文献里的 SINDy、PDE-FIND 做的是从数据反推方程；已知方程、用稀疏展开找闭式解的方向做得少。我们两个方向用同一套工具：候选库、L1、定支撑规则、重拟合。");
}

// ---------- 3 两个方向 ----------
{
  const s = content("同一套工具，两个方向", "背景");
  s.addImage({ path: IMG("overview.png"), x: 1.37, y: 1.3, w: 10.6, h: 10.6 * 906 / 1861, altText: "正问题与反问题示意图" });
  T(s, "蓝色：L1 保留的项（系数为重拟合值）；灰色：L1 压成 0 的项。示例都是 KdV。", { x: 1.37, y: 6.6, w: 10.6, h: 0.4, fontSize: 12, color: "6B6A66" });
  s.addNotes("上面是正问题：已知 KdV 方程和初边值，在 8 个通用候选项里 L1 只留下 1 和 tanh²，系数 2 和 -2，拼起来就是孤子 2sech²(x-4t)。下面是反问题：已知双孤子的采样，L1 只留下 uu_x 和 u_xxx。");
}

// ---------- 4 方法核心 ----------
{
  const s = content("方法核心：只让 L1 决定留哪几项", "方法");
  const cards = [
    ["目标函数", "½‖r(ξ)‖² + λ Σ wⱼ |ξⱼ|", "r = PDE 配点残差 + 边界残差 + 初值残差，三块等权；不给解的内部采样"],
    ["定支撑", "40 个 λ，|ξⱼ| > 10⁻⁵ 过半", "λ 从 λ_max 几何降到 10⁻⁶ λ_max；阈值不放宽"],
    ["重拟合", "只在支撑上最小二乘", "只读系数，不增删任何一项"],
  ];
  cards.forEach(([h, f, d], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.55, 3.8, 3.6, "F3F6FA", "method-card-" + i);
    badge(s, x + 0.3, 1.85, i + 1, C.accent1);
    T(s, h, { x: x + 0.95, y: 1.88, w: 2.7, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
    T(s, f, { x: x + 0.3, y: 2.65, w: 3.3, h: 0.8, fontSize: 18, fontFace: MATH, color: C.accent1 });
    T(s, d, { x: x + 0.3, y: 3.5, w: 3.3, h: 1.5, fontSize: 14 });
  });
  card(s, 0.6, 5.5, 12.1, 1.15, HEX.paleBlue, "solver-note");
  T(s, "每个 L1 子问题都用 LARS 同伦路径精确求解（KKT 误差 < 5×10⁻⁹）。非线性方程用 Gauss–Newton 线性化，每步解一个精确 L1。", { x: 0.9, y: 5.72, w: 11.5, h: 0.8, fontSize: 15, color: C.text2 });
  s.addNotes("关键约束：哪几项留下只由 L1 决定；重拟合只读系数。用精确求解器是为了排除“求解器没收敛”这种解释。");
}

// ---------- 5 难点一 ----------
{
  const s = content("难点一：反问题里普通 L1 总多留一个 uₓ", "方法");
  s.addImage({ path: IMG("inverse_path.png"), x: 0.6, y: 1.5, w: 7.4, h: 7.4 * 743 / 1539, altText: "普通 L1 沿 λ 路径的 u_x 系数" });
  T(s, "真项进来后，uₓ 系数随 λ 成比例下降，但始终不为 0", { x: 0.6, y: 5.2, w: 7.4, h: 0.5, fontSize: 14, color: "6B6A66" });
  T(s, "不可表示条件：uₓ 列的值", { x: 8.5, y: 1.5, w: 4.2, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  [["2.77", "KdV 双孤子"], ["1.39", "Burgers，Cole–Hopf"], ["1.52", "Burgers，两行波"]].forEach(([v, l], i) => {
    const y = 2.1 + i * 1.15;
    T(s, v, { x: 8.5, y, w: 1.9, h: 0.9, fontSize: 44, bold: true, color: C.accent2, valign: "middle" });
    T(s, l, { x: 10.45, y: y + 0.2, w: 2.3, h: 0.5, fontSize: 15, valign: "middle" });
  });
  card(s, 8.5, 5.6, 4.2, 1.1, "FDEEE7", "irrep-rule");
  T(s, "需要 < 1 才不会被选；> 1 时任何足够小的 λ 都会选上，与求解器无关", { x: 8.7, y: 5.72, w: 3.85, h: 0.9, fontSize: 14, color: C.text2 });
  s.addNotes("精确求解器复现了旧结果，KKT 误差 1e-10 量级，排除了没收敛。其他错列的不可表示值都小于 1，只有 u_x 大于 1，所以普通 L1 只会多留 u_x。这也解释了为什么旧的 1e-8 残差门槛整条路径都到不了：残差和 λ 成正比。");
}

// ---------- 6 修法 ----------
{
  const s = content("修法：迭代重加权 L1，两轮去掉 uₓ", "方法");
  const flow = [
    ["第 0 轮：普通 L1", "KdV 选 {uuₓ, uₓₓₓ, uₓ, uₓₓ}\nBurgers 选 {uuₓ, uₓₓ, uₓ}"],
    ["改权重", "wⱼ = 1/(|βⱼ| + ε)，β 取上一轮 L1 解\nuₓ 系数小 → 权重大"],
    ["第 1 轮起只剩真项", "KdV：uuₓ (−6)，uₓₓₓ (−1)\nBurgers：uuₓ (−1)，uₓₓ (0.1)"],
  ];
  flow.forEach(([h, d], i) => {
    const y = 1.5 + i * 1.75;
    card(s, 0.6, y, 6.2, 1.45, i === 2 ? HEX.paleBlue : "F3F6FA", "irl1-step-" + i);
    badge(s, 0.85, y + 0.25, i + 1, i === 2 ? C.accent1 : C.accent6);
    T(s, h, { x: 1.55, y: y + 0.22, w: 5.1, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
    T(s, d, { x: 1.55, y: y + 0.7, w: 5.1, h: 0.7, fontSize: 14 });
  });
  T(s, "第 0 轮 λ 比例的敏感性", { x: 7.3, y: 1.5, w: 5.4, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
  T(s, "每格汇总 4 个 ε，结果与 ε 无关", { x: 7.3, y: 1.92, w: 5.4, h: 0.35, fontSize: 13, color: "6B6A66" });
  const P = (t) => ({ text: t, options: { color: t === "通过" ? "13865E" : "C2410C", bold: true, align: "center" } });
  const rows = [
    head(["λ/λ_max", "KdV", "Burgers CH", "Burgers 两行波"]),
    ["10⁻¹", "失败", "失败", "失败"], ["10⁻²", "失败", "通过", "通过"], ["10⁻³", "通过", "通过", "通过"],
    ["10⁻⁴", "通过", "通过", "通过"], ["10⁻⁵", "通过", "通过", "通过"],
  ].map((r, i) => (i === 0 ? r : [r[0], P(r[1]), P(r[2]), P(r[3])]));
  rows.push(["合计", "12/20", "16/20", "16/20"]);
  s.addTable(rows, { ...tableOpts(7.3, 2.4, 5.4, [1.2, 1.2, 1.4, 1.6]), rowH: 0.42, fontSize: 14, objectName: "sensitivity" });
  T(s, "论文用固定 10⁻³ 并附这张表；比例太大时第 0 轮漏掉真项，重加权会锁死错的支撑", { x: 7.3, y: 5.85, w: 5.4, h: 0.85, fontSize: 13, color: "6B6A66" });
  s.addNotes("权重只来自上一轮 L1 解，不用岭回归：岭回归会把能量分给相关的 u_x，反而让 u_x 权重变小。阈值 1e-5 没有放宽。");
}

// ---------- 7 难点二 ----------
{
  const s = content("难点二：旧的正问题库里本来就放着答案", "方法");
  card(s, 0.6, 1.5, 5.85, 5.0, "F3F6FA", "old-library");
  T(s, "之前：针对真解设计的库", { x: 0.9, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  T(s, [
    { text: "KdV 库里直接有 2sech²(x − 4t) 这一列", options: { bullet: true, breakLine: true } },
    { text: "波速预先离散成几列：c = 1, 2, 4, 9", options: { bullet: true, breakLine: true } },
    { text: "L1 只是在准备好的几列里挑一列", options: { bullet: true, breakLine: true } },
    { text: "老师会问：你已经知道答案长什么样了", options: { bullet: true } },
  ], { x: 0.9, y: 2.45, w: 5.3, h: 3.6, fontSize: 16, paraSpaceAfter: 10 });
  card(s, 6.85, 1.5, 5.85, 5.0, HEX.paleBlue, "generic-library");
  T(s, "现在：三个方程共用的通用库", { x: 7.15, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  const chips = ["1", "tanh θ", "tanh² θ", "tanh³ θ", "tanh⁴ θ", "sech θ", "e^(−θ²)", "θ"];
  chips.forEach((c, i) => {
    const x = 7.15 + (i % 4) * 1.32, y = 2.5 + Math.floor(i / 4) * 0.7;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.2, h: 0.5, rectRadius: 0.06, fill: { color: i < 5 ? "FFFFFF" : "F0EFEC" }, line: { color: i < 5 ? HEX.blue : "8A8984", width: 1 } });
    s.addText(c, { isTextBox: true, x, y, w: 1.2, h: 0.5, margin: 0, align: "center", valign: "middle", fontSize: 14, fontFace: MATH, color: C.text1 });
  });
  T(s, "θ = k(x − x₀ − ct)，(k, c, x₀) 连续搜索", { x: 7.15, y: 4.05, w: 5.3, h: 0.5, fontSize: 16, fontFace: MATH, color: C.accent1 });
  T(s, [
    { text: "前五项：求行波解常用的 tanh 展开法", options: { bullet: true, breakLine: true } },
    { text: "后三项：干扰项", options: { bullet: true, breakLine: true } },
    { text: "没有任何一列等于真解，真解要由几列拼出来", options: { bullet: true } },
  ], { x: 7.15, y: 4.7, w: 5.3, h: 1.6, fontSize: 15, paraSpaceAfter: 8 });
  s.addNotes("这是为了回答“库里本来就放了答案”的质疑。搜索范围是通用量级：k 在 0.1 到 10，c 在 -8 到 8，x0 在整个计算域。");
}

// ---------- 8 流程图 ----------
{
  const s = content("算法流程：三步，L1 只管选项", "方法");
  s.addImage({ path: IMG("flowchart.png"), x: 0.75, y: 1.45, w: 11.8, h: 11.8 * 792 / 1937, altText: "算法流程图" });
  T(s, "第一步找波参数（无正则、多起点）；第二步只用 L1 定支撑；第三步在支撑上重拟合系数和波参数。反问题直接从第二步进入。", { x: 0.75, y: 6.45, w: 11.8, h: 0.6, fontSize: 14, color: "6B6A66" });
  s.addNotes("门槛 1e-8：第一步残差到不了，就判定库里没有精确解，不报支撑。Burgers 原来 300 步迭代不够，所以对最好的起点再延长迭代最多 3000 步，其他规则不变。");
}

// ---------- 9 通用库结果 ----------
{
  const s = content("结果：同一个库拼出三个方程的解析解", "结果");
  const rows = [
    head(["方程", "L1 选出的项", "拼出的解析解", "(k, c, x₀)", "起点命中", "种子"]),
    ["KdV", "1, tanh²θ", "2 − 2tanh²θ = 2sech²(x − 4t)", "(1, 4, 0)", "13/40", "5/5"],
    ["Burgers", "1, tanh θ", "0.5 − 0.5 tanh θ", "(2.5, 0.5, 0.3)", "1/40", "5/5"],
    ["Chafee–Infante", "tanh θ", "tanh(x/√2)", "(1/√2, 0, 0)", "10/40", "5/5"],
  ];
  s.addTable(rows, { ...tableOpts(0.6, 1.55, 12.1, [2.0, 1.9, 3.7, 1.9, 1.3, 1.3]), fontSize: 17, rowH: 0.6, objectName: "generic-results" });
  [["≤ 10⁻¹⁵", "波参数误差"], ["3 / 3", "λ 比例 10⁻²～10⁻⁴ 都通过"], ["≈ 36 分钟", "整体运行时间（本机）"]].forEach(([v, l], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.35, 3.8, 2.1, "F3F6FA", "stat-" + i);
    T(s, v, { x: x + 0.3, y: 4.6, w: 3.3, h: 1.0, fontSize: 40, bold: true, color: C.accent1, valign: "middle" });
    T(s, l, { x: x + 0.3, y: 5.7, w: 3.3, h: 0.55, fontSize: 15, color: C.text2 });
  });
  s.addNotes("三个正例在 5 个随机种子下都选对；重拟合后波速、宽度、位置都恢复到舍入误差。Burgers 的起点命中率最低，40 个里只有 1 个。");
}

// ---------- 10 门槛 ----------
{
  const s = content("库里没有答案时，方法会说“没找到”", "结果");
  s.addImage({ path: IMG("gate.png"), x: 0.6, y: 1.5, w: 8.0, h: 8.0 * 772 / 1967, altText: "第一步 40 个起点残差与门槛" });
  T(s, "每个情形 40 个随机起点的残差（排序）；虚线是门槛 10⁻⁸", { x: 0.6, y: 4.75, w: 8.0, h: 0.5, fontSize: 13, color: "6B6A66" });
  const items = [
    ["正例", "最好的起点都 < 2×10⁻¹⁰", C.accent3],
    ["反例 1：删掉 tanh²", "最好残差 0.178，拦下", C.accent2],
    ["反例 2：KdV 双孤子", "最好残差 0.306，拦下", C.accent2],
  ];
  items.forEach(([h, d, col], i) => {
    const y = 1.6 + i * 1.5;
    s.addShape(pres.shapes.OVAL, { x: 9.0, y: y + 0.08, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col } });
    T(s, h, { x: 9.5, y, w: 3.3, h: 0.45, fontSize: 17, bold: true, color: C.text2 });
    T(s, d, { x: 9.5, y: y + 0.5, w: 3.3, h: 0.5, fontSize: 15 });
  });
  card(s, 9.0, 6.0, 3.7, 0.85, HEX.paleBlue, "gate-note");
  T(s, "残差相差 7 个数量级\n不会硬凑出错的解", { x: 9.2, y: 6.1, w: 3.35, h: 0.65, fontSize: 14, color: C.text2, valign: "middle" });
  s.addNotes("这两个反例是专门设计的：删掉必需的 tanh² 列，或者用单一行波库去拟合双孤子。门槛把“找到了”和“只是拟合得不错”分开。");
}

// ---------- 11 x0 ----------
{
  const s = content("一个发现：位置参数要靠稀疏性才能定准", "结果");
  T(s, "tanh 库能近似“吸收”一个小平移", { x: 0.6, y: 1.5, w: 6.0, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  T(s, "tanh(θ + ε) ≈ tanh θ + ε(1 − tanh²θ)", { x: 0.6, y: 2.15, w: 6.0, h: 0.55, fontSize: 20, fontFace: MATH, color: C.accent1 });
  T(s, [
    { text: "x₀ 偏一点，常数项和 tanh² 项各补一点就能抵掉大部分误差", options: { bullet: true, breakLine: true } },
    { text: "种子 3：x₀ = 0.2895（真值 0.3），常数项 0.513、tanh² 项 −0.013，和公式吻合", options: { bullet: true, breakLine: true } },
    { text: "残差沿 x₀ 几乎是平的，优化 300 步走不到门槛", options: { bullet: true, breakLine: true } },
    { text: "L1 选出 {1, tanh θ} 后在支撑上重拟合，x₀ 精确回到 0.3（种子 0）", options: { bullet: true } },
  ], { x: 0.6, y: 2.9, w: 6.0, h: 3.8, fontSize: 15, paraSpaceAfter: 10 });
  T(s, "Burgers：延长迭代前后的最好残差", { x: 7.1, y: 1.5, w: 5.6, h: 0.5, fontSize: 17, bold: true, color: C.text2 });
  const rows = [
    head(["种子", "原来（300 步）", "延长后", "支撑"]),
    ["0", "1.4×10⁻¹⁰", "1.3×10⁻¹⁰", "{1, tanh}"], ["1", "3.4×10⁻⁹", "3.7×10⁻¹¹", "{1, tanh}"],
    ["2", "1.3×10⁻¹⁰", "1.3×10⁻¹⁰", "{1, tanh}"],
    ["3", { text: "1.9×10⁻⁸", options: { color: "C2410C", bold: true } }, "4.2×10⁻¹¹", "{1, tanh}"],
    ["4", { text: "2.5×10⁻⁸", options: { color: "C2410C", bold: true } }, "4.3×10⁻¹¹", "{1, tanh}"],
  ];
  s.addTable(rows, { ...tableOpts(7.1, 2.15, 5.6, [0.8, 1.7, 1.5, 1.6]), objectName: "seed-table" });
  T(s, "种子 3、4 原来卡在门槛上方；只延长迭代、不改其他规则后 5/5 通过", { x: 7.1, y: 5.55, w: 5.6, h: 0.8, fontSize: 13, color: "6B6A66" });
  s.addNotes("这不是落进错误的局部最优：k 和 c 都对，只有 x0 偏了约 0.01。只用最小二乘时 x0 定不准，稀疏的支撑把它唯一定了下来。重拟合目前只在种子 0 上跑过。");
}

// ---------- 12 汇总 ----------
{
  const s = content("汇总：正、反问题各 5 个，全部选对", "结果");
  const groups = [
    ["正问题（已知 PDE，求解析解）", ["热方程", "波动方程", "粘性 Burgers", "Chafee–Infante", "KdV"], "另有通用库：KdV、Burgers、Chafee–Infante 三个方程用同一个库"],
    ["反问题（已知解的采样，求方程）", ["热方程（双模态）", "波动方程（双模态）", "平流扩散", "粘性 Burgers", "KdV 双孤子"], "Burgers 和 KdV 双孤子用重加权 L1"],
  ];
  groups.forEach(([h, items, note], g) => {
    const x = 0.6 + g * 6.25;
    card(s, x, 1.5, 5.85, 5.2, g === 0 ? HEX.paleBlue : "F3F6FA", "summary-" + g);
    T(s, h, { x: x + 0.3, y: 1.75, w: 5.3, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
    items.forEach((it, i) => {
      const y = 2.5 + i * 0.62;
      s.addShape(pres.shapes.OVAL, { x: x + 0.35, y: y + 0.07, w: 0.34, h: 0.34, fill: { color: "13865E" }, line: { color: "13865E" } });
      s.addText("✓", { isTextBox: true, x: x + 0.35, y: y + 0.07, w: 0.34, h: 0.34, margin: 0, align: "center", valign: "middle", fontSize: 12, bold: true, color: C.background1 });
      T(s, it, { x: x + 0.9, y, w: 4.6, h: 0.48, fontSize: 16, valign: "middle" });
    });
    T(s, note, { x: x + 0.3, y: 5.75, w: 5.3, h: 0.8, fontSize: 13, color: "6B6A66" });
  });
  s.addNotes("选出的项都等于真项，重拟合系数和真值一致到 1e-15。数据全部是精确解，没有噪声。");
}

// ---------- 13 局限与下一步 ----------
{
  pres.addSection({ title: "总结" }); sectionName = "总结";
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "总结" });
  s.addText("局限与下一步", { isTextBox: true, x: 0.8, y: 0.9, w: 11.7, h: 0.9, fontSize: 40, bold: true, color: C.background1, margin: 0 });
  const cols = [
    ["目前的局限", ["数据都没有噪声", "通用库只覆盖单一行波，双孤子需要另设库", "λ 比例 10⁻³ 是探索中选的，10⁻² 以上可能失败", "Burgers 多起点命中率低（1～2/40），运行约 36 分钟"]],
    ["下一步", ["加噪声测试", "设计多波（ln F 或两个波变量）的候选库", "确定作者、目标会议和页数", "导师审阅论文后修改"]],
  ];
  cols.forEach(([h, items], i) => {
    const x = 0.8 + i * 6.1;
    T(s, h, { x, y: 2.3, w: 5.6, h: 0.6, fontSize: 26, bold: true, color: i === 0 ? "F6B08F" : "9CC3F2" });
    T(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
      { x, y: 3.15, w: 5.6, h: 3.8, fontSize: 20, color: "E6ECF5", paraSpaceAfter: 18 });
  });
  s.addNotes("论文初稿已写好，英文、NeurIPS 2026 模板、10 页，代码和结果都在 GitHub 的 exp/generic-library 分支。");
}

(async () => {
  const out = path.join(__dirname, "PDE_L1_进展汇报.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();
