const dataUrl = "./data/dashboard.json";

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "NA";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  return number.toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") return "NA";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  return `${(number * 100).toFixed(1)}%`;
}

function formatDateTime(value) {
  return value || "NA";
}

function actionClass(action) {
  if (action === "买入") return "buy";
  if (action === "卖出" || action === "空仓") return "sell flat";
  if (action === "持有") return "hold";
  return "";
}

function stateLabel(state) {
  if (Number(state) === 1) return "100%";
  if (Number(state) === 0) return "0%";
  return "NA";
}

function renderEmpty(target, text) {
  const div = document.createElement("div");
  div.className = "empty-state";
  div.textContent = text;
  target.replaceChildren(div);
}

function makeSignalCard(record) {
  const template = document.getElementById("signalCardTemplate");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".market-name").textContent = record.market ?? "未知市场";
  node.querySelector(".symbol-name").textContent = record.symbol ?? "NA";
  const pill = node.querySelector(".action-pill");
  pill.textContent = record.action ?? "NA";
  pill.className = `action-pill ${actionClass(record.action)}`.trim();
  node.querySelector(".signal-price").textContent = formatNumber(record.signal_price, 4);
  node.querySelector(".nq-price").textContent = formatNumber(record.nqmain_rt_value, 2);
  node.querySelector(".feature-time").textContent = formatDateTime(record.feature_bar_time);
  node.querySelector(".market-phase").textContent = record.market_phase ?? "NA";
  node.querySelector(".current-state").textContent = stateLabel(record.current_state);
  node.querySelector(".target-state").textContent = stateLabel(record.recommended_state);
  node.querySelector(".probability").textContent = formatPercent(record.probability_long);
  node.querySelector(".last-action").textContent = record.last_action_time ?? record.position_since ?? "NA";
  node.querySelector(".summary").textContent = record.signal_summary ?? "无摘要";
  node.querySelector(".drivers").textContent = record.top_drivers ?? "无驱动因子";
  return node;
}

function makeEventItem(record) {
  const template = document.getElementById("eventItemTemplate");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".event-market").textContent = `${record.market ?? "未知"} / ${record.symbol ?? "NA"}`;
  node.querySelector(".event-action").textContent = record.action ?? "NA";
  node.querySelector(".event-summary").textContent = record.signal_summary ?? "无摘要";
  node.querySelector(".event-meta").textContent =
    `run_at: ${record.run_at ?? "NA"} | feature_bar: ${record.feature_bar_time ?? "NA"} | 价格: ${formatNumber(record.signal_price, 4)}`;
  return node;
}

function renderCards(targetId, records) {
  const target = document.getElementById(targetId);
  if (!records.length) {
    renderEmpty(target, "当前没有可展示的数据。");
    return;
  }
  target.replaceChildren(...records.map(makeSignalCard));
}

function renderEvents(targetId, records, emptyText) {
  const target = document.getElementById(targetId);
  if (!records.length) {
    renderEmpty(target, emptyText);
    return;
  }
  target.replaceChildren(...records.map(makeEventItem));
}

async function main() {
  const response = await fetch(`${dataUrl}?t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch dashboard data: ${response.status}`);
  }
  const data = await response.json();

  document.getElementById("generatedAt").textContent = data.generated_at ?? "NA";
  const sources = data.sources ?? {};
  const available = [];
  if (sources.five_minute_live_exists) available.push("5m");
  if (sources.btc_live_exists) available.push("btc-5m");
  if (sources.open_nowcast_live_exists) available.push("open-nowcast");
  document.getElementById("sourceState").textContent = available.length
    ? `已加载 ${available.join(" + ")}`
    : "没有找到数据源";

  const intradayMap = data.intraday ?? {};
  const btcRecord = intradayMap.btc ? [intradayMap.btc] : [];
  const intradayRecords = Object.entries(intradayMap)
    .filter(([key]) => key !== "btc")
    .map(([, value]) => value);
  const openRecords = Object.values(data.open_nowcast ?? {});
  const btcChanges = (data.recent_changes ?? []).filter((record) => record.market === "比特币");
  renderCards("intradayCards", intradayRecords);
  renderCards("btcCard", btcRecord);
  renderCards("openCards", openRecords);
  renderEvents("btcChangesList", btcChanges, "BTC 最近没有新的买卖变化。");
  renderEvents("changesList", data.recent_changes ?? [], "最近没有新的买卖变化。");
  renderEvents("historyList", data.recent_history ?? [], "最近没有运行历史。");
}

main().catch((error) => {
  document.getElementById("generatedAt").textContent = "加载失败";
  document.getElementById("sourceState").textContent = error.message;
  renderEmpty(document.getElementById("intradayCards"), "无法加载 dashboard.json。");
  renderEmpty(document.getElementById("btcCard"), "无法加载 dashboard.json。");
  renderEmpty(document.getElementById("openCards"), "无法加载 dashboard.json。");
  renderEmpty(document.getElementById("btcChangesList"), "无法加载 dashboard.json。");
  renderEmpty(document.getElementById("changesList"), "无法加载 dashboard.json。");
  renderEmpty(document.getElementById("historyList"), "无法加载 dashboard.json。");
  console.error(error);
});
