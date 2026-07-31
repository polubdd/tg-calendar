const MONTHS = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"];
const WD = ["вс", "пн", "вт", "ср", "чт", "пт", "сб"];
const $ = (id) => document.getElementById(id);
let cursor = new Date();
cursor.setDate(1);
let todayIso = new Date().toISOString().slice(0, 10);
let current = null;

const iso = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (r.status === 204) return null;
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Ошибка запроса");
  return data;
}

async function renderMonth() {
  $("month").textContent = `${MONTHS[cursor.getMonth()]} ${cursor.getFullYear()}`;
  const first = new Date(cursor);
  const start = new Date(first);
  start.setDate(1 - ((first.getDay() + 6) % 7)); // с понедельника
  const end = new Date(start);
  end.setDate(start.getDate() + 42);
  const events = await api(`/api/events?start=${iso(start)}&end=${iso(end)}`);
  const byDay = {};
  for (const e of events) (byDay[e.start.slice(0, 10)] ||= []).push(e);

  const grid = $("grid");
  grid.innerHTML = "";
  for (let i = 0; i < 42; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    const key = iso(d);
    const cell = document.createElement("div");
    cell.className = "day" + (d.getMonth() !== cursor.getMonth() ? " other" : "") + (key === todayIso ? " today" : "");
    cell.innerHTML = `<div class="num">${d.getDate()}</div>`;
    const list = (byDay[key] || []).sort((a, b) => (b.all_day - a.all_day) || a.start.localeCompare(b.start));
    for (const e of list) {
      const b = document.createElement("button");
      b.className = "ev" + (e.status === "draft" ? " draft" : "") + (e.all_day ? " allday" : "");
      b.innerHTML = (e.all_day ? "" : `<span class="t">${e.start.slice(11, 16)}</span>`) + esc(e.title) + (e.recurring ? " ↻" : "");
      b.title = e.title;
      b.onclick = () => openDetails(e.id, e.start);
      cell.appendChild(b);
    }
    grid.appendChild(cell);
  }
}

async function renderDrafts() {
  const drafts = await api("/api/drafts");
  $("draftcount").textContent = drafts.length ? `· ${drafts.length}` : "";
  const ul = $("drafts");
  ul.innerHTML = drafts.length ? "" : '<li class="muted small">Пусто</li>';
  for (const e of drafts) {
    const li = document.createElement("li");
    const when = e.when ? `${e.when}${e.all_day ? "" : " в " + e.start.slice(11, 16)}` : fmtWhen(e.start, e.all_day);
    li.innerHTML = `<span><b>${esc(e.title)}</b><br><span class="muted small">${esc(when)}${e.source === "telegram" ? " · из Telegram" : ""}</span></span>`;
    const ok = document.createElement("button");
    ok.textContent = "✓";
    ok.title = "Подтвердить";
    ok.onclick = async () => { await api(`/api/events/${e.id}/confirm`, { method: "POST" }); refresh(); };
    const no = document.createElement("button");
    no.textContent = "✕";
    no.title = "Отклонить";
    no.onclick = async () => { await api(`/api/events/${e.id}/reject`, { method: "POST" }); refresh(); };
    li.append(ok, no);
    ul.appendChild(li);
  }
}

function fmtWhen(start, allDay) {
  const d = new Date(start.length === 10 ? start + "T00:00" : start);
  const date = `${WD[d.getDay()]}, ${d.getDate()} ${MONTHS[d.getMonth()].toLowerCase()}`;
  return allDay ? `${date}, весь день` : `${date}, ${start.slice(11, 16)}`;
}

async function openDetails(id, occurrenceStart) {
  const e = await api(`/api/events/${id}`);
  current = e;
  $("d-title").textContent = e.title;
  $("d-when").textContent = e.rrule
    ? `${e.when}${e.all_day ? "" : " в " + e.start.slice(11, 16)} (с ${e.start.slice(0, 10)})`
    : fmtWhen(occurrenceStart, e.all_day) + (e.end && !e.all_day ? `–${e.end.slice(11, 16)}` : "");
  const src = { telegram: "из переписки в Telegram", chat: "из чата", manual: "вручную" }[e.source] || e.source;
  $("d-meta").textContent = [e.status === "draft" ? "черновик" : "подтверждено", src, e.notes, e.location].filter(Boolean).join(" · ");
  $("d-confirm").style.display = e.status === "draft" ? "" : "none";
  $("details").showModal();
}

$("details").addEventListener("close", async () => {
  const v = $("details").returnValue;
  if (!current) return;
  if (v === "confirm") await api(`/api/events/${current.id}/confirm`, { method: "POST" });
  if (v === "delete") {
    // Событие из Telegram помечаем отклонённым, чтобы бот не добавил его снова.
    if (current.source === "telegram") await api(`/api/events/${current.id}/reject`, { method: "POST" });
    else await api(`/api/events/${current.id}`, { method: "DELETE" });
  }
  current = null;
  if (v === "confirm" || v === "delete") refresh();
});

function say(html, cls) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.innerHTML = html;
  $("chatlog").appendChild(div);
  $("chatlog").scrollTop = 1e9;
  return div;
}

$("chatform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = $("chatinput").value.trim();
  if (!text) return;
  $("chatinput").value = "";
  say(esc(text), "me");
  const wait = say('<span class="muted">думаю…</span>', "bot");
  try {
    const p = await api("/api/chat", { method: "POST", body: JSON.stringify({ text }) });
    const warn = p.ambiguous.map((a) => `<div class="warn">⚠ ${esc(a)}</div>`).join("");
    const how = p.source === "llm" ? '<div class="muted small">разобрала модель</div>' : "";
    wait.className = "msg bot proposal";
    wait.innerHTML = `<b>${esc(p.event.title)}</b>${esc(p.description)}${p.event.rrule ? `<div class="muted small">${esc(p.event.rrule)}</div>` : ""}${warn}${how}
      <div class="row"><button class="add">Добавить</button><button class="ghost cancel">Отмена</button></div>`;
    wait.querySelector(".add").onclick = async () => {
      await api("/api/events", { method: "POST", body: JSON.stringify(p.event) });
      wait.querySelector(".row").innerHTML = '<span class="muted small">добавлено</span>';
      const d = new Date(p.event.start.slice(0, 10) + "T00:00");
      cursor = new Date(d.getFullYear(), d.getMonth(), 1);
      refresh();
    };
    wait.querySelector(".cancel").onclick = () => { wait.querySelector(".row").innerHTML = '<span class="muted small">отменено</span>'; };
  } catch (e) {
    wait.className = "msg bot err";
    wait.textContent = e.message;
  }
});

$("prev").onclick = () => { cursor.setMonth(cursor.getMonth() - 1); renderMonth(); };
$("next").onclick = () => { cursor.setMonth(cursor.getMonth() + 1); renderMonth(); };
$("today").onclick = () => { cursor = new Date(); cursor.setDate(1); renderMonth(); };

async function loadStatus() {
  const s = await api("/api/status");
  todayIso = s.today;
  const url = location.origin + s.ics_url;
  $("subscribe").onclick = async () => {
    try { await navigator.clipboard.writeText(url); alert("Ссылка на подписку скопирована:\n" + url); }
    catch { prompt("Ссылка на подписку:", url); }
  };
  $("status").innerHTML = [
    `Модель: ${esc(s.model)} (num_ctx ${s.num_ctx}) — ${s.llm ? "доступна" : "<b>не запущена</b>, работают только правила"}`,
    `Бот: ${s.bot_configured ? (s.owners ? "настроен" : "нет ALLOWED_OWNER_IDS") : "токен не задан"}`,
    `Часовой пояс: ${esc(s.tz)}`,
  ].join("<br>");
}

function refresh() { renderMonth(); renderDrafts(); }
loadStatus().finally(refresh);
setInterval(renderDrafts, 30000); // бот может добавить черновик, пока страница открыта
