import { useMemo, useState } from "react";
import Papa from "papaparse";

const ENV_API_BASE = import.meta.env.VITE_API_BASE;
const API_BASE = ENV_API_BASE !== undefined ? ENV_API_BASE : "/api";

const ALL_FIELDS = [
  "customer_id",
  "recency_days",
  "frequency_180d",
  "monetary_180d",
  "return_rate_180d",
  "avg_discount_pct_180d",
  "avg_rating_180d",
  "category_diversity_180d",
  "ticket_count_90d",
  "negative_ticket_rate_90d",
  "avg_resolution_hours_90d",
  "days_since_signup",
  "city_tier",
  "age_group",
  "acquisition_channel",
  "loyalty_tier",
  "preferred_category",
  "marketing_consent",
  "sessions_30d",
  "product_views_30d",
  "cart_adds_30d",
  "wishlist_adds_30d",
  "abandoned_carts_30d",
  "email_opens_30d",
  "campaign_clicks_30d",
  "last_visit_days_ago",
];

const OPTIONAL_FIELDS = ["avg_rating_180d", "loyalty_tier"];

const INT_FIELDS = [
  "recency_days",
  "frequency_180d",
  "category_diversity_180d",
  "ticket_count_90d",
  "days_since_signup",
  "sessions_30d",
  "product_views_30d",
  "cart_adds_30d",
  "wishlist_adds_30d",
  "abandoned_carts_30d",
  "email_opens_30d",
  "campaign_clicks_30d",
  "last_visit_days_ago",
];

const FLOAT_FIELDS = [
  "monetary_180d",
  "return_rate_180d",
  "avg_discount_pct_180d",
  "avg_rating_180d",
  "negative_ticket_rate_90d",
  "avg_resolution_hours_90d",
];

const REQUIRED_FIELDS = ALL_FIELDS.filter((f) => !OPTIONAL_FIELDS.includes(f));

const EXAMPLE_HIGH = {
  customer_id: "CUST00042",
  recency_days: 95,
  frequency_180d: 2,
  monetary_180d: 480.0,
  return_rate_180d: 0.0,
  avg_discount_pct_180d: 0.3,
  avg_rating_180d: 3.5,
  category_diversity_180d: 1,
  ticket_count_90d: 2,
  negative_ticket_rate_90d: 0.5,
  avg_resolution_hours_90d: 18.0,
  days_since_signup: 450,
  city_tier: "Tier 2",
  age_group: "25-34",
  acquisition_channel: "Instagram",
  loyalty_tier: "",
  preferred_category: "Skin Care",
  marketing_consent: "Yes",
  sessions_30d: 0,
  product_views_30d: 0,
  cart_adds_30d: 0,
  wishlist_adds_30d: 0,
  abandoned_carts_30d: 0,
  email_opens_30d: 1,
  campaign_clicks_30d: 0,
  last_visit_days_ago: 25,
};

const EXAMPLE_LOW = {
  customer_id: "CUST00007",
  recency_days: 5,
  frequency_180d: 12,
  monetary_180d: 8400.0,
  return_rate_180d: 0.05,
  avg_discount_pct_180d: 0.1,
  avg_rating_180d: 4.6,
  category_diversity_180d: 5,
  ticket_count_90d: 0,
  negative_ticket_rate_90d: 0.0,
  avg_resolution_hours_90d: 0.0,
  days_since_signup: 980,
  city_tier: "Tier 1",
  age_group: "35-44",
  acquisition_channel: "Organic",
  loyalty_tier: "Gold",
  preferred_category: "Fragrance",
  marketing_consent: "Yes",
  sessions_30d: 24,
  product_views_30d: 120,
  cart_adds_30d: 18,
  wishlist_adds_30d: 7,
  abandoned_carts_30d: 1,
  email_opens_30d: 12,
  campaign_clicks_30d: 4,
  last_visit_days_ago: 2,
};

const defaultForm = {
  customer_id: "",
  recency_days: "",
  frequency_180d: "",
  monetary_180d: "",
  return_rate_180d: "",
  avg_discount_pct_180d: "",
  avg_rating_180d: "",
  category_diversity_180d: "",
  ticket_count_90d: "",
  negative_ticket_rate_90d: "",
  avg_resolution_hours_90d: "",
  days_since_signup: "",
  city_tier: "Tier 2",
  age_group: "25-34",
  acquisition_channel: "",
  loyalty_tier: "",
  preferred_category: "",
  marketing_consent: "Yes",
  sessions_30d: "",
  product_views_30d: "",
  cart_adds_30d: "",
  wishlist_adds_30d: "",
  abandoned_carts_30d: "",
  email_opens_30d: "",
  campaign_clicks_30d: "",
  last_visit_days_ago: "",
};

const SPARKLINE = [12, 22, 16, 30, 20, 40, 26, 34, 18, 28];

function toNumber(value, isInt) {
  if (value === "" || value === null || value === undefined) {
    return null;
  }
  const num = isInt ? parseInt(value, 10) : parseFloat(value);
  return Number.isNaN(num) ? null : num;
}

function buildPayload(form) {
  const payload = { ...form };

  INT_FIELDS.forEach((field) => {
    payload[field] = toNumber(form[field], true);
  });
  FLOAT_FIELDS.forEach((field) => {
    payload[field] = toNumber(form[field], false);
  });

  payload.customer_id = String(form.customer_id || "").trim();
  payload.acquisition_channel = String(form.acquisition_channel || "").trim();
  payload.preferred_category = String(form.preferred_category || "").trim();

  payload.loyalty_tier = form.loyalty_tier ? form.loyalty_tier : null;
  if (payload.avg_rating_180d === null) {
    payload.avg_rating_180d = null;
  }

  return payload;
}

function validateForm(form) {
  if (!String(form.customer_id || "").trim()) {
    return "Customer ID is required.";
  }
  if (!String(form.acquisition_channel || "").trim()) {
    return "Acquisition channel is required.";
  }
  if (!String(form.preferred_category || "").trim()) {
    return "Preferred category is required.";
  }

  for (const field of REQUIRED_FIELDS) {
    if (field === "customer_id") continue;
    if (field === "acquisition_channel" || field === "preferred_category") continue;
    const value = form[field];
    if (value === "" || value === null || value === undefined) {
      return `Missing value for ${field}.`;
    }
  }

  for (const field of INT_FIELDS) {
    if (form[field] === "") {
      return `Missing numeric value for ${field}.`;
    }
    if (toNumber(form[field], true) === null) {
      return `Invalid number for ${field}.`;
    }
  }

  for (const field of FLOAT_FIELDS) {
    if (field === "avg_rating_180d" && form[field] === "") continue;
    if (form[field] === "") {
      return `Missing numeric value for ${field}.`;
    }
    if (toNumber(form[field], false) === null) {
      return `Invalid number for ${field}.`;
    }
  }

  return null;
}

function downloadCsv(rows, filename) {
  if (!rows || !rows.length) return;
  const headers = Object.keys(rows[0]);
  const csv = [headers.join(",")]
    .concat(
      rows.map((row) =>
        headers
          .map((h) => {
            const val = row[h] ?? "";
            const escaped = String(val).replace(/"/g, '""');
            return `"${escaped}"`;
          })
          .join(",")
      )
    )
    .join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function App() {
  const [form, setForm] = useState(defaultForm);
  const [singleResult, setSingleResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [batchMeta, setBatchMeta] = useState(null);
  const [status, setStatus] = useState({ type: "idle", message: "" });
  const [loading, setLoading] = useState(false);
  const [csvName, setCsvName] = useState("");

  const predictions = useMemo(() => {
    if (batchResults.length) return batchResults;
    if (singleResult) return [singleResult];
    return [];
  }, [batchResults, singleResult]);

  const singleResultKey = useMemo(() => {
    if (!singleResult) return "single";
    return `${singleResult.customer_id}-${singleResult.churn_probability}`;
  }, [singleResult]);

  const batchKey = useMemo(() => {
    if (!batchMeta) return "batch";
    return `${batchMeta.total}-${batchMeta.highRisk}-${batchMeta.timeMs}`;
  }, [batchMeta]);

  const averageProb = useMemo(() => {
    if (!predictions.length) return null;
    const total = predictions.reduce(
      (acc, row) => acc + Number(row.churn_probability || 0),
      0
    );
    return total / predictions.length;
  }, [predictions]);

  const averageProbDisplay = useMemo(() => {
    if (averageProb === null) return "—";
    return `${(averageProb * 100).toFixed(1)}%`;
  }, [averageProb]);

  const thresholdUsed = useMemo(() => {
    if (!predictions.length) return "—";
    const raw = predictions[0]?.threshold_used;
    const num = Number(raw);
    return Number.isFinite(num) ? num.toFixed(3) : String(raw ?? "—");
  }, [predictions]);

  const topRisk = useMemo(() => {
    if (!predictions.length) return null;
    return [...predictions].sort(
      (a, b) => Number(b.churn_probability) - Number(a.churn_probability)
    )[0];
  }, [predictions]);

  const riskSummary = useMemo(() => {
    const levels = ["Very High", "High", "Medium", "Low", "Very Low"];
    const counts = levels.reduce((acc, level) => {
      acc[level] = 0;
      return acc;
    }, {});
    predictions.forEach((p) => {
      if (counts[p.risk_level] !== undefined) {
        counts[p.risk_level] += 1;
      }
    });
    return { levels, counts, total: predictions.length };
  }, [predictions]);

  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  function applyExample(example) {
    setForm({
      ...defaultForm,
      ...Object.fromEntries(
        Object.entries(example).map(([k, v]) => [k, v === null ? "" : v])
      ),
    });
    setStatus({ type: "idle", message: "" });
    setSingleResult(null);
  }

  async function submitSingle() {
    const error = validateForm(form);
    if (error) {
      setStatus({ type: "error", message: error });
      return;
    }

    setLoading(true);
    setStatus({ type: "loading", message: "Scoring customer..." });

    try {
      const payload = buildPayload(form);
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const data = await res.json();
          detail = data.detail ? JSON.stringify(data.detail) : detail;
        } catch (err) {
          // Ignore parse errors
        }
        throw new Error(detail);
      }

      const data = await res.json();
      setSingleResult(data);
      setBatchResults([]);
      setBatchMeta(null);
      setStatus({ type: "success", message: "Prediction ready." });
    } catch (err) {
      setStatus({ type: "error", message: err.message || "Request failed." });
    } finally {
      setLoading(false);
    }
  }

  function resetAll() {
    setForm(defaultForm);
    setSingleResult(null);
    setBatchResults([]);
    setBatchMeta(null);
    setCsvName("");
    setStatus({ type: "idle", message: "" });
  }

  function parseBatchRows(rows) {
    const requiredCsv = ALL_FIELDS.filter(
      (field) => !OPTIONAL_FIELDS.includes(field)
    );

    const cleaned = rows.filter((row) =>
      Object.values(row).some((value) => String(value).trim() !== "")
    );

    if (!cleaned.length) {
      throw new Error("CSV has no data rows.");
    }

    if (cleaned.length > 500) {
      throw new Error("CSV exceeds the 500 customer limit.");
    }

    const parsedRows = cleaned.map((row, index) => {
      const obj = { ...row };
      requiredCsv.forEach((field) => {
        if (!Object.prototype.hasOwnProperty.call(obj, field)) {
          throw new Error(`Missing column: ${field}`);
        }
      });

      const payload = buildPayload(obj);

      for (const field of requiredCsv) {
        if (payload[field] === null || payload[field] === "") {
          throw new Error(`Row ${index + 1} missing ${field}`);
        }
      }

      return payload;
    });

    return parsedRows;
  }

  async function handleCsvUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setCsvName(file.name);
    setLoading(true);
    setStatus({ type: "loading", message: "Parsing CSV..." });

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: async (results) => {
        try {
          const rows = parseBatchRows(results.data || []);
          setStatus({ type: "loading", message: "Scoring batch..." });

          const res = await fetch(`${API_BASE}/batch_predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ customers: rows }),
          });

          if (!res.ok) {
            let detail = `Request failed (${res.status})`;
            try {
              const data = await res.json();
              detail = data.detail ? JSON.stringify(data.detail) : detail;
            } catch (err) {
              // Ignore parse errors
            }
            throw new Error(detail);
          }

          const data = await res.json();
          setBatchResults(data.predictions || []);
          setBatchMeta({
            total: data.total_customers,
            highRisk: data.high_risk_count,
            timeMs: data.processing_time_ms,
          });
          setSingleResult(null);
          setStatus({ type: "success", message: "Batch scored." });
        } catch (err) {
          setStatus({ type: "error", message: err.message || "CSV failed." });
        } finally {
          setLoading(false);
        }
      },
      error: (err) => {
        setLoading(false);
        setStatus({ type: "error", message: err.message || "CSV failed." });
      },
    });
  }

  function downloadTemplate() {
    const headers = ALL_FIELDS.join(",");
    const blob = new Blob([headers + "\n"], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "churn_template.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-text">
          <p className="eyebrow">D2C Churn Intelligence</p>
          <h1>Churn Scoring Console</h1>
          <p className="subtitle">
            Score known customers or brand new app users with the same model
            logic used in Part 3.
          </p>
          <div className="hero-badges">
            <span className="badge">Real-time scoring</span>
            <span className="badge">CSV batch up to 500</span>
            <span className="badge">Risk explanations</span>
          </div>
          <div className="status">
            <span className={`status-pill ${status.type}`}>
              {status.type === "idle" ? "Ready" : status.message}
            </span>
            <span className="status-note">API base: {API_BASE}</span>
          </div>
        </div>
        <div className="hero-card">
          <h2>Quick batch upload</h2>
          <p>
            Upload a CSV to score up to 500 customers.
          </p>
          <div className="batch-actions">
            <label className="file-input">
              <input type="file" accept=".csv" onChange={handleCsvUpload} />
              <span>Upload CSV</span>
            </label>
            <button className="btn ghost" onClick={downloadTemplate}>
              Download template
            </button>
          </div>
          {csvName ? <p className="small">Loaded: {csvName}</p> : null}
          {batchMeta ? (
            <div key={batchKey} className="batch-meta animate-in">
              <div>
                <span>Total</span>
                <strong>{batchMeta.total}</strong>
              </div>
              <div>
                <span>High risk</span>
                <strong>{batchMeta.highRisk}</strong>
              </div>
              <div>
                <span>Time</span>
                <strong>{batchMeta.timeMs} ms</strong>
              </div>
            </div>
          ) : (
            <details className="csv-details">
              <summary>CSV columns (30)</summary>
              <p className="small muted">{ALL_FIELDS.join(", ")}</p>
            </details>
          )}
        </div>
        <div className="hero-visual">
          <div className="visual-shell">
            <div className="orb">
              <span className="orb-core" />
              <span className="orb-ring ring-1" />
              <span className="orb-ring ring-2" />
              <span className="orb-ring ring-3" />
            </div>
            <div className="signal-grid">
              <div className="signal-card">
                <span>Signals</span>
                <strong>30</strong>
              </div>
              <div className="signal-card">
                <span>Snapshot</span>
                <strong>2025-09-30</strong>
              </div>
              <div className="signal-card">
                <span>Avg risk</span>
                <strong>{averageProbDisplay}</strong>
              </div>
            </div>
          </div>
          <div className="sparkline-card">
            <div className="sparkline">
              {SPARKLINE.map((height, index) => (
                <span key={String(height) + index} style={{ height }} />
              ))}
            </div>
            <p className="small muted">Live risk pulse</p>
          </div>
        </div>
      </header>

      <section className="insight-strip">
        <div className="insight-card">
          <p className="eyebrow">Decision threshold</p>
          <h3>{thresholdUsed}</h3>
          <p className="muted">Optimized by business cost matrix.</p>
        </div>
        <div className="insight-card">
          <p className="eyebrow">Top risk customer</p>
          <h3>{topRisk ? topRisk.customer_id : "—"}</h3>
          <p className="muted">
            {topRisk
              ? `${topRisk.risk_level} risk, ${topRisk.churn_probability}`
              : "Score customers to surface the highest risk."}
          </p>
        </div>
        <div className="insight-card">
          <p className="eyebrow">Scored today</p>
          <h3>{predictions.length}</h3>
          <p className="muted">Single plus batch results in view.</p>
        </div>
      </section>

      <main className="content">
        <section className="panel form-panel">
          <div className="panel-header">
            <h2>Single customer</h2>
            <p>Fill required fields and score instantly.</p>
          </div>

          <div className="quick-actions">
            <span className="label">Quick fill</span>
            <div className="quick-buttons">
              <button className="btn ghost" onClick={() => applyExample(EXAMPLE_HIGH)}>
                High-risk
              </button>
              <button className="btn ghost" onClick={() => applyExample(EXAMPLE_LOW)}>
                Low-risk
              </button>
              <button className="btn ghost" onClick={resetAll}>
                Reset
              </button>
            </div>
          </div>

          <div className="form-grid">
            <div className="field">
              <label>Customer ID</label>
              <input
                type="text"
                name="customer_id"
                value={form.customer_id}
                onChange={handleChange}
                placeholder="CUST12345"
              />
            </div>

            <div className="field">
              <label>City tier</label>
              <select name="city_tier" value={form.city_tier} onChange={handleChange}>
                <option value="Tier 1">Tier 1</option>
                <option value="Tier 2">Tier 2</option>
                <option value="Tier 3">Tier 3</option>
              </select>
            </div>

            <div className="field">
              <label>Age group</label>
              <select name="age_group" value={form.age_group} onChange={handleChange}>
                <option value="18-24">18-24</option>
                <option value="25-34">25-34</option>
                <option value="35-44">35-44</option>
                <option value="45+">45+</option>
              </select>
            </div>

            <div className="field">
              <label>Acquisition channel</label>
              <input
                type="text"
                name="acquisition_channel"
                value={form.acquisition_channel}
                onChange={handleChange}
                placeholder="Instagram, Organic, Referral"
              />
            </div>

            <div className="field">
              <label>Loyalty tier (optional)</label>
              <select name="loyalty_tier" value={form.loyalty_tier} onChange={handleChange}>
                <option value="">None</option>
                <option value="Silver">Silver</option>
                <option value="Gold">Gold</option>
                <option value="Platinum">Platinum</option>
              </select>
            </div>

            <div className="field">
              <label>Preferred category</label>
              <input
                type="text"
                name="preferred_category"
                value={form.preferred_category}
                onChange={handleChange}
                placeholder="Skin Care, Makeup, Fragrance"
              />
            </div>

            <div className="field">
              <label>Marketing consent</label>
              <select
                name="marketing_consent"
                value={form.marketing_consent}
                onChange={handleChange}
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>

          <div className="section-divider">
            <h3>RFM and value</h3>
          </div>
          <div className="form-grid">
            <div className="field">
              <label>Recency days</label>
              <input
                type="number"
                name="recency_days"
                min="0"
                max="1000"
                value={form.recency_days}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Frequency 180d</label>
              <input
                type="number"
                name="frequency_180d"
                min="0"
                max="200"
                value={form.frequency_180d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Monetary 180d</label>
              <input
                type="number"
                name="monetary_180d"
                min="0"
                max="200000"
                step="1"
                value={form.monetary_180d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Return rate 180d</label>
              <input
                type="number"
                name="return_rate_180d"
                min="0"
                max="1"
                step="0.01"
                value={form.return_rate_180d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Avg discount pct 180d</label>
              <input
                type="number"
                name="avg_discount_pct_180d"
                min="0"
                max="1"
                step="0.01"
                value={form.avg_discount_pct_180d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Avg rating 180d (optional)</label>
              <input
                type="number"
                name="avg_rating_180d"
                min="1"
                max="5"
                step="0.1"
                value={form.avg_rating_180d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Category diversity 180d</label>
              <input
                type="number"
                name="category_diversity_180d"
                min="0"
                max="10"
                value={form.category_diversity_180d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Days since signup</label>
              <input
                type="number"
                name="days_since_signup"
                min="0"
                max="3000"
                value={form.days_since_signup}
                onChange={handleChange}
              />
            </div>
          </div>

          <div className="section-divider">
            <h3>Support and service</h3>
          </div>
          <div className="form-grid">
            <div className="field">
              <label>Ticket count 90d</label>
              <input
                type="number"
                name="ticket_count_90d"
                min="0"
                max="50"
                value={form.ticket_count_90d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Negative ticket rate 90d</label>
              <input
                type="number"
                name="negative_ticket_rate_90d"
                min="0"
                max="1"
                step="0.01"
                value={form.negative_ticket_rate_90d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Avg resolution hours 90d</label>
              <input
                type="number"
                name="avg_resolution_hours_90d"
                min="0"
                max="500"
                step="0.1"
                value={form.avg_resolution_hours_90d}
                onChange={handleChange}
              />
            </div>
          </div>

          <div className="section-divider">
            <h3>Web activity (30d)</h3>
          </div>
          <div className="form-grid">
            <div className="field">
              <label>Sessions 30d</label>
              <input
                type="number"
                name="sessions_30d"
                min="0"
                max="500"
                value={form.sessions_30d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Product views 30d</label>
              <input
                type="number"
                name="product_views_30d"
                min="0"
                max="2000"
                value={form.product_views_30d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Cart adds 30d</label>
              <input
                type="number"
                name="cart_adds_30d"
                min="0"
                max="500"
                value={form.cart_adds_30d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Wishlist adds 30d</label>
              <input
                type="number"
                name="wishlist_adds_30d"
                min="0"
                max="500"
                value={form.wishlist_adds_30d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Abandoned carts 30d</label>
              <input
                type="number"
                name="abandoned_carts_30d"
                min="0"
                max="200"
                value={form.abandoned_carts_30d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Email opens 30d</label>
              <input
                type="number"
                name="email_opens_30d"
                min="0"
                max="500"
                value={form.email_opens_30d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Campaign clicks 30d</label>
              <input
                type="number"
                name="campaign_clicks_30d"
                min="0"
                max="200"
                value={form.campaign_clicks_30d}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Last visit days ago</label>
              <input
                type="number"
                name="last_visit_days_ago"
                min="0"
                max="365"
                value={form.last_visit_days_ago}
                onChange={handleChange}
              />
            </div>
          </div>

          <div className="form-actions">
            <button className="btn primary" onClick={submitSingle} disabled={loading}>
              {loading ? "Scoring..." : "Score customer"}
            </button>
            <button className="btn ghost" onClick={resetAll} disabled={loading}>
              Clear all
            </button>
          </div>
        </section>

        <section className="panel results-panel">
          <div className="panel-header">
            <h2>Results</h2>
            <p>Single and batch predictions appear here.</p>
          </div>

          {loading ? (
            <div className="skeleton-stack">
              <div className="skeleton-card">
                <div className="skeleton-line w-40" />
                <div className="skeleton-line w-70" />
                <div className="skeleton-line w-55" />
              </div>
              <div className="skeleton-card compact">
                <div className="skeleton-pill" />
                <div className="skeleton-pill" />
                <div className="skeleton-pill" />
              </div>
            </div>
          ) : null}

          {singleResult ? (
            <div key={singleResultKey} className="result-card animate-in">
              <div>
                <p className="label">Customer</p>
                <h3>{singleResult.customer_id}</h3>
              </div>
              <div className="result-metrics">
                <div>
                  <p className="label">Churn probability</p>
                  <strong>{singleResult.churn_probability}</strong>
                </div>
                <div>
                  <p className="label">Risk level</p>
                  <strong>{singleResult.risk_level}</strong>
                </div>
                <div>
                  <p className="label">Confidence</p>
                  <strong>{singleResult.confidence}</strong>
                </div>
              </div>
              <div className="result-note">
                <p>{singleResult.risk_explanation}</p>
                <span>Threshold: {singleResult.threshold_used}</span>
              </div>
            </div>
          ) : null}

          <div className="risk-bars">
            {riskSummary.levels.map((level) => {
              const count = riskSummary.counts[level];
              const width = riskSummary.total
                ? Math.round((count / riskSummary.total) * 100)
                : 0;
              return (
                <div key={level} className="risk-row">
                  <span>{level}</span>
                  <div className="bar">
                    <div className="fill" style={{ width: `${width}%` }} />
                  </div>
                  <strong>{count}</strong>
                </div>
              );
            })}
          </div>

          {predictions.length ? (
            <div key={batchKey} className="table-wrap animate-in">
              <div className="table-actions">
                <button
                  className="btn ghost"
                  onClick={() => downloadCsv(predictions, "churn_predictions.csv")}
                >
                  Download results
                </button>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Customer ID</th>
                    <th>Probability</th>
                    <th>Class</th>
                    <th>Risk</th>
                    <th>Confidence</th>
                    <th>Threshold</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((row) => (
                    <tr key={row.customer_id}>
                      <td>{row.customer_id}</td>
                      <td>{row.churn_probability}</td>
                      <td>{row.predicted_class}</td>
                      <td>{row.risk_level}</td>
                      <td>{row.confidence}</td>
                      <td>{row.threshold_used}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="small muted">No predictions yet.</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
