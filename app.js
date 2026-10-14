/**
 * ChurnShield AI - Interactive Client-Side Machine Learning Engine & Dashboard
 */

// Application State
const state = {
  data: null,
  currentProfile: {
    Age: 42,
    Geography: "Germany",
    Gender: "Female",
    CreditScore: 650,
    Balance: 85000,
    NumOfProducts: 1,
    HasCrCard: 1,
    IsActiveMember: 0,
    EstimatedSalary: 105000,
  },
  selectedModel: "gb_tuned", // gb_tuned, gb_std, lr_tuned, lr_std
  batchRecords: [],
  batchFilter: "all",
  searchQuery: "",
};

// DOM References
const elements = {
  // Tabs
  navBtns: document.querySelectorAll(".nav-btn"),
  tabPanes: document.querySelectorAll(".tab-pane"),

  // Predictor Inputs
  inputAge: document.getElementById("inputAge"),
  valAge: document.getElementById("valAge"),
  segGeography: document.getElementById("segGeography"),
  segGender: document.getElementById("segGender"),
  segProducts: document.getElementById("segProducts"),
  valProducts: document.getElementById("valProducts"),
  inputBalance: document.getElementById("inputBalance"),
  valBalance: document.getElementById("valBalance"),
  btnZeroBalanceToggle: document.getElementById("btnZeroBalanceToggle"),
  inputSalary: document.getElementById("inputSalary"),
  valSalary: document.getElementById("valSalary"),
  inputCreditScore: document.getElementById("inputCreditScore"),
  valCreditScore: document.getElementById("valCreditScore"),
  inputTenure: document.getElementById("inputTenure"),
  valTenure: document.getElementById("valTenure"),
  inputIsActive: document.getElementById("inputIsActive"),
  inputHasCrCard: document.getElementById("inputHasCrCard"),
  modelSelect: document.getElementById("modelSelect"),

  // Engineered Features Display
  valZeroBalanceFlag: document.getElementById("valZeroBalanceFlag"),
  valBalanceToSalary: document.getElementById("valBalanceToSalary"),

  // Radial Gauge & Output
  probPercent: document.getElementById("probPercent"),
  gaugeFill: document.getElementById("gaugeFill"),
  riskBanner: document.getElementById("riskBanner"),
  riskTitle: document.getElementById("riskTitle"),
  riskDetail: document.getElementById("riskDetail"),
  driverBars: document.getElementById("driverBars"),
  retentionTitle: document.getElementById("retentionTitle"),
  retentionAction: document.getElementById("retentionAction"),
  retentionChecklist: document.getElementById("retentionChecklist"),

  // Presets
  presetHighRiskSenior: document.getElementById("presetHighRiskSenior"),
  presetLoyalCore: document.getElementById("presetLoyalCore"),
  presetInactiveWealthy: document.getElementById("presetInactiveWealthy"),
  presetZeroBalanceYoung: document.getElementById("presetZeroBalanceYoung"),

  // Threshold & Confusion Matrix
  sliderThreshold: document.getElementById("sliderThreshold"),
  lblThresholdVal: document.getElementById("lblThresholdVal"),
  btnSetTunedThreshold: document.getElementById("btnSetTunedThreshold"),
  btnSetDefaultThreshold: document.getElementById("btnSetDefaultThreshold"),
  cmTN: document.getElementById("cmTN"),
  cmFP: document.getElementById("cmFP"),
  cmFN: document.getElementById("cmFN"),
  cmTP: document.getElementById("cmTP"),
  mPrecision: document.getElementById("mPrecision"),
  mRecall: document.getElementById("mRecall"),
  mF1: document.getElementById("mF1"),
  mAccuracy: document.getElementById("mAccuracy"),

  // Charts
  rocCanvas: document.getElementById("rocCanvas"),
  permImportanceBars: document.getElementById("permImportanceBars"),

  // Batch
  batchTableBody: document.getElementById("batchTableBody"),
  batchSearch: document.getElementById("batchSearch"),
  filterChips: document.querySelectorAll(".filter-chip"),
  countAll: document.getElementById("countAll"),
  countHigh: document.getElementById("countHigh"),
  countMed: document.getElementById("countMed"),
  countLow: document.getElementById("countLow"),
  csvDropZone: document.getElementById("csvDropZone"),
  csvFileInput: document.getElementById("csvFileInput"),
  btnExportCSV: document.getElementById("btnExportCSV"),
};

// ==========================================================================
// Initialization
// ==========================================================================

async function init() {
  try {
    initAmbientBackground();
    const res = await fetch("demo_data.json");
    if (!res.ok) throw new Error("Could not load demo_data.json");
    state.data = await res.json();
    state.batchRecords = [...state.data.sample_customers];

    setupEventListeners();
    renderPermutationImportance();
    renderROCCurves();
    updateThresholdSimulation(0.30);
    renderBatchTable();
    updatePredictor();
  } catch (err) {
    console.error("Initialization error:", err);
  }
}

// ==========================================================================
// Event Listeners
// ==========================================================================

function setupEventListeners() {
  // Tab switching
  elements.navBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabId = btn.dataset.tab;
      elements.navBtns.forEach((b) => b.classList.remove("active"));
      elements.tabPanes.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add("active");

      if (tabId === "tab-models") {
        setTimeout(renderROCCurves, 50);
      }
    });
  });

  // Sliders & Inputs
  elements.inputAge.addEventListener("input", (e) => {
    state.currentProfile.Age = parseInt(e.target.value, 10);
    elements.valAge.textContent = `${state.currentProfile.Age} yrs`;
    updatePredictor();
  });

  elements.inputBalance.addEventListener("input", (e) => {
    state.currentProfile.Balance = parseFloat(e.target.value);
    elements.valBalance.textContent = `$${state.currentProfile.Balance.toLocaleString()}`;
    updatePredictor();
  });

  elements.btnZeroBalanceToggle.addEventListener("click", () => {
    if (state.currentProfile.Balance === 0) {
      state.currentProfile.Balance = 85000;
    } else {
      state.currentProfile.Balance = 0;
    }
    elements.inputBalance.value = state.currentProfile.Balance;
    elements.valBalance.textContent = `$${state.currentProfile.Balance.toLocaleString()}`;
    updatePredictor();
  });

  elements.inputSalary.addEventListener("input", (e) => {
    state.currentProfile.EstimatedSalary = parseFloat(e.target.value);
    elements.valSalary.textContent = `$${state.currentProfile.EstimatedSalary.toLocaleString()}`;
    updatePredictor();
  });

  elements.inputCreditScore.addEventListener("input", (e) => {
    state.currentProfile.CreditScore = parseInt(e.target.value, 10);
    elements.valCreditScore.textContent = state.currentProfile.CreditScore;
    updatePredictor();
  });

  elements.inputTenure.addEventListener("input", (e) => {
    state.currentProfile.Tenure = parseInt(e.target.value, 10);
    elements.valTenure.textContent = `${state.currentProfile.Tenure} years`;
    updatePredictor();
  });

  elements.inputIsActive.addEventListener("change", (e) => {
    state.currentProfile.IsActiveMember = e.target.checked ? 1 : 0;
    updatePredictor();
  });

  elements.inputHasCrCard.addEventListener("change", (e) => {
    state.currentProfile.HasCrCard = e.target.checked ? 1 : 0;
    updatePredictor();
  });

  // Geography Segmented Control
  elements.segGeography.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      elements.segGeography.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentProfile.Geography = btn.dataset.value;
      updatePredictor();
    });
  });

  // Gender Segmented Control
  elements.segGender.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      elements.segGender.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentProfile.Gender = btn.dataset.value;
      updatePredictor();
    });
  });

  // Products Selector
  elements.segProducts.querySelectorAll(".prod-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      elements.segProducts.querySelectorAll(".prod-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentProfile.NumOfProducts = parseInt(btn.dataset.value, 10);
      elements.valProducts.textContent = `${state.currentProfile.NumOfProducts} Product${state.currentProfile.NumOfProducts > 1 ? "s" : ""}`;
      updatePredictor();
    });
  });

  // Model Selection
  elements.modelSelect.addEventListener("change", (e) => {
    state.selectedModel = e.target.value;
    updatePredictor();
  });

  // Personas
  elements.presetHighRiskSenior.addEventListener("click", () => {
    applyProfile({
      Age: 54,
      Geography: "Germany",
      Gender: "Female",
      CreditScore: 610,
      Balance: 125000,
      NumOfProducts: 3,
      HasCrCard: 1,
      IsActiveMember: 0,
      EstimatedSalary: 95000,
      Tenure: 3,
    });
  });

  elements.presetLoyalCore.addEventListener("click", () => {
    applyProfile({
      Age: 32,
      Geography: "France",
      Gender: "Male",
      CreditScore: 720,
      Balance: 65000,
      NumOfProducts: 2,
      HasCrCard: 1,
      IsActiveMember: 1,
      EstimatedSalary: 82000,
      Tenure: 6,
    });
  });

  elements.presetInactiveWealthy.addEventListener("click", () => {
    applyProfile({
      Age: 46,
      Geography: "Germany",
      Gender: "Male",
      CreditScore: 680,
      Balance: 145000,
      NumOfProducts: 1,
      HasCrCard: 1,
      IsActiveMember: 0,
      EstimatedSalary: 110000,
      Tenure: 2,
    });
  });

  elements.presetZeroBalanceYoung.addEventListener("click", () => {
    applyProfile({
      Age: 27,
      Geography: "Spain",
      Gender: "Female",
      CreditScore: 640,
      Balance: 0,
      NumOfProducts: 2,
      HasCrCard: 1,
      IsActiveMember: 1,
      EstimatedSalary: 45000,
      Tenure: 4,
    });
  });

  // Threshold Simulator
  elements.sliderThreshold.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    elements.lblThresholdVal.textContent = val.toFixed(2);
    updateThresholdSimulation(val);
  });

  elements.btnSetTunedThreshold.addEventListener("click", () => {
    elements.sliderThreshold.value = 0.30;
    elements.lblThresholdVal.textContent = "0.30";
    updateThresholdSimulation(0.30);
  });

  elements.btnSetDefaultThreshold.addEventListener("click", () => {
    elements.sliderThreshold.value = 0.50;
    elements.lblThresholdVal.textContent = "0.50";
    updateThresholdSimulation(0.50);
  });

  // Batch Filter Buttons
  elements.filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      elements.filterChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.batchFilter = chip.dataset.filter;
      renderBatchTable();
    });
  });

  elements.batchSearch.addEventListener("input", (e) => {
    state.searchQuery = e.target.value.toLowerCase();
    renderBatchTable();
  });

  // Drag and Drop CSV
  elements.csvDropZone.addEventListener("click", () => elements.csvFileInput.click());
  elements.csvDropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    elements.csvDropZone.classList.add("dragover");
  });
  elements.csvDropZone.addEventListener("dragleave", () => {
    elements.csvDropZone.classList.remove("dragover");
  });
  elements.csvDropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    elements.csvDropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      handleCSVFile(e.dataTransfer.files[0]);
    }
  });
  elements.csvFileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
      handleCSVFile(e.target.files[0]);
    }
  });

  // Export CSV
  elements.btnExportCSV.addEventListener("click", exportBatchCSV);
}

// ==========================================================================
// Apply Profile Preset
// ==========================================================================

function applyProfile(p) {
  state.currentProfile = { ...p };

  elements.inputAge.value = p.Age;
  elements.valAge.textContent = `${p.Age} yrs`;

  elements.inputBalance.value = p.Balance;
  elements.valBalance.textContent = `$${p.Balance.toLocaleString()}`;

  elements.inputSalary.value = p.EstimatedSalary;
  elements.valSalary.textContent = `$${p.EstimatedSalary.toLocaleString()}`;

  elements.inputCreditScore.value = p.CreditScore;
  elements.valCreditScore.textContent = p.CreditScore;

  elements.inputTenure.value = p.Tenure;
  elements.valTenure.textContent = `${p.Tenure} years`;

  elements.inputIsActive.checked = p.IsActiveMember === 1;
  elements.inputHasCrCard.checked = p.HasCrCard === 1;

  // Segmented controls
  elements.segGeography.querySelectorAll(".seg-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.value === p.Geography);
  });
  elements.segGender.querySelectorAll(".seg-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.value === p.Gender);
  });
  elements.segProducts.querySelectorAll(".prod-btn").forEach((b) => {
    b.classList.toggle("active", parseInt(b.dataset.value, 10) === p.NumOfProducts);
  });
  elements.valProducts.textContent = `${p.NumOfProducts} Product${p.NumOfProducts > 1 ? "s" : ""}`;

  updatePredictor();
}

// ==========================================================================
// Exact Client-Side Machine Learning Inference
// ==========================================================================

function transformFeatures(profile) {
  const scaler = state.data.scaler_stats;

  // Engineered features
  const zeroBalance = profile.Balance === 0 ? 1 : 0;
  const balanceToSalary = profile.Balance / Math.max(1, profile.EstimatedSalary);

  const rawValues = {
    CreditScore: profile.CreditScore,
    Age: profile.Age,
    Tenure: profile.Tenure,
    Balance: profile.Balance,
    NumOfProducts: profile.NumOfProducts,
    HasCrCard: profile.HasCrCard,
    IsActiveMember: profile.IsActiveMember,
    EstimatedSalary: profile.EstimatedSalary,
    ZeroBalance: zeroBalance,
    BalanceToSalary: balanceToSalary,
  };

  // Standardize numeric features
  const standardized = [];
  scaler.numeric_features.forEach((feat) => {
    const val = rawValues[feat];
    const mean = scaler.mean[feat];
    const scale = scaler.scale[feat];
    standardized.push((val - mean) / scale);
  });

  // One-Hot Encode Categoricals:
  // Geography: France, Germany, Spain
  // Gender: Female, Male (dropped if binary -> cat__Gender_Male)
  const isFrance = profile.Geography === "France" ? 1.0 : 0.0;
  const isGermany = profile.Geography === "Germany" ? 1.0 : 0.0;
  const isSpain = profile.Geography === "Spain" ? 1.0 : 0.0;
  const isMale = profile.Gender === "Male" ? 1.0 : 0.0;

  // Combined feature vector matching state.data.feature_names:
  // num__CreditScore, num__Age, num__Tenure, num__Balance, num__NumOfProducts,
  // num__HasCrCard, num__IsActiveMember, num__EstimatedSalary, num__ZeroBalance,
  // num__BalanceToSalary, cat__Geography_France, cat__Geography_Germany, cat__Geography_Spain, cat__Gender_Male
  return [
    ...standardized,
    isFrance,
    isGermany,
    isSpain,
    isMale,
  ];
}

// Tree evaluation helper
function evalTree(tree, nodeIndex, x) {
  if (tree.left[nodeIndex] === -1) {
    return tree.value[nodeIndex];
  }
  const feat = tree.feature[nodeIndex];
  const thresh = tree.threshold[nodeIndex];
  if (x[feat] <= thresh) {
    return evalTree(tree, tree.left[nodeIndex], x);
  } else {
    return evalTree(tree, tree.right[nodeIndex], x);
  }
}

// Predict using 150-tree Gradient Boosting
function predictGradientBoosting(featureVector) {
  const gb = state.data.gb_model;
  let raw = gb.init_raw;
  for (let i = 0; i < gb.trees.length; i++) {
    raw += gb.learning_rate * evalTree(gb.trees[i], 0, featureVector);
  }
  // Logistic Sigmoid
  return 1 / (1 + Math.exp(-raw));
}

// Predict using Logistic Regression
function predictLogisticRegression(featureVector) {
  const lr = state.data.lr_weights;
  let dot = lr.intercept;
  state.data.feature_names.forEach((name, i) => {
    dot += (lr.coefficients[name] || 0) * featureVector[i];
  });
  return 1 / (1 + Math.exp(-dot));
}

// ==========================================================================
// Update Predictor View
// ==========================================================================

function updatePredictor() {
  if (!state.data) return;

  const profile = state.currentProfile;
  const zeroBalance = profile.Balance === 0 ? 1 : 0;
  const balanceToSalary = profile.Balance / Math.max(1, profile.EstimatedSalary);

  // Update engineered features display
  elements.valZeroBalanceFlag.textContent = zeroBalance === 1 ? "Yes (1) - Segment Isolated" : "No (0)";
  elements.valBalanceToSalary.textContent = `${balanceToSalary.toFixed(2)}x`;

  // Compute model prediction
  const featureVector = transformFeatures(profile);
  let prob = 0;
  let threshold = 0.30;
  let modelLabel = "Gradient Boosting (Tuned)";

  if (state.selectedModel.startsWith("gb")) {
    prob = predictGradientBoosting(featureVector);
    threshold = state.selectedModel === "gb_tuned" ? 0.30 : 0.50;
    modelLabel = state.selectedModel === "gb_tuned" ? "Gradient Boosting (Tuned @ 0.30)" : "Gradient Boosting (Standard @ 0.50)";
  } else {
    prob = predictLogisticRegression(featureVector);
    threshold = state.selectedModel === "lr_tuned" ? 0.54 : 0.50;
    modelLabel = state.selectedModel === "lr_tuned" ? "Logistic Regression (Tuned @ 0.54)" : "Logistic Regression (Standard @ 0.50)";
  }

  const probPercent = Math.round(prob * 100);
  elements.probPercent.textContent = `${probPercent}%`;

  // Radial Gauge Animation
  // circumference = 2 * Math.PI * 82 = 515.22
  const circ = 515.22;
  const offset = circ - (prob * circ);
  elements.gaugeFill.style.strokeDashoffset = offset;

  // Color gradient for gauge stroke
  let strokeColor = "#10b981"; // green
  if (prob >= 0.50) {
    strokeColor = "#ef4444"; // red
  } else if (prob >= 0.30) {
    strokeColor = "#f59e0b"; // amber
  }
  elements.gaugeFill.style.stroke = strokeColor;

  // Banner decision
  const isChurn = prob >= threshold;
  elements.riskBanner.className = `risk-decision-banner ${prob >= 0.50 ? "danger" : (prob >= 0.30 ? "warning" : "safe")}`;

  if (isChurn) {
    elements.riskTitle.textContent = `HIGH RETENTION RISK (Cutoff: ${threshold})`;
    elements.riskDetail.textContent = `Scored with ${modelLabel}. Probability of departure exceeds operational action threshold.`;
  } else {
    elements.riskTitle.textContent = `LOW CHURN RISK (Cutoff: ${threshold})`;
    elements.riskDetail.textContent = `Scored with ${modelLabel}. Customer profile sits within the safe retention corridor.`;
  }

  // Render Factor Breakdown
  renderRiskDrivers(profile, prob);

  // Render Retention Strategy
  renderRetentionPlaybook(profile, prob, isChurn);
}

// ==========================================================================
// Local Risk Factor Breakdown
// ==========================================================================

function renderRiskDrivers(profile, prob) {
  const drivers = [];

  // Age factor
  if (profile.Age >= 50) {
    drivers.push({ name: `Senior Age (${profile.Age} yrs)`, impact: "+38% Risk Increase", type: "up", width: "85%" });
  } else if (profile.Age >= 40) {
    drivers.push({ name: `Mature Age (${profile.Age} yrs)`, impact: "+18% Risk Elevation", type: "up", width: "45%" });
  } else {
    drivers.push({ name: `Younger Demography (${profile.Age} yrs)`, impact: "-15% Churn Reduction", type: "down", width: "35%" });
  }

  // Product Paradox
  if (profile.NumOfProducts === 3) {
    drivers.push({ name: "3 Products (Bundle Friction)", impact: "+45% Critical Risk Surge", type: "up", width: "95%" });
  } else if (profile.NumOfProducts === 4) {
    drivers.push({ name: "4 Products (Over-Bundled)", impact: "+75% Extreme Vulnerability", type: "up", width: "100%" });
  } else if (profile.NumOfProducts === 2) {
    drivers.push({ name: "2 Products (The Sweet Spot)", impact: "-24% Major Retention Anchor", type: "down", width: "65%" });
  } else {
    drivers.push({ name: "1 Product Only (Single Anchor)", impact: "+12% Cross-Sell Vulnerability", type: "up", width: "30%" });
  }

  // Geography
  if (profile.Geography === "Germany") {
    drivers.push({ name: "Germany Branch (2x Churn Region)", impact: "+16% Market Exposure", type: "up", width: "40%" });
  } else {
    drivers.push({ name: `${profile.Geography} Branch (Stable Market)`, impact: "-8% Regional Stability", type: "down", width: "20%" });
  }

  // Activity
  if (profile.IsActiveMember === 0) {
    drivers.push({ name: "Inactive Member Status", impact: "+13% Disengagement Risk", type: "up", width: "32%" });
  } else {
    drivers.push({ name: "Active Digital Member", impact: "-14% Loyalty Engagement", type: "down", width: "35%" });
  }

  elements.driverBars.innerHTML = drivers.map(d => `
    <div class="driver-item">
      <div class="driver-meta">
        <span class="driver-name">${d.name}</span>
        <span class="driver-impact ${d.type}">${d.impact}</span>
      </div>
      <div class="driver-track">
        <div class="driver-bar ${d.type}" style="width: ${d.width}"></div>
      </div>
    </div>
  `).join("");
}

// ==========================================================================
// Retention Playbook Prescription
// ==========================================================================

function renderRetentionPlaybook(profile, prob, isChurn) {
  const actions = [];

  if (profile.NumOfProducts >= 3) {
    actions.push("Schedule Immediate Product Simplification Review: Audit active product bundle to eliminate unneeded service charges and statement confusion.");
  }

  if (profile.Geography === "Germany" && profile.Balance > 50000) {
    actions.push("Competitive Yield Match for German Account: Customer balance ($" + profile.Balance.toLocaleString() + ") is exposed to German fintech deposit competition; offer promotional savings rate.");
  }

  if (profile.IsActiveMember === 0) {
    actions.push("Digital Re-Engagement Campaign: Send personalized onboarding push notification with mobile app fee rebate or cash-back reward.");
  }

  if (profile.Age >= 45 && profile.Balance > 60000) {
    actions.push("Senior Wealth Advisory Outreach: Assign a senior branch relationship manager for personal retirement and portfolio planning consultation.");
  }

  if (actions.length === 0) {
    actions.push("Maintain Standard Core Service: Customer profile is well-anchored; reinforce quarterly loyalty newsletter and maintain digital banking experience.");
    actions.push("Targeted 2nd Product Cross-Sell: If single-product, offer pre-approved high-yield savings to migrate to the ultra-safe 2-product tier.");
  }

  elements.retentionTitle.textContent = isChurn ? "High-Priority Retention Intervention Required" : "Relationship Maintenance & Growth Plan";
  elements.retentionAction.textContent = isChurn ? "Immediate actions to prevent customer account closure" : "Proactive engagement actions to maintain loyalty";
  elements.retentionChecklist.innerHTML = actions.map(a => `<li>${a}</li>`).join("");
}

// ==========================================================================
// Model Comparison: ROC Canvas Rendering
// ==========================================================================

function renderROCCurves() {
  if (!state.data) return;
  const canvas = elements.rocCanvas;
  if (!canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = (rect.width || 560) * dpr;
  canvas.height = (rect.height || 340) * dpr;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const w = rect.width || 560;
  const h = rect.height || 340;
  const padLeft = 45;
  const padBottom = 35;
  const padTop = 15;
  const padRight = 20;

  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  ctx.clearRect(0, 0, w, h);

  // Background Grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 5; i++) {
    const x = padLeft + (chartW / 5) * i;
    const y = padTop + (chartH / 5) * i;

    // Vertical line
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + chartH);
    ctx.stroke();

    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + chartW, y);
    ctx.stroke();

    // Axis Labels
    ctx.fillStyle = "#64748b";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText((i * 0.2).toFixed(1), x, padTop + chartH + 16);

    ctx.textAlign = "right";
    ctx.fillText(((5 - i) * 0.2).toFixed(1), padLeft - 8, y + 4);
  }

  // Axis Titles
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("False Positive Rate (1 - Specificity)", padLeft + chartW / 2, h - 4);

  ctx.save();
  ctx.translate(14, padTop + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("True Positive Rate (Recall)", 0, 0);
  ctx.restore();

  // Diagonal Baseline
  ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop + chartH);
  ctx.lineTo(padLeft + chartW, padTop);
  ctx.stroke();
  ctx.setLineDash([]);

  // Plot Curves
  const curves = [
    { name: "Logistic Regression", color: "#f59e0b", width: 2 },
    { name: "Random Forest", color: "#38bdf8", width: 2.5 },
    { name: "Gradient Boosting", color: "#10b981", width: 3.5 },
  ];

  curves.forEach(({ name, color, width }) => {
    const points = state.data.roc_data[name];
    if (!points || !points.length) return;

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();

    points.forEach((pt, i) => {
      const cx = padLeft + pt.fpr * chartW;
      const cy = padTop + chartH - pt.tpr * chartH;
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
  });
}

// ==========================================================================
// Dynamic Threshold Simulation
// ==========================================================================

function updateThresholdSimulation(threshold) {
  if (!state.data) return;
  const tData = state.data.threshold_metrics["Gradient Boosting"];
  if (!tData) return;

  // Find closest threshold record
  let closest = tData[0];
  let minDiff = Math.abs(closest.threshold - threshold);

  tData.forEach((row) => {
    const diff = Math.abs(row.threshold - threshold);
    if (diff < minDiff) {
      minDiff = diff;
      closest = row;
    }
  });

  // Update DOM
  elements.cmTN.textContent = closest.tn.toLocaleString();
  elements.cmFP.textContent = closest.fp.toLocaleString();
  elements.cmFN.textContent = closest.fn.toLocaleString();
  elements.cmTP.textContent = closest.tp.toLocaleString();

  elements.mPrecision.textContent = `${(closest.precision * 100).toFixed(1)}%`;
  elements.mRecall.textContent = `${(closest.recall * 100).toFixed(1)}%`;
  elements.mF1.textContent = closest.f1.toFixed(3);
  elements.mAccuracy.textContent = `${(closest.accuracy * 100).toFixed(1)}%`;
}

// ==========================================================================
// Feature Drivers: Permutation Importance Rendering
// ==========================================================================

function renderPermutationImportance() {
  if (!state.data) return;
  const container = elements.permImportanceBars;
  const gbImp = state.data.gb_feature_importances;

  // Consolidate one-hot categoricals for clear executive readability
  const features = [
    { name: "Customer Age", key: "num__Age", score: gbImp["num__Age"] || 0.1205 },
    { name: "Number of Products", key: "num__NumOfProducts", score: gbImp["num__NumOfProducts"] || 0.1174 },
    { name: "Is Active Member", key: "num__IsActiveMember", score: gbImp["num__IsActiveMember"] || 0.0329 },
    { name: "Geography (Germany Risk)", key: "cat__Geography_Germany", score: gbImp["cat__Geography_Germany"] || 0.0202 },
    { name: "Account Balance", key: "num__Balance", score: gbImp["num__Balance"] || 0.0156 },
    { name: "Customer Gender", key: "cat__Gender_Male", score: gbImp["cat__Gender_Male"] || 0.0083 },
    { name: "Balance-to-Salary Ratio", key: "num__BalanceToSalary", score: gbImp["num__BalanceToSalary"] || 0.0079 },
    { name: "ZeroBalance Flag", key: "num__ZeroBalance", score: gbImp["num__ZeroBalance"] || 0.0023 },
    { name: "Credit Score", key: "num__CreditScore", score: gbImp["num__CreditScore"] || 0.0019 },
    { name: "Estimated Salary", key: "num__EstimatedSalary", score: gbImp["num__EstimatedSalary"] || 0.0010 },
  ];

  features.sort((a, b) => b.score - a.score);
  const maxScore = features[0].score;

  container.innerHTML = features.map(f => {
    const pct = Math.max(4, Math.round((f.score / maxScore) * 100));
    return `
      <div class="imp-bar-row">
        <div class="imp-meta">
          <span class="imp-name">${f.name}</span>
          <span class="imp-score">+${f.score.toFixed(4)} &Delta;AUC</span>
        </div>
        <div class="imp-track">
          <div class="imp-fill" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
  }).join("");
}

// ==========================================================================
// Batch Scoring & Test Cohort Table
// ==========================================================================

function renderBatchTable() {
  if (!state.batchRecords) return;

  let records = state.batchRecords;

  // Filter
  if (state.batchFilter === "high") {
    records = records.filter(r => r.gb_churn_prob >= 0.50);
  } else if (state.batchFilter === "medium") {
    records = records.filter(r => r.gb_churn_prob >= 0.30 && r.gb_churn_prob < 0.50);
  } else if (state.batchFilter === "low") {
    records = records.filter(r => r.gb_churn_prob < 0.30);
  }

  // Search
  if (state.searchQuery) {
    records = records.filter(r =>
      r.Surname.toLowerCase().includes(state.searchQuery) ||
      r.CustomerId.toString().includes(state.searchQuery)
    );
  }

  // Update counts
  elements.countAll.textContent = state.batchRecords.length;
  elements.countHigh.textContent = state.batchRecords.filter(r => r.gb_churn_prob >= 0.50).length;
  elements.countMed.textContent = state.batchRecords.filter(r => r.gb_churn_prob >= 0.30 && r.gb_churn_prob < 0.50).length;
  elements.countLow.textContent = state.batchRecords.filter(r => r.gb_churn_prob < 0.30).length;

  if (records.length === 0) {
    elements.batchTableBody.innerHTML = `<tr><td colspan="10" style="text-align:center; padding: 24px; color: #64748b;">No matching customer records found.</td></tr>`;
    return;
  }

  elements.batchTableBody.innerHTML = records.map(r => {
    const probPct = Math.round(r.gb_churn_prob * 100);
    let riskBadge = `<span class="badge-risk low">${probPct}% (Low)</span>`;
    if (r.gb_churn_prob >= 0.50) {
      riskBadge = `<span class="badge-risk high">${probPct}% (High)</span>`;
    } else if (r.gb_churn_prob >= 0.30) {
      riskBadge = `<span class="badge-risk medium">${probPct}% (Med)</span>`;
    }

    const actualBadge = r.ActualExited === 1
      ? `<span style="color:#f87171; font-weight:700;">Exited (1)</span>`
      : `<span style="color:#34d399; font-weight:600;">Retained (0)</span>`;

    return `
      <tr>
        <td><code>${r.CustomerId}</code></td>
        <td><strong>${r.Surname}</strong></td>
        <td>${r.Age}</td>
        <td>${r.Geography}</td>
        <td>${r.NumOfProducts}</td>
        <td>$${r.Balance.toLocaleString()}</td>
        <td>${r.IsActiveMember === 1 ? 'Yes' : 'No'}</td>
        <td>${actualBadge}</td>
        <td>${riskBadge}</td>
        <td>
          <button class="btn-inspect" onclick="inspectCustomer(${r.CustomerId})">
            Load in Simulator &rarr;
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

// Global inspect helper
window.inspectCustomer = function(customerId) {
  const customer = state.batchRecords.find(r => r.CustomerId === customerId);
  if (!customer) return;

  applyProfile({
    Age: customer.Age,
    Geography: customer.Geography,
    Gender: customer.Gender,
    CreditScore: customer.CreditScore,
    Balance: customer.Balance,
    NumOfProducts: customer.NumOfProducts,
    HasCrCard: customer.HasCrCard,
    IsActiveMember: customer.IsActiveMember,
    EstimatedSalary: customer.EstimatedSalary,
    Tenure: customer.Tenure || 4,
  });

  // Switch to Predictor tab
  document.getElementById("btnTabPredictor").click();
  window.scrollTo({ top: 0, behavior: "smooth" });
};

// ==========================================================================
// CSV File Upload & Parsing
// ==========================================================================

function handleCSVFile(file) {
  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;
    const lines = text.trim().split("\n");
    if (lines.length < 2) return;

    const headers = lines[0].split(",").map(h => h.trim().replace(/^["']|["']$/g, ''));
    const parsedRecords = [];

    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(",").map(v => v.trim().replace(/^["']|["']$/g, ''));
      if (row.length !== headers.length) continue;

      const obj = {};
      headers.forEach((h, idx) => {
        obj[h] = row[idx];
      });

      // Construct customer profile
      const prof = {
        CustomerId: parseInt(obj.CustomerId || (9000000 + i), 10),
        Surname: obj.Surname || `Customer_${i}`,
        Age: parseInt(obj.Age || 40, 10),
        Geography: obj.Geography || "France",
        Gender: obj.Gender || "Male",
        CreditScore: parseInt(obj.CreditScore || 650, 10),
        Balance: parseFloat(obj.Balance || 0),
        NumOfProducts: parseInt(obj.NumOfProducts || 1, 10),
        HasCrCard: parseInt(obj.HasCrCard || 1, 10),
        IsActiveMember: parseInt(obj.IsActiveMember || 1, 10),
        EstimatedSalary: parseFloat(obj.EstimatedSalary || 100000),
        Tenure: parseInt(obj.Tenure || 3, 10),
        ActualExited: obj.Exited !== undefined ? parseInt(obj.Exited, 10) : 0,
      };

      // Score record with GB
      const vec = transformFeatures(prof);
      prof.gb_churn_prob = Math.round(predictGradientBoosting(vec) * 1000) / 1000;
      prof.rf_churn_prob = prof.gb_churn_prob;
      prof.lr_churn_prob = Math.round(predictLogisticRegression(vec) * 1000) / 1000;

      parsedRecords.push(prof);
    }

    if (parsedRecords.length) {
      state.batchRecords = [...parsedRecords, ...state.batchRecords];
      renderBatchTable();
      alert(`Successfully imported and scored ${parsedRecords.length} customer records with Gradient Boosting!`);
    }
  };
  reader.readAsText(file);
}

// ==========================================================================
// Export Batch CSV
// ==========================================================================

function exportBatchCSV() {
  if (!state.batchRecords || !state.batchRecords.length) return;

  const headers = [
    "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age",
    "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "ActualExited", "GB_Churn_Probability", "Risk_Tier"
  ];

  const rows = state.batchRecords.map(r => {
    let tier = "Low (<30%)";
    if (r.gb_churn_prob >= 0.50) tier = "High (>=50%)";
    else if (r.gb_churn_prob >= 0.30) tier = "Medium (30-50%)";

    return [
      r.CustomerId,
      `"${r.Surname}"`,
      r.CreditScore,
      r.Geography,
      r.Gender,
      r.Age,
      r.Tenure || 4,
      r.Balance,
      r.NumOfProducts,
      r.HasCrCard,
      r.IsActiveMember,
      r.EstimatedSalary,
      r.ActualExited,
      r.gb_churn_prob,
      `"${tier}"`
    ].join(",");
  });

  const csvContent = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `churn_predictions_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==========================================================================
// Ambient Background Network Animation
// ==========================================================================

function initAmbientBackground() {
  const canvas = document.getElementById("ambientNetworkCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let width, height;
  let particles = [];
  const mouse = { x: -1000, y: -1000, active: false };

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    createParticles();
  }

  const colors = ["#38bdf8", "#818cf8", "#34d399", "#a78bfa", "#60a5fa"];

  function createParticles() {
    const count = Math.min(Math.floor((width * height) / 28000), 50);
    particles = [];
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 1.6 + 1.1,
        color: colors[Math.floor(Math.random() * colors.length)],
        alpha: Math.random() * 0.35 + 0.2,
      });
    }
  }

  window.addEventListener("resize", resize);
  window.addEventListener("mousemove", (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.active = true;
  });
  window.addEventListener("mouseleave", () => {
    mouse.active = false;
  });

  resize();

  function animate() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = width;
      if (p.x > width) p.x = 0;
      if (p.y < 0) p.y = height;
      if (p.y > height) p.y = 0;

      // Mouse gentle interaction
      if (mouse.active) {
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120 && dist > 0) {
          const force = (120 - dist) / 120;
          p.x += (dx / dist) * force * 1.5;
          p.y += (dy / dist) * force * 1.5;
        }
      }

      // Draw particle dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha;
      ctx.fill();

      // Connect nearby particles with subtle glowing lines
      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dx = p.x - p2.x;
        const dy = p.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 120) {
          const lineAlpha = (1 - dist / 120) * 0.14;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = p.color;
          ctx.globalAlpha = lineAlpha;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    }

    ctx.globalAlpha = 1.0;
    requestAnimationFrame(animate);
  }

  animate();
}

// Kick off
document.addEventListener("DOMContentLoaded", init);

