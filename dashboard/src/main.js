import './style.css';
import Chart from 'chart.js/auto';

const app = document.querySelector('#app');

// State
let state = {
  currentRoute: 'overview',
  data: {}
};

// Layout Header/Sidebar
const renderLayout = () => {
  app.innerHTML = `
    <nav class="sidebar">
      <h1>RAG Analytics</h1>
      <ul class="nav-menu">
        <li class="nav-item ${state.currentRoute === 'overview' ? 'active' : ''}" data-route="overview">Overview</li>
        <li class="nav-item ${state.currentRoute === 'yelp' ? 'active' : ''}" data-route="yelp">Yelp Domain</li>
        <li class="nav-item ${state.currentRoute === 'legal' ? 'active' : ''}" data-route="legal">Legal Domain</li>
        <li class="nav-item ${state.currentRoute === 'comparison' ? 'active' : ''}" data-route="comparison">Comparison</li>
        <li class="nav-item ${state.currentRoute === 'classifier' ? 'active' : ''}" data-route="classifier">Classifier</li>
        <li class="nav-item ${state.currentRoute === 'retrieval' ? 'active' : ''}" data-route="retrieval">Retrieval</li>
      </ul>
    </nav>
    <main class="main-content" id="main-content">
      <div class="loader-container">
        <div class="spinner"></div>
        <p>Loading analytics data...</p>
      </div>
    </main>
  `;

  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', (e) => {
      const route = e.target.dataset.route;
      if (route !== state.currentRoute) {
        state.currentRoute = route;
        renderLayout();
        renderCurrentPage();
      }
    });
  });
};

// Fetch data
const loadData = async () => {
  try {
    const urls = {
      yelp_eval: '/data/yelp_evaluation_report.json',
      legal_eval: '/data/legal_evaluation_report.json',
      classifier: '/data/classifier_results.json',
      comparison: '/data/comparison_table.json',
      retrieval: '/data/regime_aware_evaluation.json'
    };

    for (const [key, url] of Object.entries(urls)) {
      const res = await fetch(url);
      if (res.ok) {
        state.data[key] = await res.json();
      } else {
        console.warn(`Could not load ${url}`);
      }
    }
  } catch (err) {
    console.error("Error loading data", err);
  }
};

// Chart helpers
let chartInstances = [];
const createChart = (canvasId, config) => {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';
  const chart = new Chart(ctx, config);
  chartInstances.push(chart);
  return chart;
};

const clearCharts = () => {
  chartInstances.forEach(c => c.destroy());
  chartInstances = [];
};

// Render Pages
const renderCurrentPage = () => {
  clearCharts();
  const main = document.getElementById('main-content');
  if (!main) return;

  if (state.currentRoute === 'overview') renderOverview(main);
  else if (state.currentRoute === 'comparison') renderComparison(main);
  else if (state.currentRoute === 'retrieval') renderRetrieval(main);
  else if (state.currentRoute === 'yelp') renderDomain(main, 'yelp');
  else if (state.currentRoute === 'legal') renderDomain(main, 'legal');
  else if (state.currentRoute === 'classifier') renderClassifier(main);
  else {
    main.innerHTML = `<div class="page-header"><h2 class="page-title">${state.currentRoute.toUpperCase()}</h2><p class="page-subtitle">Coming soon or currently unavailable.</p></div>`;
  }
};

const renderOverview = (main) => {
  const yelpCount = state.data.yelp_eval ? state.data.yelp_eval.length : 0;
  const legalCount = state.data.legal_eval ? state.data.legal_eval.length : 0;

  main.innerHTML = `
    <div class="page-header">
      <h2 class="page-title">Executive Overview</h2>
      <p class="page-subtitle">High-level KPIs for Query-Adaptive RAG across domains</p>
    </div>
    
    <div class="grid-4">
      <div class="glass-panel kpi-card">
        <div class="kpi-value">${yelpCount + legalCount}</div>
        <div class="kpi-label">Total Queries Analyzed</div>
      </div>
      <div class="glass-panel kpi-card">
        <div class="kpi-value">2</div>
        <div class="kpi-label">Domains</div>
      </div>
      <div class="glass-panel kpi-card">
        <div class="kpi-value">${state.data.comparison ? state.data.comparison.filter(c => c.significance.includes('*')).length : 0}</div>
        <div class="kpi-label">Significant Differences</div>
      </div>
    </div>
    
    <div class="grid-2">
      <div class="glass-panel">
        <h3>Distribution of Evaluated Queries</h3>
        <div class="chart-container large">
          <canvas id="overviewChart"></canvas>
        </div>
      </div>
      
      <div class="glass-panel" style="overflow-y: auto;">
        <h3>Theoretical Framework: Similarity Distributions</h3>
        <img src="/img/score_curves_overview.png" alt="Score Curves" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;" onerror="this.style.display='none'">
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
          The core thesis of this framework states that <strong>Similarity Score Distributions</strong> generated by vector databases like FAISS contain predictive signals regarding the optimal RAG retrieval strategy.
        </p>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">
          <strong>Entropy:</strong> Measures the "flatness" or uniformity of the distribution. High entropy suggests that many chunks share similarly low/moderate relevance (common in general domains), while low entropy signals a few chunks dominate.
        </p>
        <p style="color: var(--text-secondary);">
          <strong>Kurtosis:</strong> Measures the "peakedness." A high kurtosis indicates that relevance scores form a sharp spike, meaning a single document is a perfect semantic match. Together with <em>Score Range</em>, these signals indicate whether the RAG system should employ a Discriminative (low K) or Flat (high K) retrieval regime.
        </p>
      </div>
    </div>
  `;

  setTimeout(() => {
    createChart('overviewChart', {
      type: 'doughnut',
      data: {
        labels: ['Yelp (Consumer)', 'Legal (Statutes)'],
        datasets: [{
          data: [yelpCount, legalCount],
          backgroundColor: ['#FF6B6B', '#4ECDC4'],
          borderWidth: 0
        }]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right' } } }
    });
  }, 100);
};

const renderDomain = (main, domain) => {
  const dataList = domain === 'yelp' ? state.data.yelp_eval : state.data.legal_eval;
  if (!dataList) {
    main.innerHTML = `<p>No data found for ${domain}.</p>`;
    return;
  }
  
  const avgEntropy = dataList.reduce((acc, r) => acc + (r.entropy || 0), 0) / dataList.length;
  const maxSim = dataList.reduce((acc, r) => acc + (r.max_similarity || 0), 0) / dataList.length;
  
  let extraImages = '';
  if (domain === 'legal') {
    extraImages = `
      <div class="glass-panel">
        <h3>Feature by Intent Taxonomy</h3>
        <img src="/img/features_by_intent_legal.png" alt="Features by Intent" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;" onerror="this.style.display='none'">
        <img src="/img/taxonomy_legal.png" alt="Legal Taxonomy" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;" onerror="this.style.display='none'">
        <p style="color: var(--text-secondary); font-size: 0.95rem;">
          The plots above visualize how distinct intents inside the Legal domain cluster against similarity score taxonomy. Certain query intents inherently produce sharp, discriminative distributions, whereas open-ended intents (e.g., summaries) produce overlapping, ambiguous clusters.
        </p>
      </div>
    `;
  }

  main.innerHTML = `
    <div class="page-header">
      <h2 class="page-title" style="text-transform: capitalize;">${domain} Domain Analytics</h2>
      <p class="page-subtitle">Deep dive into ${domain} dataset characteristics</p>
    </div>
    <div class="grid-2">
      <div class="glass-panel kpi-card">
        <div class="kpi-value">${avgEntropy.toFixed(3)}</div>
        <div class="kpi-label">Avg Entropy (Flatness)</div>
      </div>
      <div class="glass-panel kpi-card">
        <div class="kpi-value">${maxSim.toFixed(3)}</div>
        <div class="kpi-label">Avg Max Similarity</div>
      </div>
    </div>
    <div class="grid-2">
      <div class="glass-panel">
        <h3>Similarity Distribution</h3>
        <div class="chart-container"><canvas id="simChart"></canvas></div>
      </div>
      <div class="glass-panel">
        <h3>Feature Radar</h3>
        <div class="chart-container"><canvas id="radarChart"></canvas></div>
      </div>
    </div>
    ${extraImages}
  `;

  setTimeout(() => {
    const simBins = Array(10).fill(0);
    dataList.forEach(item => {
      const sim = item.max_similarity;
      if (sim) {
        const bin = Math.min(9, Math.max(0, Math.floor(sim * 10)));
        simBins[bin]++;
      }
    });

    createChart('simChart', {
      type: 'bar',
      data: {
        labels: ['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1.0'],
        datasets: [{
          label: 'Queries',
          data: simBins,
          backgroundColor: domain === 'yelp' ? '#FF6B6B' : '#4ECDC4'
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });

    const avgScoreRange = dataList.reduce((acc, r) => acc + (r.score_range || 0), 0) / dataList.length;
    const avgKurtosis = dataList.reduce((acc, r) => acc + (r.kurtosis || 0), 0) / dataList.length;
    const avgDropR = dataList.reduce((acc, r) => acc + (r.drop_ratio_k5 || 0), 0) / dataList.length;

    createChart('radarChart', {
      type: 'radar',
      data: {
        labels: ['Max Similarity', 'Entropy', 'Score Range (x10)', 'Kurtosis', 'Drop Ratio @K5'],
        datasets: [{
          label: domain.toUpperCase(),
          data: [maxSim, avgEntropy / 5, avgScoreRange * 10, avgKurtosis, avgDropR],
          backgroundColor: domain === 'yelp' ? 'rgba(255,107,107,0.2)' : 'rgba(78,205,196,0.2)',
          borderColor: domain === 'yelp' ? '#FF6B6B' : '#4ECDC4',
          borderWidth: 2
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }, 100);
};

const renderComparison = (main) => {
  const compData = state.data.comparison || [];
  let tableRows = compData.map(row => `
    <tr>
      <td>${row.metric}</td>
      <td>${row.yelp_mean.toFixed(3)} ± ${row.yelp_std.toFixed(3)}</td>
      <td>${row.legal_mean.toFixed(3)} ± ${row.legal_std.toFixed(3)}</td>
      <td>${row.p_value}</td>
      <td><span class="badge" style="background: rgba(255,255,255,0.1)">${row.significance}</span></td>
    </tr>
  `).join('');

  main.innerHTML = `
    <div class="page-header">
      <h2 class="page-title">Cross-Domain Comparison</h2>
      <p class="page-subtitle">Statistically significant differences between Yelp and Legal texts</p>
    </div>
    
    <div class="glass-panel">
      <h3>How RAG Behaves Across Domains</h3>
      <div class="grid-2">
        <img src="/img/cross_domain_comparison.png" alt="Cross Domain Comparison" style="width: 100%; border-radius: 8px;" onerror="this.style.display='none'">
        <div>
          <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.95rem;">
            A foundational insight of our analytics is that <strong>domain structure dictates the RAG retrieval regime</strong>.
          </p>
          <ul style="color: var(--text-secondary); font-size: 0.95rem; margin-left: 1.5rem; margin-bottom: 1rem;">
            <li style="margin-bottom: 0.5rem"><strong>Legal Texts (Discriminative):</strong> Characterized by highly specific terminology ("statutes", "jurisdiction"). Because the semantic meaning is narrow, embedding models like BGE easily distinguish between the "correct" chunk and "incorrect" background noise. This leads to significantly higher peak <em>max_similarity</em> and <em>score_range</em>. The rapid drop-off implies we only need K=2 or K=3 chunks to supply context.</li>
            <li><strong>Yelp Consumer Reviews (Flat):</strong> Characterized by varying informal adjectives and semantic overlap. Different users use distinct vocabularies for similar experiences ("awesome food" vs "delicious meal"). This results in a flatter distribution, forcing the RAG system to employ a higher K (like K=5 or K=10) to pull ample context to guarantee an accurate generation.</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="glass-panel">
      <h3>Metric Comparison</h3>
      <div class="chart-container large" style="margin-bottom: 2rem">
        <canvas id="compChart"></canvas>
      </div>
      <div style="overflow-x:auto;">
        <table class="data-table">
          <thead>
            <tr><th>Metric</th><th>Yelp (μ±σ)</th><th>Legal (μ±σ)</th><th>P-Value</th><th>Significance</th></tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    </div>
  `;

  setTimeout(() => {
    const displayMetrics = compData.filter(d => ['max_similarity', 'mean_similarity', 'score_range', 'norm_confidence_k5'].includes(d.metric));
    createChart('compChart', {
      type: 'bar',
      data: {
        labels: displayMetrics.map(d => d.metric),
        datasets: [{
          label: 'Yelp',
          data: displayMetrics.map(d => d.yelp_mean),
          backgroundColor: '#FF6B6B'
        }, {
          label: 'Legal',
          data: displayMetrics.map(d => d.legal_mean),
          backgroundColor: '#4ECDC4'
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }, 100);
};

const renderClassifier = (main) => {
  const clData = state.data.classifier || {};
  main.innerHTML = `
    <div class="page-header">
      <h2 class="page-title">Regime Classifier Performance</h2>
      <p class="page-subtitle">Accuracy of NLP-based predictors prior to retrieval</p>
    </div>
    
    <div class="grid-2">
      <div class="glass-panel">
        <h3>Class Distribution</h3>
        <div class="chart-container"><canvas id="clsDist"></canvas></div>
      </div>
      <div class="glass-panel">
        <h3>Model Accuracy</h3>
        <div class="chart-container"><canvas id="clsAcc"></canvas></div>
      </div>
    </div>
  `;
  setTimeout(() => {
    if(!clData.cross_validation) return;
    const lr = clData.cross_validation.LogisticRegression;
    const rf = clData.cross_validation.RandomForest;
    createChart('clsAcc', {
      type: 'bar',
      data: {
        labels: ['Accuracy', 'F1 Score'],
        datasets: [
          { label: 'Logistic Regression', data: [lr.accuracy_mean, lr.f1_macro_mean], backgroundColor: '#a78bfa' },
          { label: 'Random Forest', data: [rf.accuracy_mean, rf.f1_macro_mean], backgroundColor: '#60a5fa' }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0 } } }
    });

    createChart('clsDist', {
      type: 'pie',
      data: {
        labels: ['Flat', 'Discriminative'],
        datasets: [{
          data: Object.values(clData.class_distribution || { 'flat': 50, 'discriminative': 50 }),
          backgroundColor: ['#eab308', '#10b981']
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }, 100);
};

const renderRetrieval = (main) => {
  const retData = state.data.retrieval || {};
  if (!retData.performance_comparison) return;

  const b_p = retData.performance_comparison.baseline.avg_precision_k5;
  const a_p = retData.performance_comparison.regime_aware.avg_precision_k5;
  
  const b_r = retData.performance_comparison.baseline.avg_recall_k5 || 0;
  const a_r = retData.performance_comparison.regime_aware.avg_recall_k5 || 0;
  
  const metrics = `
    <tr><td>Avg Precision</td><td>${b_p.toFixed(3)}</td><td style="color:var(--success)">${a_p.toFixed(3)}</td></tr>
    <tr><td>Avg Recall</td><td>${b_r.toFixed(3)}</td><td style="color:var(--success)">${a_r.toFixed(3)}</td></tr>
  `;

  main.innerHTML = `
    <div class="page-header">
      <h2 class="page-title">Regime-Aware Evaluation Analytics</h2>
      <p class="page-subtitle">Quantifying the impact of dynamic K optimization</p>
    </div>

    <div class="glass-panel">
      <h3>Theory of Adaptive K-Selection</h3>
      <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.95rem;">
        Baseline RAG typically operates on a "Fixed K" strategy (e.g., K=5 chunks), irrespective of query complexity. Our implementation introduces <strong>Regime-Aware Retrieval</strong>. By training an NLP classifier using pre-retrieval features (spaCy entities, query length, POS tags), we predict whether the query sits in a Discriminative or Flat regime. 
      </p>
      <p style="color: var(--text-secondary); font-size: 0.95rem;">
        For Discriminative queries, the system dynamically shifts to low K (K=2, K=3), minimizing the ingestion of lower-ranked background noise. For Flat queries, the system expands K (K=10). This directly optimizes precision without sacrificing recall since we adapt based on the predicted shape of the similarity distribution.
      </p>
    </div>
    
    <div class="grid-2">
      <div class="glass-panel">
        <h3>Performance Metrics</h3>
        <table class="data-table">
          <thead><tr><th>Metric</th><th>Baseline Fixed-K</th><th>Regime-Aware</th></tr></thead>
          <tbody>${metrics}</tbody>
        </table>
      </div>
      <div class="glass-panel">
        <h3>Improvement Margin</h3>
        <div class="chart-container"><canvas id="retChart"></canvas></div>
      </div>
    </div>
  `;
  setTimeout(() => {
    createChart('retChart', {
      type: 'bar',
      data: {
        labels: ['Precision', 'Recall'],
        datasets: [
          { label: 'Baseline', data: [b_p, b_r], backgroundColor: 'rgba(255, 255, 255, 0.2)' },
          { label: 'Regime Aware', data: [a_p, a_r], backgroundColor: '#10b981' }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }, 100);
};

// Initialize
const init = async () => {
  renderLayout();
  await loadData();
  renderCurrentPage();
};

init();
