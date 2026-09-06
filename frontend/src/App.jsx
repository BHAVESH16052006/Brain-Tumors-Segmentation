import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const MODALITIES = [
  {
    id: "t1n",
    title: "T1 Native",
    description: "T1-weighted native MRI",
  },
  {
    id: "t1c",
    title: "T1 Contrast",
    description: "T1-weighted contrast enhanced MRI",
  },
  {
    id: "t2w",
    title: "T2 Weighted",
    description: "T2-weighted MRI",
  },
  {
    id: "t2f",
    title: "T2 FLAIR",
    description: "T2 FLAIR MRI",
  },
];

function App() {
  const [files, setFiles] = useState({
    t1n: null,
    t1c: null,
    t2w: null,
    t2f: null,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [activeModality, setActiveModality] =
    useState("t1n");

  const [slice, setSlice] = useState(0);

  const [fullscreen, setFullscreen] =
    useState(false);

  // Additive UI controls only: no model/API behavior is changed.
  const [compareMode, setCompareMode] = useState(false);
  const [viewMode, setViewMode] = useState("segmentation");

  // All four BraTS MRI modalities are required before inference.
  const allFilesUploaded = Boolean(
    files.t1n && files.t1c && files.t2w && files.t2f
  );

  const currentModality =
    MODALITIES.find(
      (item) =>
        item.id === activeModality
    );

  const totalSlices =
    Number(result?.num_slices) ||
    Number(
      result?.dimensions?.slices
    ) ||
    155;

  const mriUrl = useMemo(() => {
    if (!result?.case_id) {
      return "";
    }

    return (
      `${API_BASE}/case/${result.case_id}/slice` +
      `?modality=${activeModality}` +
      `&slice=${slice}` +
      `&overlay=false`
    );
  }, [
    result?.case_id,
    activeModality,
    slice,
  ]);

  const overlayUrl = useMemo(() => {
    if (!result?.case_id) {
      return "";
    }

    return (
      `${API_BASE}/case/${result.case_id}/slice` +
      `?modality=${activeModality}` +
      `&slice=${slice}` +
      `&overlay=true`
    );
  }, [
    result?.case_id,
    activeModality,
    slice,
  ]);
  function handleFileChange(
    modality,
    event
  ) {
    const selectedFile =
      event.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    const fileName =
      selectedFile.name.toLowerCase();

    const valid =
      fileName.endsWith(".nii") ||
      fileName.endsWith(".nii.gz");

    if (!valid) {
      setError(
        "Please upload a valid NIfTI file (.nii or .nii.gz)."
      );

      return;
    }

    setError("");

    setFiles((previous) => ({
      ...previous,
      [modality]: selectedFile,
    }));
  }

  function removeFile(modality) {
    setFiles((previous) => ({
      ...previous,
      [modality]: null,
    }));
  }

  async function analyzeMRI() {
    if (!allFilesUploaded) {
      setError(
        "Please upload all four MRI modalities before analysis."
      );

      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData =
      new FormData();

    formData.append(
      "t1n",
      files.t1n
    );

    formData.append(
      "t1c",
      files.t1c
    );

    formData.append(
      "t2w",
      files.t2w
    );

    formData.append(
      "t2f",
      files.t2f
    );

    try {
      const response =
        await fetch(
          `${API_BASE}/predict`,
          {
            method: "POST",
            body: formData,
          }
        );

      const data =
        await response.json();

      if (
        !response.ok ||
        !data.success
      ) {
        throw new Error(
          data.error ||
          "MRI analysis failed."
        );
      }

      const numberOfSlices =
        Number(data.num_slices) ||
        Number(
          data.dimensions?.slices
        ) ||
        155;

      let initialSlice =
        Number(data.initial_slice);

      if (
        !Number.isInteger(
          initialSlice
        )
      ) {
        initialSlice =
          Math.floor(
            numberOfSlices / 2
          );
      }

      initialSlice =
        Math.max(
          0,
          Math.min(
            initialSlice,
            numberOfSlices - 1
          )
        );

      // ----------------------------------------------------
      // Statistics
      // ----------------------------------------------------

      const totalTumor =
        Number(
          data.tumor_voxels
        ) || 0;

      const edema =
        Number(
          data.edema_voxels
        ) || 0;

      const enhancingTumor =
        Number(
          data.enhancing_tumor_voxels
        ) || 0;

      let ncrNet =
        Number(
          data.ncr_net_voxels
        );

      if (
        !Number.isFinite(ncrNet) ||
        (
          ncrNet === 0 &&
          totalTumor > 0 &&
          edema > 0 &&
          enhancingTumor > 0
        )
      ) {
        ncrNet =
          Math.max(
            0,
            totalTumor -
            edema -
            enhancingTumor
          );
      }

      // ----------------------------------------------------
      // Volume values
      //
      // Backend calculates these using actual NIfTI spacing.
      // ----------------------------------------------------

      const tumorVolume =
        Number(
          data.tumor_volume_cm3
        );

      const edemaVolume =
        Number(
          data.edema_volume_cm3
        );

      const ncrNetVolume =
        Number(
          data.ncr_net_volume_cm3
        );

      const enhancingTumorVolume =
        Number(
          data.enhancing_tumor_volume_cm3
        );

      const normalizedResult = {
        ...data,

        tumor_voxels:
          totalTumor,

        edema_voxels:
          edema,

        ncr_net_voxels:
          ncrNet,

        enhancing_tumor_voxels:
          enhancingTumor,

        tumor_volume_cm3:
          Number.isFinite(
            tumorVolume
          )
            ? tumorVolume
            : 0,

        edema_volume_cm3:
          Number.isFinite(
            edemaVolume
          )
            ? edemaVolume
            : 0,

        ncr_net_volume_cm3:
          Number.isFinite(
            ncrNetVolume
          )
            ? ncrNetVolume
            : 0,

        enhancing_tumor_volume_cm3:
          Number.isFinite(
            enhancingTumorVolume
          )
            ? enhancingTumorVolume
            : 0,

        num_slices:
          numberOfSlices,

        initial_slice:
          initialSlice,

        dimensions: {
          ...(data.dimensions || {}),
          slices:
            numberOfSlices,
        },
      };

      setResult(
        normalizedResult
      );

      setSlice(
        initialSlice
      );

      setActiveModality(
        "t1n"
      );
      setCompareMode(false);
      setViewMode("segmentation");

      window.setTimeout(() => {
        const element =
          document.getElementById(
            "analysis-results"
          );

        if (element) {
          element.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }
      }, 150);

    } catch (err) {
      console.error(
        "MRI analysis error:",
        err
      );

      setError(
        err.message ||
        "Unable to connect to the Brain Tumor Segmentation API."
      );

    } finally {
      setLoading(false);
    }
  }

  function newAnalysis() {
    setFiles({
      t1n: null,
      t1c: null,
      t2w: null,
      t2f: null,
    });

    setResult(null);
    setError("");
    setSlice(0);
    setActiveModality("t1n");
    setFullscreen(false);
    setCompareMode(false);
    setViewMode("segmentation");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }

  function previousSlice() {
    setSlice((previous) =>
      Math.max(
        0,
        previous - 1
      )
    );
  }

  function nextSlice() {
    setSlice((previous) =>
      Math.min(
        totalSlices - 1,
        previous + 1
      )
    );
  }

  function handleSliceChange(
    event
  ) {
    setSlice(
      Number(
        event.target.value
      )
    );
  }

  function toggleFullscreen() {
    setFullscreen(
      (previous) => !previous
    );
  }

  function formatNumber(value) {
    return Number(
      value || 0
    ).toLocaleString();
  }

  function formatVolume(value) {
    return Number(
      value || 0
    ).toFixed(2);
  }

  function formatPercent(value) {
    return Number(value || 0).toFixed(1);
  }

  function getBurdenWidth(value) {
    const values = result?.slice_burden_percent || [];
    const maxValue = Math.max(...values, 0.01);
    return Math.max(2, Math.min(100, (Number(value || 0) / maxValue) * 100));
  }

  return (
    <div className={fullscreen ? "app fullscreen-view" : "app"}>
      <header className="hospital-header">
        <div className="hospital-brand">
          <div className="hospital-logo">✚</div>
          <div>
            <div className="hospital-title">NeuroScan AI</div>
            <div className="hospital-subtitle">Brain Tumor Detection &amp; Segmentation</div>
          </div>
        </div>
        <div className="department-block">
          <strong>Department of AI &amp; Medical Imaging</strong>
          <span>Precision&nbsp; | &nbsp;Analysis&nbsp; | &nbsp;Better Care</span>
        </div>
        <div className="profile-icon">♙</div>
      </header>

      <div className="hospital-shell">
        <aside className="sidebar">
          <button className="side-nav active" type="button"><span>⌂</span>Analyze</button>
          <button className="side-nav" type="button"><span>▣</span>Results</button>
          <button className="side-nav" type="button"><span>▤</span>Report</button>
          <button className="side-nav" type="button"><span>◷</span>History</button>
          <button className="side-nav" type="button"><span>⚙</span>Settings</button>
          <div className="sidebar-brand">
            <div className="sidebar-brain">♧</div>
            <strong>NeuroScan AI</strong>
            <span>Multimodal MRI Analysis</span>
            <em>AI for a<br />Healthier Tomorrow</em>
          </div>
        </aside>

        <main className="clinical-main" id="analysis-results">
          {!result && !loading && (
            <section className="welcome-panel">
              <div className="welcome-kicker">AI-POWERED MRI ANALYSIS</div>
              <h1>Brain Tumor Detection &amp; Segmentation</h1>
              <p>Analyze four multimodal MRI sequences using your trained 3D SegResNet model and inspect the predicted tumor regions slice by slice.</p>
            </section>
          )}

          <section className="upload-panel">
            <div className="panel-heading">
              <div>
                <h2><span className="heading-icon">✚</span> Upload MRI Scans <small>(NIfTI .nii.gz)</small></h2>
                <p>Provide all four required BraTS MRI modalities for SegResNet analysis.</p>
              </div>
              <div className="format-badge">NIfTI&nbsp; .nii.gz</div>
            </div>

            <div className="clinical-upload-grid">
              {MODALITIES.map((modality) => {
                const file = files[modality.id];
                return (
                  <div key={modality.id} className={file ? "clinical-upload-card selected" : "clinical-upload-card"}>
                    <input id={`file-${modality.id}`} type="file" accept=".nii,.nii.gz" onChange={(event) => handleFileChange(modality.id, event)} />
                    <label htmlFor={`file-${modality.id}`} className="clinical-upload-label">
                      <span className="upload-cloud">⇧</span>
                      <span className="upload-modality">{modality.title}</span>
                      <span className="upload-file-name">{file ? file.name : "Upload File"}</span>
                    </label>
                    {file && <button type="button" className="clinical-remove" onClick={() => removeFile(modality.id)} aria-label={`Remove ${modality.title}`}>×</button>}
                  </div>
                );
              })}
              <button type="button" className="run-analysis-button" onClick={analyzeMRI} disabled={loading}>
                <span>▶</span> {loading ? "Analyzing..." : "Run Analysis"}
              </button>
            </div>
          </section>

          {error && <div className="clinical-error">⚠ {error}</div>}

          {loading && (
            <section className="clinical-loading">
              <div className="loading-brain">🧠</div>
              <h2>Analyzing MRI</h2>
              <p>Your four MRI volumes are being processed by the trained 3D SegResNet model.</p>
              <div className="loading-progress"><div className="loading-progress-bar"></div></div>
              <div className="loading-steps"><span>✓ MRI upload</span><span>• Preprocessing</span><span>• SegResNet inference</span><span>• Segmentation</span></div>
            </section>
          )}

          {result && !loading && (
            <>
              <section className="clinical-dashboard-grid">
                <section className="viewer-card clinical-card">
                  <div className="clinical-card-title"><span>▣</span> MRI Slice View</div>
                  <div className="axial-label">Axial View (Slice {slice + 1} / {totalSlices})</div>
                  <div className="viewer-mode-toggle">
                    <button type="button" className={viewMode === "original" ? "active" : ""} onClick={() => setViewMode("original")}>Original MRI</button>
                    <button type="button" className={viewMode === "segmentation" ? "active" : ""} onClick={() => setViewMode("segmentation")}>Segmentation</button>
                  </div>
                  <div className="clinical-image-wrap">
                    <img src={viewMode === "original" ? mriUrl : overlayUrl} alt={viewMode === "original" ? "Original MRI" : "MRI segmentation overlay"} />
                    <button type="button" className="image-arrow left" onClick={previousSlice} disabled={slice === 0}>‹</button>
                    <button type="button" className="image-arrow right" onClick={nextSlice} disabled={slice === totalSlices - 1}>›</button>
                  </div>
                  <div className="clinical-slider-row">
                    <input type="range" min="0" max={Math.max(0, totalSlices - 1)} value={Math.min(slice, totalSlices - 1)} onChange={handleSliceChange} className="clinical-slider" />
                    <span>Slice {slice + 1} / {totalSlices}</span>
                  </div>
                  <div className="slice-minimap" aria-label="Tumor-containing slice mini-map">
                    <div className="slice-minimap-label"><span>Slice Map</span><strong>{(result.slice_burden_percent || []).filter((v) => Number(v) > 0).length} tumor-positive slices</strong></div>
                    <div className="slice-minimap-track">
                      {(result.slice_burden_percent || []).map((value, index) => (
                        <button type="button" key={index} title={`Slice ${index + 1}: ${formatPercent(value)}% burden`} className={index === slice ? "mini-slice active" : Number(value) > 0 ? "mini-slice positive" : "mini-slice"} onClick={() => setSlice(index)}><span style={{height:`${getBurdenWidth(value)}%`}}></span></button>
                      ))}
                    </div>
                  </div>
                  <div className="clinical-thumbnails">
                    {MODALITIES.map((modality) => (
                      <button key={modality.id} type="button" className={activeModality === modality.id ? "clinical-thumb active" : "clinical-thumb"} onClick={() => setActiveModality(modality.id)}>
                        <span>{modality.id.toUpperCase()}</span>
                        <img src={`${API_BASE}/case/${result.case_id}/slice?modality=${modality.id}&slice=${slice}&overlay=false`} alt={modality.title} />
                      </button>
                    ))}
                    <button type="button" className="clinical-thumb segmentation-thumb active">
                      <span>Segmentation</span>
                      <img src={overlayUrl} alt="Segmentation" />
                    </button>
                  </div>
                  <div className="viewer-footer-controls">
                    <div className="legend">
                      <span><i className="legend-dot edema"></i>Edema</span>
                      <span><i className="legend-dot ncr"></i>NCR / NET</span>
                      <span><i className="legend-dot enhancing"></i>Enhancing Tumor</span>
                    </div>
                    <div className="viewer-action-group">
                      <button type="button" className={compareMode ? "compare-button active" : "compare-button"} onClick={() => setCompareMode((previous) => !previous)}>▦ {compareMode ? "Hide Compare" : "Compare MRI"}</button>
                      <button type="button" className="fullscreen-button" onClick={toggleFullscreen}>⛶ {fullscreen ? "Exit" : "Fullscreen"}</button>
                    </div>
                  </div>
                </section>

                <section className="analysis-card clinical-card">
                  <div className="clinical-card-title"><span>▥</span> Tumor Analysis</div>
                  <div className="total-volume-row"><span>Total Tumor Volume</span><strong>{formatVolume(result.tumor_volume_cm3)} cm³</strong></div>
                  <div className="metric-list">
                    <div className="metric-row"><span className="metric-name"><i className="metric-dot enhancing"></i>Enhancing Tumor</span><strong>{formatVolume(result.enhancing_tumor_volume_cm3)} cm³</strong><em>({formatPercent(result.enhancing_tumor_percentage)}%)</em></div>
                    <div className="metric-row"><span className="metric-name"><i className="metric-dot ncr"></i>Tumor Core</span><strong>{formatVolume((Number(result.ncr_net_volume_cm3) || 0) + (Number(result.enhancing_tumor_volume_cm3) || 0))} cm³</strong><em>({formatPercent((Number(result.ncr_net_percentage) || 0) + (Number(result.enhancing_tumor_percentage) || 0))}%)</em></div>
                    <div className="metric-row"><span className="metric-name"><i className="metric-dot edema"></i>Edema</span><strong>{formatVolume(result.edema_volume_cm3)} cm³</strong><em>({formatPercent(result.edema_percentage)}%)</em></div>
                  </div>
                  <div className="composition-bars">
                    <div><span>Enhancing Tumor</span><b style={{width:`${Math.min(100, Number(result.enhancing_tumor_percentage || 0))}%`}}></b></div>
                    <div><span>Tumor Core</span><b style={{width:`${Math.min(100, Number(result.ncr_net_percentage || 0) + Number(result.enhancing_tumor_percentage || 0))}%`}}></b></div>
                    <div><span>Edema</span><b style={{width:`${Math.min(100, Number(result.edema_percentage || 0))}%`}}></b></div>
                  </div>
                </section>

                <section className="prediction-card clinical-card">
                  <div className="clinical-card-title"><span>♧</span> AI Prediction</div>
                  <div className={result.tumor_detected ? "prediction-status detected" : "prediction-status clear"}>
                    <span>✓</span>{result.tumor_detected ? "Tumor Detected" : "No Tumor Detected"}
                  </div>
                  <div className="confidence-line">Confidence: <strong>{formatPercent(result.tumor_confidence)}% ({result.confidence_level || "LOW"})</strong></div>
                  <div className="burden-line">Tumor Burden: <strong>{result.tumor_burden_category || "No Tumor"}</strong></div>
                  <div className="clinical-disclaimer">⚠ <i>AI-derived analysis for research use only.<br />Not a clinical diagnosis.</i></div>
                </section>
              </section>

              {compareMode && (
                <section className="clinical-card compare-panel">
                  <div className="clinical-card-title"><span>▦</span> Multimodal MRI Comparison</div>
                  <div className="compare-subtitle">All four MRI sequences are synchronized to Slice {slice + 1} / {totalSlices}.</div>
                  <div className="compare-grid">
                    {MODALITIES.map((modality) => (
                      <button key={modality.id} type="button" className="compare-image-card" onClick={() => { setActiveModality(modality.id); setCompareMode(false); }}>
                        <div className="compare-image-title"><strong>{modality.title}</strong><span>{modality.id.toUpperCase()}</span></div>
                        <img src={`${API_BASE}/case/${result.case_id}/slice?modality=${modality.id}&slice=${slice}&overlay=false`} alt={`${modality.title} comparison`} />
                        <small>Click to focus</small>
                      </button>
                    ))}
                  </div>
                </section>
              )}

              <section className="insights-dashboard-grid">
                <section className="clinical-card composition-insight">
                  <div className="clinical-card-title"><span>◔</span> Tumor Composition</div>
                  <div className="composition-insight-layout">
                    <div className="composition-legend-list">
                      <div><i className="metric-dot enhancing"></i><span>Enhancing Tumor</span><strong>{formatPercent(result.enhancing_tumor_percentage)}%</strong></div>
                      <div><i className="metric-dot ncr"></i><span>Tumor Core</span><strong>{formatPercent((Number(result.ncr_net_percentage)||0)+(Number(result.enhancing_tumor_percentage)||0))}%</strong></div>
                      <div><i className="metric-dot edema"></i><span>Edema</span><strong>{formatPercent(result.edema_percentage)}%</strong></div>
                    </div>
                    <div className="composition-ring" style={{"--et": `${Number(result.enhancing_tumor_percentage || 0)}%`, "--core": `${Number(result.ncr_net_percentage || 0)}%`, "--edema": `${Number(result.edema_percentage || 0)}%`}}><span>{formatVolume(result.tumor_volume_cm3)}<small>cm³</small></span></div>
                  </div>
                </section>

                <section className="clinical-card heatmap-clinical">
                  <div className="clinical-card-title"><span>▥</span> Tumor Burden (Slice-wise)</div>
                  <div className="heatmap-chart">
                    <div className="heatmap-y-labels"><span>25%</span><span>20%</span><span>15%</span><span>10%</span><span>5%</span><span>0%</span></div>
                    <div className="heatmap-bars">
                      {(result.slice_burden_percent || []).map((value, index) => (
                        <button type="button" key={index} className={index === slice ? "clinical-heat-column active" : "clinical-heat-column"} title={`Slice ${index + 1}: ${formatPercent(value)}%`} onClick={() => setSlice(index)}>
                          <span style={{height:`${getBurdenWidth(value)}%`}}></span>
                        </button>
                      ))}
                      <div className="peak-marker" style={{left:`${Math.min(99, Math.max(0, Number(result.max_burden_slice || 0) / Math.max(1, totalSlices - 1) * 100))}%`}}>Peak: {formatPercent(result.max_slice_burden_percent)}%<br />Slice: {Number(result.max_burden_slice || 0) + 1}</div>
                    </div>
                  </div>
                  <div className="chart-axis"><span>1</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span><span>120</span><span>140</span><span>{totalSlices}</span></div>
                </section>

                <section className="clinical-card burden-clinical">
                  <div className="clinical-card-title"><span>▱</span> AI-derived Tumor Burden Category</div>
                  <div className="burden-category-value">{result.tumor_burden_category || "No Tumor"}</div>
                  <p>Tumor volume is in the higher range based on AI analysis.</p>
                  <div className="burden-scale-clinical"><span>🟢 Low &lt; 20 cm³</span><span>🟡 Moderate 20 – 60 cm³</span><span>🟠 High 60 – 100 cm³</span><span>🔴 Very High &gt; 100 cm³</span></div>
                  <small>Volume-based research visualization only; not a clinical severity grade or cancer stage.</small>
                </section>

                <section className="clinical-card report-clinical">
                  <div className="clinical-card-title"><span>▤</span> Report</div>
                  <p>Generate the detailed case report for the current patient and selected slice.</p>
                  <div className="report-buttons">
                    <button type="button" className="view-report-button" onClick={() => { if (!result?.case_id) return; window.open(`${API_BASE}/case/${result.case_id}/report?slice=${slice}`, "_blank"); }}>▤ View Detailed Report</button>
                    <button type="button" className="download-report-button" onClick={() => { if (!result?.case_id) return; window.open(`${API_BASE}/case/${result.case_id}/report?slice=${slice}`, "_blank"); }}>⇩ Download PDF Report</button>
                  </div>
                </section>
              </section>

              <section className="summary-dashboard-grid">
                <section className="clinical-card ai-summary-card">
                  <div className="clinical-card-title"><span>✓</span> AI Analysis Summary</div>
                  <div className="summary-status-line">
                    <span className={result.tumor_detected ? "summary-check detected" : "summary-check clear"}>{result.tumor_detected ? "✓" : "—"}</span>
                    <div><strong>{result.tumor_detected ? "Tumor detected" : "No tumor detected"}</strong><small>Automated multimodal MRI segmentation result</small></div>
                  </div>
                  <div className="summary-stat-grid">
                    <div><span>Tumor Volume</span><strong>{formatVolume(result.tumor_volume_cm3)} cm³</strong></div>
                    <div><span>Confidence</span><strong>{formatPercent(result.tumor_confidence)}%</strong></div>
                    <div><span>Tumor Burden</span><strong>{result.tumor_burden_category || "No Tumor"}</strong></div>
                    <div><span>Positive Slices</span><strong>{(result.slice_burden_percent || []).filter((v) => Number(v) > 0).length} / {totalSlices}</strong></div>
                  </div>
                </section>

                <section className="clinical-card case-summary-card">
                  <div className="clinical-card-title"><span>▤</span> Case Analysis Summary</div>
                  <div className="case-summary-grid">
                    <div><span>Case ID</span><strong>{result.case_id}</strong></div>
                    <div><span>Sequences</span><strong>4 / 4</strong></div>
                    <div><span>Model</span><strong>3D SegResNet</strong></div>
                    <div><span>Input Format</span><strong>NIfTI (.nii.gz)</strong></div>
                    <div><span>Analysis Status</span><strong className="case-complete">✓ Complete</strong></div>
                    <div><span>Current Slice</span><strong>{slice + 1} / {totalSlices}</strong></div>
                  </div>
                </section>
              </section>

              <div className="clinical-bottom-actions">
                <span>Case ID: <strong>{result.case_id}</strong></span>
                <button type="button" className="new-analysis-button" onClick={newAnalysis}>＋ New Analysis</button>
              </div>
            </>
          )}

          {!result && !loading && (
            <section className="clinical-empty-note">
              <strong>Workflow</strong>
              <span>Upload T1N, T1C, T2W and T2F → Run Analysis → Review segmentation → Download report.</span>
            </section>
          )}

          <footer className="clinical-footer">
            <span>NeuroScan AI&nbsp; | &nbsp;Advanced Multimodal MRI Analysis</span>
            <span>© 2026 NeuroScan AI. All rights reserved.</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

export default App;
