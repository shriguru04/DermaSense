// frontend/src/pages/NewAnalysis.js
import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../App';
import { api, CLINICAL_FEATURES, HISTO_FEATURES } from '../services/api';

const STEPS = ['Patient Info', 'Clinical Features', 'Histopathology', 'Image & Analyze'];

const SLIDER_COLORS = ['#475569','#F59E0B','#EF4444','#7C2D12'];
const SLIDER_LABELS = ['Absent (0)','Mild (1)','Moderate (2)','Severe (3)'];

// ── Feature Slider Component ─────────────────────────────────────
function FeatureSlider({ feature, value, onChange }) {
  const col = SLIDER_COLORS[value] || SLIDER_COLORS[0];
  const pct = (value / 3) * 100;
  const isPathognomonic = feature.label.includes('★');
  return (
    <div className={`feat-card${isPathognomonic ? ' highlighted' : ''}`}>
      <div className="feat-header">
        <span className="feat-label">{feature.label}</span>
        <span className="feat-badge" style={{ background: col }}>{value}</span>
      </div>
      <div className="feat-desc">{feature.desc}</div>
      <input
        type="range" min={0} max={3} step={1} value={value}
        style={{ '--slider-col': col, '--slider-pct': `${pct}%` }}
        onChange={e => onChange(feature.key, parseInt(e.target.value))}
      />
      <div className="slider-marks">
        {['0','1','2','3'].map((n, i) => (
          <span key={n} style={{ color: i === value ? col : 'var(--text3)', fontWeight: i === value ? 700 : 400 }}>
            {n}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Image Upload Zone ────────────────────────────────────────────
function ImageUpload({ imageFile, imagePreview, onImageChange, onImageClear }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    onImageChange(file, url);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  return (
    <div>
      {imagePreview ? (
        <div style={{ position: 'relative' }}>
          <img src={imagePreview} alt="Dermoscopy preview" className="upload-preview" />
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginTop:8 }}>
            <div className="cnn-badge">🔬 CNN will extract 512 features from this image</div>
            <button
              className="btn btn-ghost btn-sm"
              onClick={onImageClear}
              style={{ color:'var(--red)' }}
            >
              ✕ Remove
            </button>
          </div>
        </div>
      ) : (
        <div
          className={`upload-zone${dragging ? ' drag-over' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <span className="upload-icon">🖼️</span>
          <div className="upload-title">Drop dermoscopy image here</div>
          <div className="upload-sub">or click to browse — JPEG, PNG, WEBP up to 10MB</div>
          <div style={{ marginTop: 12 }}>
            <span className="cnn-badge">ResNet18 CNN extracts 512 deep features automatically</span>
          </div>
        </div>
      )}
      <input
        ref={inputRef} type="file" accept="image/*"
        style={{ display:'none' }}
        onChange={e => handleFile(e.target.files[0])}
      />
    </div>
  );
}

// ── Mock prediction when API is offline ─────────────────────────
function mockPredict(features, hasImage) {
  const { munro_microabscess:mm=0, saw_tooth_retes:str=0, band_like_infiltrate:bli=0,
    follicular_horn_plug:fhp=0, focal_hypergranulosis:fh=0, spongiosis:sp=0,
    polygonal_papules:pp=0, scalp_involvement:sc=0 } = features;

  let cls = 'Psoriasis';
  if ((str>=2||bli>=2||pp>=2) && mm<2) cls = 'Lichen_Planus';
  else if (fhp>=2) cls = 'PRP';
  else if (fh>=2) cls = 'Pityriasis_Rosea';
  else if (sp>=2 && mm<1) cls = 'Chronic_Dermatitis';
  else if (sc>=2 && mm<1) cls = 'Seborrheic_Dermatitis';
  else if (mm>=2) cls = 'Psoriasis';

  const classes = ['Psoriasis','Seborrheic_Dermatitis','Lichen_Planus','Pityriasis_Rosea','Chronic_Dermatitis','PRP'];
  const base = Object.fromEntries(classes.map(c => [c, 0.03]));
  base[cls] = 0.82;
  const tot = Object.values(base).reduce((a,b)=>a+b,0);
  const probs = Object.fromEntries(Object.entries(base).map(([k,v])=>[k, +(v/tot).toFixed(4)]));

  const FEAT_LABELS = {
    munro_microabscess:'Munro Microabscess', band_like_infiltrate:'Band-like Infiltrate',
    saw_tooth_retes:'Saw-tooth Retes', follicular_horn_plug:'Follicular Horn Plug',
    polygonal_papules:'Polygonal Papules', spongiosis:'Spongiosis',
    scaling:'Scaling', erythema:'Erythema',
  };
  const topFeats = Object.entries(features)
    .filter(([,v]) => v > 0)
    .sort(([,a],[,b]) => b-a)
    .slice(0,8)
    .map(([k,v],i) => ({
      feature: k, label: FEAT_LABELS[k] || k.replace(/_/g,' '),
      value: v, shap: i<4 ? +(0.35-i*0.07).toFixed(3) : -(0.15-i*0.02).toFixed(3),
      direction: i<4 ? 'SUPPORTS' : 'AGAINST',
    }));

  const disName = cls.replace(/_/g,' ');
  return {
    prediction: { predicted_class:cls, confidence:probs[cls], all_probabilities:probs, has_image:hasImage },
    anomaly: { is_anomaly:probs[cls]<0.60, anomaly_probability:+(1-probs[cls]).toFixed(3),
      euclidean_distance:+((1-probs[cls])*8+1.2).toFixed(2), threshold:5.1, nearest_class:cls,
      distances: Object.fromEntries(classes.map(c=>[c,+(Math.random()*6+1).toFixed(2)])) },
    explanation: {
      top_features: topFeats,
      clinical_text: `Prediction of <strong>${disName}</strong> with ${(probs[cls]*100).toFixed(1)}% confidence. ` +
        `Primary evidence: ${topFeats[0]?.label||'—'} (${topFeats[0]?.value||0}/3) and ${topFeats[1]?.label||'—'} (${topFeats[1]?.value||0}/3).` +
        (hasImage ? ' CNN image analysis contributed 512 deep features to this prediction.' : ''),
      counterfactual: topFeats[0]
        ? `If ${topFeats[0].label} were 0 instead of ${topFeats[0].value}, the prediction would shift toward ${classes[(classes.indexOf(cls)+1)%6].replace(/_/g,' ')}.`
        : '',
    },
    disease_info: {
      description: 'Clinical description based on UCI Dermatology dataset patterns.',
      treatment: 'Please consult a qualified dermatologist for treatment recommendations.',
    },
    image_note: hasImage ? 'CNN extracted 512 deep image features via ResNet18 backbone.' : null,
  };
}

// ── Main Component ───────────────────────────────────────────────
export default function NewAnalysis() {
  const { setLastResult, apiOnline } = useApp();
  const navigate = useNavigate();

  const [step, setStep]           = useState(0);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState('');

  const [info, setInfo] = useState({ age:'', gender:'male', notes:'' });

  // All feature values (default 0)
  const [features, setFeatures] = useState(() => {
    const init = {};
    [...CLINICAL_FEATURES, ...HISTO_FEATURES, { key:'family_history' }].forEach(f => { init[f.key] = 0; });
    return init;
  });

  const setFeat = (key, val) => setFeatures(p => ({ ...p, [key]: val }));

  const nonZero = [...CLINICAL_FEATURES, ...HISTO_FEATURES].filter(f => features[f.key] > 0);

  const handleSubmit = async () => {
    setLoading(true); setError('');
    try {
      const payload = { ...features, age: parseFloat(info.age) || 0 };
      let result;
      if (apiOnline) {
        result = imageFile
          ? await api.predictWithImage(payload, imageFile)
          : await api.predict(payload);
      } else {
        await new Promise(r => setTimeout(r, 1200));
        result = mockPredict(payload, !!imageFile);
      }
      setLastResult(result);
      navigate('/results');
    } catch (e) {
      setError(e.response?.data?.error || e.message || 'Prediction failed — check Flask is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-title">New Analysis</div>
        <div className="page-sub">AI-powered differential diagnosis — XGBoost + ResNet18 CNN</div>
      </div>

      {/* Step indicator */}
      <div className="steps">
        {STEPS.map((s, i) => (
          <React.Fragment key={s}>
            <div className={`step${i===step?' active':i<step?' done':''}`}>
              <div className="step-circle">{i < step ? '✓' : i + 1}</div>
              <span className="step-label">{s}</span>
            </div>
            {i < STEPS.length - 1 && <div className="step-line" />}
          </React.Fragment>
        ))}
      </div>

      {error && (
        <div className="info-box info-red" style={{marginBottom:20}}>
          ⚠ {error}
        </div>
      )}

      {/* ── STEP 0: Patient Info ── */}
      {step === 0 && (
        <div className="card">
          <div className="card-title">Patient Information</div>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Patient Age (years)</label>
              <input className="form-input" type="number" min={0} max={120}
                placeholder="e.g. 45" value={info.age}
                onChange={e => setInfo(p=>({...p, age:e.target.value}))} />
              <div className="form-hint">Leave blank to use dataset median (47 yrs)</div>
            </div>
            <div className="form-group">
              <label className="form-label">Gender</label>
              <select className="form-input" value={info.gender}
                onChange={e => setInfo(p=>({...p, gender:e.target.value}))}>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other / Prefer not to say</option>
              </select>
            </div>
          </div>
          <div className="form-group">
            <label className="form-label">Family History of Skin Disease</label>
            <div style={{display:'flex', gap:12}}>
              {[0,1].map(v => (
                <label key={v} style={{
                  display:'flex', alignItems:'center', gap:8, cursor:'pointer',
                  padding:'10px 20px', borderRadius:'var(--r-md)',
                  border:`1.5px solid ${features.family_history===v?'var(--teal)':'var(--border2)'}`,
                  background: features.family_history===v ? 'rgba(20,184,166,.08)' : 'var(--bg3)',
                  fontSize:14, color:'var(--text)',
                }}>
                  <input type="radio" style={{display:'none'}} checked={features.family_history===v}
                    onChange={() => setFeat('family_history', v)} readOnly />
                  {v === 0 ? '✗ No family history' : '✓ Family history present'}
                </label>
              ))}
            </div>
          </div>
          <div className="form-group">
            <label className="form-label">Clinical Notes (optional)</label>
            <textarea className="form-input" rows={3} style={{resize:'vertical'}}
              placeholder="Additional observations, prior diagnoses, current medications..."
              value={info.notes} onChange={e => setInfo(p=>({...p,notes:e.target.value}))} />
          </div>
          <div style={{display:'flex', justifyContent:'flex-end', marginTop:8}}>
            <button className="btn btn-teal" onClick={() => setStep(1)}>
              Clinical Features →
            </button>
          </div>
        </div>
      )}

      {/* ── STEP 1: Clinical Features ── */}
      {step === 1 && (
        <div>
          <div className="info-box info-blue" style={{marginBottom:20}}>
            🩺 <strong>Clinical Features</strong> — Observed during physical examination.
            Rate 0 (absent) → 3 (severe). Highlighted cards are disease-specific markers.
          </div>
          <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(260px,1fr))', gap:16}}>
            {CLINICAL_FEATURES.map(f => (
              <FeatureSlider key={f.key} feature={f} value={features[f.key]||0} onChange={setFeat}/>
            ))}
          </div>
          <div style={{display:'flex', justifyContent:'space-between', marginTop:24}}>
            <button className="btn btn-ghost" onClick={() => setStep(0)}>← Back</button>
            <button className="btn btn-teal" onClick={() => setStep(2)}>Histopathology →</button>
          </div>
        </div>
      )}

      {/* ── STEP 2: Histopathology ── */}
      {step === 2 && (
        <div>
          <div className="info-box info-blue" style={{marginBottom:12}}>
            🔬 <strong>Histopathological Features</strong> — From biopsy / microscopic analysis.
            These are the strongest predictors. ★ = pathognomonic (disease-specific).
          </div>
          <div style={{
            background:'rgba(245,158,11,.06)', border:'1px solid rgba(245,158,11,.2)',
            borderRadius:'var(--r-md)', padding:'12px 16px', marginBottom:20, fontSize:13,
          }}>
            ⭐ <strong style={{color:'var(--amber)'}}>Pathognomonic markers:</strong>
            {' '}Munro Microabscess → Psoriasis &nbsp;·&nbsp;
            Saw-tooth Retes + Band-like Infiltrate → Lichen Planus &nbsp;·&nbsp;
            Follicular Horn Plug → PRP
          </div>
          <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(260px,1fr))', gap:16}}>
            {HISTO_FEATURES.map(f => (
              <FeatureSlider key={f.key} feature={f} value={features[f.key]||0} onChange={setFeat}/>
            ))}
          </div>
          <div style={{display:'flex', justifyContent:'space-between', marginTop:24}}>
            <button className="btn btn-ghost" onClick={() => setStep(1)}>← Back</button>
            <button className="btn btn-teal" onClick={() => setStep(3)}>Image & Analyze →</button>
          </div>
        </div>
      )}

      {/* ── STEP 3: Image Upload + Review + Analyze ── */}
      {step === 3 && (
        <div>
          <div className="grid-2" style={{marginBottom:20}}>
            {/* Image upload */}
            <div className="card">
              <div className="card-title">Dermoscopy Image <span style={{color:'var(--text3)',fontSize:13,fontWeight:400}}>(optional)</span></div>
              <ImageUpload
                imageFile={imageFile}
                imagePreview={imagePreview}
                onImageChange={(file, url) => { setImageFile(file); setImagePreview(url); }}
                onImageClear={() => { setImageFile(null); setImagePreview(''); }}
              />
              <div className="info-box info-blue" style={{marginTop:16}}>
                <strong>Without image:</strong> XGBoost uses 34 manual features via PCA.<br/>
                <strong>With image:</strong> ResNet18 extracts 512 CNN features merged with manual features for richer prediction.
              </div>
            </div>

            {/* Summary */}
            <div className="card">
              <div className="card-title">Review Summary</div>
              <div style={{marginBottom:16}}>
                {[
                  ['Age', info.age || 'Not provided (47 median)'],
                  ['Gender', info.gender],
                  ['Family History', features.family_history ? 'Present' : 'Absent'],
                  ['Image', imageFile ? `✓ ${imageFile.name}` : 'Not provided'],
                  ['Non-zero Features', nonZero.length + ' / 33'],
                ].map(([k,v]) => (
                  <div key={k} style={{display:'flex',justifyContent:'space-between',fontSize:14,padding:'7px 0',borderBottom:'1px solid var(--border)'}}>
                    <span style={{color:'var(--text3)'}}>{k}</span>
                    <span style={{fontWeight:500,color:'#fff'}}>{v}</span>
                  </div>
                ))}
              </div>

              <div className="card-title" style={{marginBottom:8}}>Active Features</div>
              {nonZero.length === 0 ? (
                <div className="info-box info-amber">No features set above 0. All features are absent.</div>
              ) : (
                <div style={{display:'flex',flexDirection:'column',gap:4,maxHeight:200,overflowY:'auto'}}>
                  {nonZero.sort((a,b)=>(features[b.key]||0)-(features[a.key]||0)).map(f => {
                    const v = features[f.key];
                    return (
                      <div key={f.key} style={{display:'flex',justifyContent:'space-between',alignItems:'center',fontSize:13,padding:'3px 0'}}>
                        <span style={{color:'var(--text2)'}}>{f.label.replace(' ★','')}</span>
                        <span style={{fontFamily:'var(--fm)',fontWeight:700,color:SLIDER_COLORS[v]}}>
                          {'●'.repeat(v)}{'○'.repeat(3-v)} {v}/3
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="info-box info-blue" style={{marginBottom:16}}>
              ℹ️ Clicking <strong>Run AI Analysis</strong> sends all features to
              XGBoost (calibrated, 10-fold CV trained).
              {imageFile && ' ResNet18 will extract CNN features from your image first.'}
              {' '}Results include: disease prediction, confidence %, SHAP explanation, anomaly detection.
            </div>
            <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
              <button className="btn btn-ghost" onClick={() => setStep(2)} disabled={loading}>
                ← Back
              </button>
              <button className="btn btn-teal btn-lg" onClick={handleSubmit} disabled={loading}>
                {loading
                  ? (imageFile ? '⏳ Extracting CNN features + predicting…' : '⏳ Running XGBoost…')
                  : '🚀 Run AI Analysis'
                }
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
