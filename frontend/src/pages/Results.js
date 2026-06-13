// frontend/src/pages/Results.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../App';
import { CLASS_INFO } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function Results() {
  const { lastResult } = useApp();
  const navigate = useNavigate();
  const [tab, setTab] = useState('overview');

  if (!lastResult) {
    return (
      <div className="page">
        <div className="card" style={{textAlign:'center', padding:60}}>
          <div style={{fontSize:48, marginBottom:16}}>🔬</div>
          <div style={{fontFamily:'var(--fd)', fontSize:22, color:'#fff', marginBottom:8}}>No analysis yet</div>
          <div style={{color:'var(--text3)', marginBottom:24}}>Run a new analysis to see results here</div>
          <button className="btn btn-teal" onClick={() => navigate('/new')}>+ New Analysis</button>
        </div>
      </div>
    );
  }

  const { prediction, anomaly, explanation, disease_info, image_note } = lastResult;
  const predicted  = prediction?.predicted_class || 'Unknown';
  const info       = CLASS_INFO[predicted] || { color:'#64748B', bg:'#1E2535', icon:'🔬' };
  const conf       = parseFloat(prediction?.confidence || 0);
  const probs      = prediction?.all_probabilities || {};
  const topFeats   = explanation?.top_features || [];
  const maxShap    = Math.max(...topFeats.map(f => Math.abs(f.shap || 0)), 0.01);

  // Sort probs highest first
  const sortedProbs = Object.entries(probs).sort(([,a],[,b]) => b - a);

  // Chart data
  const chartData = sortedProbs.map(([d, p]) => ({
    name: d.replace(/_/g,' ').replace('Seborrheic Dermatitis','Seborrheic D.').replace('Pityriasis Rosea','Pit. Rosea').replace('Chronic Dermatitis','Chronic D.'),
    value: parseFloat((p*100).toFixed(1)),
    color: CLASS_INFO[d]?.color || '#64748B',
  }));

  return (
    <div className="page">
      {/* Header */}
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:24}}>
        <div>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/')} style={{marginBottom:8}}>
            ← Dashboard
          </button>
          <div className="page-title">Analysis Results</div>
          <div className="page-sub">XGBoost prediction · SHAP explanation · Anomaly detection</div>
        </div>
        <div style={{display:'flex', gap:10}}>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/new')}>+ New Analysis</button>
        </div>
      </div>

      {/* Prediction Hero */}
      <div className="pred-hero" style={{background:`linear-gradient(135deg,${info.color}cc,${info.color}66)`}}>
        <div className="hero-icon">{info.icon}</div>
        <div className="hero-body">
          <div className="hero-eyebrow">AI Diagnosis — XGBoost (Calibrated)</div>
          <div className="hero-disease">{predicted.replace(/_/g,' ')}</div>
          <div className="hero-conf">
            Confidence: {(conf*100).toFixed(1)}%
            {prediction?.has_image && <span className="cnn-badge" style={{marginLeft:12}}>🔬 CNN+Manual</span>}
          </div>
        </div>
        <div style={{textAlign:'right'}}>
          <div className="hero-pct">{(conf*100).toFixed(0)}<span>%</span></div>
          <div style={{fontSize:12, opacity:.6}}>model certainty</div>
        </div>
      </div>

      {/* CNN note */}
      {image_note && (
        <div className="info-box info-blue" style={{marginBottom:16}}>
          🔬 {image_note}
        </div>
      )}

      {/* Anomaly Banner */}
      <div className={`anom-banner ${anomaly?.is_anomaly ? 'anom-flag' : 'anom-ok'}`}>
        <div style={{fontSize:28}}>{anomaly?.is_anomaly ? '⚠️' : '✅'}</div>
        <div style={{flex:1}}>
          <div className="anom-title">
            {anomaly?.is_anomaly ? 'Anomalous Presentation — Clinician Review Recommended' : 'Normal Presentation — Within Training Distribution'}
          </div>
          <div className="anom-detail">
            Euclidean distance: <strong>{anomaly?.euclidean_distance?.toFixed(2)}</strong> (threshold: {anomaly?.threshold})
            &nbsp;·&nbsp; Anomaly probability: <strong>{((anomaly?.anomaly_probability||0)*100).toFixed(0)}%</strong>
            &nbsp;·&nbsp; Nearest class: {anomaly?.nearest_class?.replace(/_/g,' ')}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs">
        {[
          ['overview', 'Overview'],
          ['probabilities', 'All Probabilities'],
          ['shap', 'SHAP Explanation'],
          ['disease', 'Disease Info'],
        ].map(([id, label]) => (
          <button key={id} className={`tab${tab===id?' active':''}`} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div className="grid-2">
          {/* SHAP top features */}
          <div className="card">
            <div className="card-title">Top Contributing Features (SHAP)</div>
            {topFeats.length === 0 ? (
              <div style={{color:'var(--text3)',fontSize:14}}>SHAP unavailable — start Flask backend for explanations.</div>
            ) : (
              <>
                {topFeats.map((f, i) => {
                  const pct = (Math.abs(f.shap) / maxShap) * 50;
                  return (
                    <div key={i} className="shap-row">
                      <div className="shap-name">
                        <span style={{fontWeight:500}}>{f.label}</span>
                      </div>
                      <div className="shap-val">{f.value}/3</div>
                      <div className="shap-track">
                        {f.shap >= 0
                          ? <div className="shap-pos" style={{width:`${pct}%`}}/>
                          : <div className="shap-neg" style={{width:`${pct}%`}}/>
                        }
                      </div>
                      <div className="shap-score" style={{color:f.shap>=0?'#60A5FA':'#F87171'}}>
                        {f.shap>=0?'+':''}{f.shap.toFixed(3)}
                      </div>
                    </div>
                  );
                })}
                <div style={{fontSize:11,color:'var(--text3)',marginTop:12}}>
                  Blue = pushes toward prediction · Red = pushes against prediction
                </div>
              </>
            )}
          </div>

          {/* Clinical explanation */}
          <div className="card">
            <div className="card-title">Clinical Explanation</div>
            <div
              style={{fontSize:14,lineHeight:1.8,color:'var(--text2)',marginBottom:20}}
              dangerouslySetInnerHTML={{__html: explanation?.clinical_text || 'No explanation available.'}}
            />
            {explanation?.counterfactual && (
              <div style={{
                background:'var(--bg3)', borderRadius:'var(--r-md)',
                padding:'14px 16px', fontSize:13, color:'var(--text3)',
                borderLeft:'3px solid var(--border2)',
              }}>
                <strong style={{color:'var(--text)'}}>Counterfactual:</strong><br/>
                {explanation.counterfactual}
              </div>
            )}

            {/* Anomaly distances */}
            {anomaly?.distances && (
              <div style={{marginTop:20}}>
                <div style={{fontSize:13,fontWeight:600,color:'var(--text)',marginBottom:8}}>
                  Euclidean Distance to Each Class Centroid
                </div>
                {Object.entries(anomaly.distances).sort(([,a],[,b])=>a-b).map(([cls, dist]) => {
                  const ci = CLASS_INFO[cls] || {color:'#64748B'};
                  const maxD = Math.max(...Object.values(anomaly.distances));
                  return (
                    <div key={cls} className="prob-row">
                      <div className="prob-row-label" style={{width:180}}>
                        {CLASS_INFO[cls]?.icon} {cls.replace(/_/g,' ')}
                      </div>
                      <div className="prob-track">
                        <div className="prob-fill" style={{width:`${(dist/maxD)*100}%`, background:ci.color, opacity:.7}}/>
                      </div>
                      <div className="prob-pct" style={{color:ci.color}}>{dist.toFixed(1)}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── PROBABILITIES ── */}
      {tab === 'probabilities' && (
        <div className="grid-2">
          <div className="card">
            <div className="card-title">Probability Distribution</div>
            {sortedProbs.map(([d, p]) => {
              const ci = CLASS_INFO[d] || {color:'#64748B', icon:'?'};
              const pct = (p*100).toFixed(1);
              const isTop = d === predicted;
              return (
                <div key={d} style={{
                  padding:'10px 14px', borderRadius:'var(--r-md)', marginBottom:6,
                  background: isTop ? `${ci.color}11` : 'transparent',
                  border: isTop ? `1px solid ${ci.color}33` : '1px solid transparent',
                }}>
                  <div className="prob-row" style={{padding:0}}>
                    <div className="prob-row-label">
                      {ci.icon} {d.replace(/_/g,' ')}
                      {isTop && <span className="badge badge-blue" style={{marginLeft:8,fontSize:10}}>TOP</span>}
                    </div>
                    <div className="prob-track">
                      <div className="prob-fill" style={{width:`${pct}%`, background:ci.color, opacity:isTop?1:0.5}}/>
                    </div>
                    <div className="prob-pct" style={{color:ci.color, fontWeight:isTop?700:500}}>
                      {pct}%
                    </div>
                  </div>
                </div>
              );
            })}
            <div className="info-box info-blue" style={{marginTop:16}}>
              Probabilities from Platt-calibrated XGBoost. Uncertainty = {(100-conf*100).toFixed(1)}%
            </div>
          </div>

          <div className="card">
            <div className="card-title">Probability Bar Chart</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} margin={{top:10,right:10,left:-20,bottom:60}}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.06)"/>
                <XAxis dataKey="name" tick={{fontSize:11,fill:'#94A3B8'}} angle={-35} textAnchor="end" interval={0}/>
                <YAxis tick={{fontSize:11,fill:'#94A3B8'}} tickFormatter={v=>`${v}%`} domain={[0,100]}/>
                <Tooltip
                  formatter={v=>[`${v}%`,'Probability']}
                  contentStyle={{background:'#161B27',border:'1px solid rgba(255,255,255,.1)',borderRadius:8}}
                  labelStyle={{color:'#fff'}}
                />
                <Bar dataKey="value" radius={[4,4,0,0]}>
                  {chartData.map((d,i) => <Cell key={i} fill={d.color}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── SHAP TAB ── */}
      {tab === 'shap' && (
        <div className="grid-2">
          <div className="card">
            <div className="card-title">SHAP Waterfall — Feature Impact</div>
            <div style={{fontSize:13,color:'var(--text3)',marginBottom:16}}>
              Each bar shows how much a feature pushed toward (blue) or away (red) from{' '}
              <strong style={{color:'#fff'}}>{predicted.replace(/_/g,' ')}</strong>
            </div>
            {topFeats.map((f, i) => {
              const pct = (Math.abs(f.shap)/maxShap)*100;
              return (
                <div key={i} style={{marginBottom:14}}>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:13,marginBottom:4}}>
                    <span style={{fontWeight:500,color:'var(--text)'}}>
                      {f.label}
                      <span style={{color:'var(--text3)',fontWeight:400,marginLeft:6}}>({f.value}/3)</span>
                    </span>
                    <span style={{fontFamily:'var(--fm)',fontWeight:700,color:f.shap>=0?'#60A5FA':'#F87171'}}>
                      {f.shap>=0?'+':''}{f.shap.toFixed(3)}
                    </span>
                  </div>
                  <div style={{height:10,background:'rgba(255,255,255,.06)',borderRadius:5,overflow:'hidden'}}>
                    <div style={{
                      height:'100%', width:`${pct}%`,
                      background: f.shap>=0?'linear-gradient(90deg,#3B82F6,#1D4ED8)':'linear-gradient(90deg,#EF4444,#991B1B)',
                      borderRadius:5, transition:'width 1s',
                    }}/>
                  </div>
                  <div style={{fontSize:11,color:'var(--text3)',marginTop:2}}>
                    {f.direction === 'SUPPORTS' ? '↑ Supports this diagnosis' : '↓ Against this diagnosis'}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="card">
            <div className="card-title">XAI Methodology</div>
            {[
              ['#3B82F6','SHAP (TreeExplainer)',
               'SHapley Additive exPlanations applied to the XGBoost base estimator. Exact Shapley values computed via tree traversal. Output: per-feature contribution to predicted class log-odds.'],
              ['#14B8A6','PCA Projection',
               `SHAP values computed in PCA space (${lastResult?.stats?.n_pca_components || '~20'} components) then projected back to original 34-feature space via PCA components matrix. Allows interpretable feature-level attribution.`],
              ['#A78BFA','Calibrated Probabilities',
               'Platt scaling (sigmoid calibration) applied via CalibratedClassifierCV (cv=5). Ensures predicted % matches actual frequency — 80% confidence means 80% of similar cases are correct.'],
              ['#EF4444','Anomaly Detection',
               'Layer 1: Euclidean distance to class centroids in PCA space. Layer 2: Isolation Forest (contamination=0.05). Both must agree before high-confidence anomaly flag is raised.'],
            ].map(([col,title,desc]) => (
              <div key={title} style={{
                padding:'14px 16px', background:'var(--bg3)', borderRadius:'var(--r-md)',
                borderLeft:`3px solid ${col}`, marginBottom:12,
              }}>
                <div style={{fontSize:14,fontWeight:600,color:'#fff',marginBottom:4}}>{title}</div>
                <div style={{fontSize:13,color:'var(--text2)',lineHeight:1.6}}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── DISEASE INFO ── */}
      {tab === 'disease' && (
        <div className="card">
          <div style={{display:'flex',alignItems:'center',gap:14,marginBottom:20}}>
            <span style={{fontSize:42}}>{info.icon}</span>
            <div>
              <div style={{fontFamily:'var(--fd)',fontSize:24,fontWeight:700,color:'#fff'}}>
                {predicted.replace(/_/g,' ')}
              </div>
              <div style={{fontSize:13,color:'var(--text2)',marginTop:2}}>Clinical Reference</div>
            </div>
          </div>
          <div className="grid-2">
            <div>
              <div style={{fontSize:15,fontWeight:600,color:'#fff',marginBottom:8}}>Description</div>
              <p style={{fontSize:14,lineHeight:1.8,color:'var(--text2)'}}>
                {disease_info?.description || 'Clinical description based on UCI Dermatology dataset patterns.'}
              </p>
            </div>
            <div>
              <div style={{fontSize:15,fontWeight:600,color:'#fff',marginBottom:8}}>Treatment Options</div>
              <div className="info-box info-blue">
                {disease_info?.treatment || 'Consult a qualified dermatologist for diagnosis confirmation and treatment.'}
              </div>
            </div>
          </div>
          <div className="info-box info-amber" style={{marginTop:20}}>
            ⚠️ <strong>Clinical Disclaimer:</strong> This AI prediction is a clinical decision support tool only.
            Final diagnosis must be confirmed by a qualified dermatologist. Always correlate with complete
            clinical history and physical examination.
          </div>
        </div>
      )}
    </div>
  );
}
