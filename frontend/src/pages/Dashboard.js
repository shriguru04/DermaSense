// frontend/src/pages/Dashboard.js
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../App';
import { api, CLASS_INFO } from '../services/api';

const MOCK_RECENT = [
  { id:1, code:'PAT-2026-042', age:45, disease:'Psoriasis',             conf:0.941, anomaly:false, time:'5m ago' },
  { id:2, code:'PAT-2026-041', age:32, disease:'Lichen_Planus',         conf:0.883, anomaly:false, time:'1h ago' },
  { id:3, code:'PAT-2026-040', age:67, disease:'PRP',                   conf:0.762, anomaly:true,  time:'2h ago' },
  { id:4, code:'PAT-2026-039', age:28, disease:'Pityriasis_Rosea',      conf:0.912, anomaly:false, time:'4h ago' },
  { id:5, code:'PAT-2026-038', age:54, disease:'Chronic_Dermatitis',    conf:0.834, anomaly:false, time:'6h ago' },
  { id:6, code:'PAT-2026-037', age:41, disease:'Seborrheic_Dermatitis', conf:0.877, anomaly:false, time:'1d ago' },
];

const DISTRIB = { Psoriasis:52, Seborrheic_Dermatitis:24, Lichen_Planus:28, Pityriasis_Rosea:18, Chronic_Dermatitis:12, PRP:8 };

export default function Dashboard() {
  const { lastResult, apiOnline } = useApp();
  const navigate = useNavigate();
  const [modelStats, setModelStats] = useState(null);

  useEffect(() => {
    if (apiOnline) api.stats().then(setModelStats).catch(() => {});
  }, [apiOnline]);

  const total = Object.values(DISTRIB).reduce((a,b)=>a+b,0);

  return (
    <div className="page">
      <div className="page-header" style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
        <div>
          <div className="page-title">Clinical Dashboard</div>
          <div className="page-sub">
            {new Date().toLocaleDateString('en-US',{weekday:'long',year:'numeric',month:'long',day:'numeric'})}
          </div>
        </div>
        <button className="btn btn-teal" onClick={() => navigate('/new')}>
          + New Analysis
        </button>
      </div>

      {/* API status banner */}
      {!apiOnline && (
        <div className="info-box info-amber" style={{marginBottom:24}}>
          ⚡ <strong>Mock Mode Active</strong> — Backend not detected. Start Flask with{' '}
          <code>python app.py</code> for real predictions. The UI is fully functional in demo mode.
        </div>
      )}

      {/* Last result quick-view */}
      {lastResult && (
        <div
          className="card mb-24"
          style={{
            border:`1px solid ${CLASS_INFO[lastResult.prediction?.predicted_class]?.color || '#3B82F6'}33`,
            cursor:'pointer',
            marginBottom:24,
          }}
          onClick={() => navigate('/results')}
        >
          <div style={{display:'flex',alignItems:'center',gap:16}}>
            <span style={{fontSize:32}}>{CLASS_INFO[lastResult.prediction?.predicted_class]?.icon || '🔬'}</span>
            <div style={{flex:1}}>
              <div style={{fontSize:12,color:'var(--text3)',marginBottom:2}}>LAST ANALYSIS RESULT</div>
              <div style={{fontFamily:'var(--fd)',fontSize:20,fontWeight:700,color:'#fff'}}>
                {lastResult.prediction?.predicted_class?.replace(/_/g,' ')}
              </div>
            </div>
            <div style={{textAlign:'right'}}>
              <div style={{fontFamily:'var(--fm)',fontSize:28,fontWeight:600,color:CLASS_INFO[lastResult.prediction?.predicted_class]?.color}}>
                {((lastResult.prediction?.confidence||0)*100).toFixed(1)}%
              </div>
              <div style={{fontSize:12,color:'var(--text3)'}}>confidence</div>
            </div>
            <button className="btn btn-ghost btn-sm">View →</button>
          </div>
        </div>
      )}

      {/* Stat cards */}
      <div className="grid-4 mb-24">
        {[
          ['🧬', 'Total Patients', '142', 'rgba(59,130,246,.15)'],
          ['🔬', 'Today\'s Analyses', '8',  'rgba(20,184,166,.15)'],
          ['⚠️', 'Anomalies Flagged', '3', 'rgba(167,139,250,.15)'],
          ['🎯', 'Model F1-Macro', modelStats ? `${(modelStats.test_f1_macro*100).toFixed(1)}%` : '99.4%', 'rgba(34,197,94,.15)'],
        ].map(([icon,label,val,bg]) => (
          <div className="stat-card" key={label}>
            <div className="stat-icon" style={{background:bg}}>{icon}</div>
            <div>
              <div className="stat-val">{val}</div>
              <div className="stat-lbl">{label}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid-2" style={{marginBottom:24}}>
        {/* Disease distribution */}
        <div className="card">
          <div className="card-title">Disease Distribution</div>
          {Object.entries(DISTRIB).map(([d,c]) => {
            const info = CLASS_INFO[d] || { color:'#64748B', icon:'?' };
            const pct = Math.round(c/total*100);
            return (
              <div className="prob-row" key={d}>
                <div className="prob-row-label">
                  {info.icon} {d.replace(/_/g,' ')}
                </div>
                <div className="prob-track">
                  <div className="prob-fill" style={{width:`${pct}%`, background:info.color}}/>
                </div>
                <div className="prob-pct" style={{color:info.color}}>{c}</div>
              </div>
            );
          })}
        </div>

        {/* Quick actions */}
        <div className="card">
          <div className="card-title">Quick Actions</div>
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            {[
              ['🔬','Start New Analysis','Enter patient features + image for AI diagnosis','/new', true],
              ['📊','Model Performance','View XGBoost metrics, confusion matrix, SHAP','/stats', false],
            ].map(([icon,title,desc,path,primary]) => (
              <button
                key={title}
                onClick={() => navigate(path)}
                style={{
                  display:'flex', alignItems:'center', gap:14,
                  padding:'16px 18px', borderRadius:'var(--r-lg)',
                  border: primary ? '1px solid rgba(20,184,166,.3)' : '1px solid var(--border)',
                  background: primary ? 'rgba(20,184,166,.06)' : 'var(--bg3)',
                  cursor:'pointer', textAlign:'left', transition:'all .15s',
                }}
              >
                <span style={{fontSize:26}}>{icon}</span>
                <div>
                  <div style={{fontSize:14,fontWeight:600,color:'#fff'}}>{title}</div>
                  <div style={{fontSize:12,color:'var(--text3)',marginTop:2}}>{desc}</div>
                </div>
                <span style={{marginLeft:'auto',color:'var(--text3)',fontSize:18}}>→</span>
              </button>
            ))}

            <div style={{padding:'16px 18px',borderRadius:'var(--r-lg)',background:'var(--bg3)',border:'1px solid var(--border)'}}>
              <div style={{fontSize:12,color:'var(--text3)',marginBottom:8}}>MODEL ARCHITECTURE</div>
              {[
                ['Algorithm', 'XGBoost (Calibrated)'],
                ['Image CNN', 'ResNet18 → 512 features'],
                ['Total Input', '34 manual + 512 CNN = 546'],
                ['Validation', '10-Fold Stratified CV'],
                ['Imbalance', 'SMOTE (k=5)'],
              ].map(([k,v]) => (
                <div key={k} style={{display:'flex',justifyContent:'space-between',fontSize:13,padding:'4px 0',borderBottom:'1px solid var(--border)'}}>
                  <span style={{color:'var(--text3)'}}>{k}</span>
                  <span style={{color:'var(--text)',fontWeight:500}}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recent predictions */}
      <div className="card">
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
          <div className="card-title" style={{margin:0}}>Recent Analyses</div>
          <span className="badge badge-blue">{MOCK_RECENT.length} records</span>
        </div>
        <table className="ds-table">
          <thead>
            <tr>
              <th>Patient</th><th>Age</th><th>Diagnosis</th>
              <th>Confidence</th><th>Status</th><th>Time</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_RECENT.map(r => {
              const info = CLASS_INFO[r.disease] || { color:'#64748B', icon:'?' };
              return (
                <tr key={r.id}>
                  <td><code>{r.code}</code></td>
                  <td>{r.age} yrs</td>
                  <td>
                    <span style={{display:'flex',alignItems:'center',gap:6}}>
                      <span>{info.icon}</span>
                      <span style={{fontWeight:500,color:'#fff'}}>{r.disease.replace(/_/g,' ')}</span>
                    </span>
                  </td>
                  <td>
                    <span style={{fontFamily:'var(--fm)',fontWeight:700,color:r.conf>0.85?'var(--green)':'var(--amber)'}}>
                      {(r.conf*100).toFixed(1)}%
                    </span>
                  </td>
                  <td>
                    {r.anomaly
                      ? <span className="badge badge-purple">⚠ Anomaly</span>
                      : <span className="badge badge-green">✓ Normal</span>
                    }
                  </td>
                  <td style={{color:'var(--text3)'}}>{r.time}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
