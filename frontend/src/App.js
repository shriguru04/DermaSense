// frontend/src/App.js
import React, { useState, useEffect, createContext, useContext } from 'react';
import { BrowserRouter, Routes, Route, NavLink, useNavigate, Navigate } from 'react-router-dom';
import Dashboard   from './pages/Dashboard';
import NewAnalysis from './pages/NewAnalysis';
import Results     from './pages/Results';
import StatsPage   from './pages/StatsPage';
import { api }     from './services/api';
import './index.css';

export const AppCtx = createContext({});
export const useApp = () => useContext(AppCtx);

function Sidebar({ apiOnline }) {
  const navigate = useNavigate();
  const links = [
    { to:'/',       icon:'⬛', label:'Dashboard'       },
    { to:'/new',    icon:'＋', label:'New Analysis'    },
    { to:'/stats',  icon:'◎',  label:'Model Stats'     },
  ];
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-mark">DS</span>
        <div><div className="brand-name">DermaSense</div><div className="brand-sub">Clinical AI · v2.0</div></div>
      </div>
      <nav className="sidebar-nav">
        {links.map(l => (
          <NavLink key={l.to} to={l.to} end={l.to==='/'} className={({isActive})=>`nav-link${isActive?' active':''}`}>
            <span className="nav-icon">{l.icon}</span>{l.label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className={`api-pill ${apiOnline?'on':'off'}`}>
          <span className="dot"/>{apiOnline ? 'API Online' : 'Mock Mode'}
        </div>
        <button className="btn-new" onClick={()=>navigate('/new')}>+ New Analysis</button>
      </div>
    </aside>
  );
}

export default function App() {
  const [lastResult, setLastResult] = useState(null);
  const [apiOnline,  setApiOnline]  = useState(false);

  useEffect(() => {
    api.health().then(h => setApiOnline(h.model_ready === true));
    const iv = setInterval(() => api.health().then(h => setApiOnline(h.model_ready===true)), 15000);
    return () => clearInterval(iv);
  }, []);

  return (
    <AppCtx.Provider value={{ lastResult, setLastResult, apiOnline }}>
      <BrowserRouter>
        <div className="app-shell">
          <Sidebar apiOnline={apiOnline}/>
          <div className="main-area">
            <Routes>
              <Route path="/"       element={<Dashboard/>}/>
              <Route path="/new"    element={<NewAnalysis/>}/>
              <Route path="/results" element={<Results/>}/>
              <Route path="/stats"  element={<StatsPage/>}/>
              <Route path="*"       element={<Navigate to="/"/>}/>
            </Routes>
          </div>
        </div>
      </BrowserRouter>
    </AppCtx.Provider>
  );
}
