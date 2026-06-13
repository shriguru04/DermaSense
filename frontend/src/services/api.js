// frontend/src/services/api.js
import axios from 'axios';

const BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const client = axios.create({ baseURL: BASE, timeout: 60000 });

// ── FEATURE DEFINITIONS ──────────────────────────────────────────
export const CLINICAL_FEATURES = [
  { key:'erythema',                 label:'Erythema',                   desc:'Skin redness severity' },
  { key:'scaling',                  label:'Scaling',                    desc:'Skin flaking / desquamation' },
  { key:'definite_borders',         label:'Definite Borders',           desc:'Clarity of lesion edge' },
  { key:'itching',                  label:'Itching',                    desc:'Pruritus severity' },
  { key:'koebner_phenomenon',       label:'Koebner Phenomenon',         desc:'Lesions appear at sites of trauma' },
  { key:'polygonal_papules',        label:'Polygonal Papules',          desc:'Flat-topped polygonal lesions — LP marker' },
  { key:'follicular_papules',       label:'Follicular Papules',         desc:'Hair follicle-centered lesions' },
  { key:'oral_mucosal_involvement', label:'Oral Mucosal Involvement',   desc:'Mouth/gum lesions — LP marker' },
  { key:'knee_elbow_involvement',   label:'Knee & Elbow Involvement',   desc:'Extensor surface lesions — Psoriasis marker' },
  { key:'scalp_involvement',        label:'Scalp Involvement',          desc:'Scalp lesion severity' },
];

export const HISTO_FEATURES = [
  { key:'melanin_incontinence',          label:'Melanin Incontinence',          desc:'Pigment cell damage into dermis' },
  { key:'eosinophils_infiltrate',        label:'Eosinophil Infiltrate',         desc:'Allergic response cells — atopic marker' },
  { key:'pnl_infiltrate',               label:'PNL Infiltrate',                desc:'Neutrophil presence in epidermis' },
  { key:'fibrosis_papillary_dermis',    label:'Fibrosis (Papillary Dermis)',   desc:'Fibrous tissue / scarring' },
  { key:'exocytosis',                   label:'Exocytosis',                    desc:'Inflammatory cells entering epidermis' },
  { key:'acanthosis',                   label:'Acanthosis',                    desc:'Epidermal thickening' },
  { key:'hyperkeratosis',               label:'Hyperkeratosis',                desc:'Thickening of stratum corneum' },
  { key:'parakeratosis',                label:'Parakeratosis',                 desc:'Abnormal keratinocyte maturation — Psoriasis' },
  { key:'clubbing_rete_ridges',         label:'Clubbing of Rete Ridges',       desc:'Bulbous rete ridge ends — Psoriasis' },
  { key:'elongation_rete_ridges',       label:'Elongation of Rete Ridges',     desc:'Extended rete ridges — Psoriasis' },
  { key:'thinning_suprapapillary',      label:'Thinning Suprapapillary',       desc:'Thin epidermis above dermal papillae' },
  { key:'spongiform_pustule',           label:'Spongiform Pustule',            desc:'Neutrophilic microabscess' },
  { key:'munro_microabscess',           label:'Munro Microabscess ★',          desc:'PATHOGNOMONIC for Psoriasis' },
  { key:'focal_hypergranulosis',        label:'Focal Hypergranulosis',         desc:'Localized granular layer increase — Pit. Rosea' },
  { key:'disappearance_granular_layer', label:'Disappearance Granular Layer',  desc:'Loss of granular cells — Psoriasis' },
  { key:'vacuolisation_basal_layer',    label:'Vacuolisation Basal Layer ★',   desc:'PATHOGNOMONIC for Lichen Planus' },
  { key:'spongiosis',                   label:'Spongiosis',                    desc:'Intercellular oedema — Chronic Dermatitis' },
  { key:'saw_tooth_retes',              label:'Saw-tooth Retes ★',             desc:'PATHOGNOMONIC for Lichen Planus' },
  { key:'follicular_horn_plug',         label:'Follicular Horn Plug ★',        desc:'PATHOGNOMONIC for PRP' },
  { key:'perifollicular_parakeratosis', label:'Perifollicular Parakeratosis',  desc:'Parakeratosis around follicles — PRP' },
  { key:'inflammatory_infiltrate',      label:'Inflammatory Infiltrate',       desc:'Lymphocytic infiltration' },
  { key:'band_like_infiltrate',         label:'Band-like Infiltrate ★',        desc:'PATHOGNOMONIC for Lichen Planus' },
];

export const ALL_FEATURES = [...CLINICAL_FEATURES, HISTO_FEATURES[9], ...HISTO_FEATURES];

export const CLASS_INFO = {
  Psoriasis:             { color:'#3B82F6', bg:'#EFF6FF', icon:'🔵' },
  Seborrheic_Dermatitis: { color:'#F59E0B', bg:'#FFFBEB', icon:'🟡' },
  Lichen_Planus:         { color:'#8B5CF6', bg:'#F5F3FF', icon:'🟣' },
  Pityriasis_Rosea:      { color:'#10B981', bg:'#ECFDF5', icon:'🟢' },
  Chronic_Dermatitis:    { color:'#EF4444', bg:'#FEF2F2', icon:'🔴' },
  PRP:                   { color:'#EC4899', bg:'#FDF2F8', icon:'🩷' },
};

// ── API CALLS ────────────────────────────────────────────────────
export const api = {
  health: () => client.get('/api/health').then(r => r.data).catch(() => ({ status:'offline', model_ready:false })),

  predict: (features) =>
    client.post('/api/predict', features).then(r => r.data),

  predictWithImage: (features, imageFile) => {
    const fd = new FormData();
    fd.append('features', JSON.stringify(features));
    fd.append('image', imageFile);
    return client.post('/api/predict-with-image', fd, {
      headers:{ 'Content-Type':'multipart/form-data' }
    }).then(r => r.data);
  },

  stats: () => client.get('/api/stats').then(r => r.data),
};
