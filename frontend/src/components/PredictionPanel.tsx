import { useState } from 'react'

type ModelKey = 'logistic' | 'forest' | 'neural'

const MODEL_OPTIONS: { key: ModelKey; label: string }[] = [
  { key: 'logistic', label: 'Logistic Regression' },
  { key: 'forest', label: 'Random Forest' },
  { key: 'neural', label: 'Neural Network' },
]

type Feature = {
  name: string
  key: string
  min: number
  max: number
  step: number
  default: number
  sampleMalignant: number
  sampleBenign: number
}

const TOP_FEATURES: Feature[] = [
  { name: 'Mean Radius', key: 'mean_radius', min: 6, max: 28, step: 0.1, default: 14, sampleMalignant: 20.57, sampleBenign: 13.08 },
  { name: 'Mean Texture', key: 'mean_texture', min: 9, max: 40, step: 0.1, default: 19, sampleMalignant: 17.77, sampleBenign: 15.71 },
  { name: 'Mean Perimeter', key: 'mean_perimeter', min: 43, max: 188, step: 0.5, default: 90, sampleMalignant: 132.9, sampleBenign: 84.11 },
  { name: 'Mean Area', key: 'mean_area', min: 143, max: 2501, step: 1, default: 650, sampleMalignant: 1326.0, sampleBenign: 536.5 },
  { name: 'Mean Smoothness', key: 'mean_smoothness', min: 0.05, max: 0.16, step: 0.001, default: 0.096, sampleMalignant: 0.0847, sampleBenign: 0.0948 },
  { name: 'Mean Compactness', key: 'mean_compactness', min: 0.02, max: 0.35, step: 0.001, default: 0.1, sampleMalignant: 0.0786, sampleBenign: 0.0612 },
  { name: 'Mean Concavity', key: 'mean_concavity', min: 0, max: 0.43, step: 0.001, default: 0.09, sampleMalignant: 0.2654, sampleBenign: 0.0553 },
  { name: 'Mean Concave Points', key: 'mean_concave_points', min: 0, max: 0.2, step: 0.001, default: 0.048, sampleMalignant: 0.1471, sampleBenign: 0.0337 },
  { name: 'Mean Symmetry', key: 'mean_symmetry', min: 0.1, max: 0.3, step: 0.001, default: 0.18, sampleMalignant: 0.1812, sampleBenign: 0.1619 },
  { name: 'Mean Fractal Dimension', key: 'mean_fractal_dimension', min: 0.05, max: 0.1, step: 0.0001, default: 0.062, sampleMalignant: 0.0551, sampleBenign: 0.0575 },
]

// Means for the remaining 20 features (features 10-29 from breast cancer dataset)
const REMAINING_MEANS = [
  0.2654, 0.9853, 0.6009, 0.5874, 0.0768, 0.0454,
  0.0812, 0.0276, 0.0233, 0.0077,
  24.99, 23.41, 158.8, 1956.0, 0.1238, 0.1866,
  0.2416, 0.1860, 0.2750, 0.0890,
]

type PredictResult = {
  prediction: number
  probability: number
  confidence: string
  label: string
}

const API = import.meta.env.VITE_API_URL ?? ''

export default function PredictionPanel() {
  const [selectedModel, setSelectedModel] = useState<ModelKey>('logistic')
  const [featureValues, setFeatureValues] = useState<Record<string, number>>(
    Object.fromEntries(TOP_FEATURES.map((f) => [f.key, f.default]))
  )
  const [result, setResult] = useState<PredictResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function useSamplePatient(type: 'malignant' | 'benign') {
    const vals: Record<string, number> = {}
    for (const f of TOP_FEATURES) {
      vals[f.key] = type === 'malignant' ? f.sampleMalignant : f.sampleBenign
    }
    setFeatureValues(vals)
    setResult(null)
  }

  function buildFeatureArray(): number[] {
    const top10 = TOP_FEATURES.map((f) => featureValues[f.key])
    return [...top10, ...REMAINING_MEANS]
  }

  async function handlePredict() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const features = buildFeatureArray()
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: selectedModel, features }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setResult(json)
    } catch (e: unknown) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <h2 className="text-xl font-semibold text-slate-700">Make a Prediction</h2>

      {/* Model selector */}
      <div className="bg-white rounded-xl shadow p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Select Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value as ModelKey)}
            className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
          </select>
        </div>

        {/* Sample buttons */}
        <div className="flex gap-3">
          <button
            onClick={() => useSamplePatient('malignant')}
            className="px-4 py-2 text-sm bg-red-50 text-red-700 border border-red-200 rounded-lg hover:bg-red-100 transition"
          >
            Use Sample Malignant Patient
          </button>
          <button
            onClick={() => useSamplePatient('benign')}
            className="px-4 py-2 text-sm bg-green-50 text-green-700 border border-green-200 rounded-lg hover:bg-green-100 transition"
          >
            Use Sample Benign Patient
          </button>
        </div>
      </div>

      {/* Feature sliders */}
      <div className="bg-white rounded-xl shadow p-6 space-y-5">
        <h3 className="font-semibold text-slate-700">Feature Values (Top 10)</h3>
        {TOP_FEATURES.map((f) => (
          <div key={f.key}>
            <div className="flex justify-between text-sm mb-1">
              <span className="font-medium text-slate-600">{f.name}</span>
              <span className="font-mono text-indigo-600">{featureValues[f.key]}</span>
            </div>
            <input
              type="range"
              min={f.min}
              max={f.max}
              step={f.step}
              value={featureValues[f.key]}
              onChange={(e) =>
                setFeatureValues((prev) => ({ ...prev, [f.key]: parseFloat(e.target.value) }))
              }
              className="w-full accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-slate-400 mt-0.5">
              <span>{f.min}</span>
              <span>{f.max}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Predict button */}
      <button
        onClick={handlePredict}
        disabled={loading}
        className="w-full py-3 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-700 disabled:opacity-50 transition"
      >
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {error && <div className="text-red-500 text-sm bg-red-50 rounded-lg p-3">Error: {error}</div>}

      {/* Result card */}
      {result && (
        <div className="bg-white rounded-xl shadow p-6 space-y-4">
          <div className={`text-3xl font-bold text-center py-3 rounded-lg ${
            result.label === 'Malignant' ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600'
          }`}>
            {result.label}
          </div>

          {/* Probability bar */}
          <div>
            <div className="flex justify-between text-sm text-slate-600 mb-1">
              <span>Confidence Probability</span>
              <span className="font-mono font-semibold">{(result.probability * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full h-3 bg-slate-200 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  result.label === 'Malignant' ? 'bg-red-500' : 'bg-green-500'
                }`}
                style={{ width: `${result.probability * 100}%` }}
              />
            </div>
          </div>

          {/* Confidence badge */}
          <div className="flex justify-center">
            <span className={`px-4 py-1.5 rounded-full text-sm font-semibold ${
              result.confidence === 'High'
                ? 'bg-green-100 text-green-700'
                : result.confidence === 'Medium'
                ? 'bg-yellow-100 text-yellow-700'
                : 'bg-red-100 text-red-700'
            }`}>
              {result.confidence} Confidence
            </span>
          </div>

          <p className="text-xs text-slate-400 text-center italic">
            For educational purposes only. Not for clinical use.
          </p>
        </div>
      )}
    </div>
  )
}
