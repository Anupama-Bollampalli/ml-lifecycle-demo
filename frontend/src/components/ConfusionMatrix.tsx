import { useEffect, useState } from 'react'

type ModelKey = 'logistic' | 'forest' | 'neural'

const MODEL_OPTIONS: { key: ModelKey; label: string }[] = [
  { key: 'logistic', label: 'Logistic Regression' },
  { key: 'forest', label: 'Random Forest' },
  { key: 'neural', label: 'Neural Network' },
]

type CMData = { tp: number; fp: number; tn: number; fn: number }

const API = import.meta.env.VITE_API_URL ?? ''

export default function ConfusionMatrix() {
  const [selectedModel, setSelectedModel] = useState<ModelKey>('logistic')
  const [cm, setCm] = useState<CMData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(`${API}/models/${selectedModel}/confusion-matrix`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(setCm)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedModel])

  const sensitivity = cm ? (cm.tp / (cm.tp + cm.fn || 1)) : null
  const specificity = cm ? (cm.tn / (cm.tn + cm.fp || 1)) : null
  const ppv = cm ? (cm.tp / (cm.tp + cm.fp || 1)) : null
  const npv = cm ? (cm.tn / (cm.tn + cm.fn || 1)) : null

  const CELL_SIZE = 100
  const MARGIN = 70
  const W = MARGIN + CELL_SIZE * 2 + 20
  const H = MARGIN + CELL_SIZE * 2 + 20

  return (
    <div className="space-y-6 max-w-2xl">
      <h2 className="text-xl font-semibold text-slate-700">Confusion Matrix</h2>

      <div className="bg-white rounded-xl shadow p-6 space-y-6">
        {/* Model selector */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Select Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value as ModelKey)}
            className="border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
          </select>
        </div>

        {loading && <div className="text-slate-400 text-sm">Loading...</div>}
        {error && <div className="text-red-500 text-sm">Error: {error}</div>}

        {cm && !loading && (
          <>
            {/* SVG Grid */}
            <div className="flex justify-center">
              <svg width={W} height={H}>
                {/* Axis labels */}
                <text x={MARGIN + CELL_SIZE} y={18} textAnchor="middle" fontSize={12} fontWeight={600} fill="#475569">
                  Predicted
                </text>
                <text x={MARGIN + CELL_SIZE * 0.5} y={MARGIN - 8} textAnchor="middle" fontSize={11} fill="#64748b">
                  Positive
                </text>
                <text x={MARGIN + CELL_SIZE * 1.5} y={MARGIN - 8} textAnchor="middle" fontSize={11} fill="#64748b">
                  Negative
                </text>
                {/* Rotated "Actual" label */}
                <text
                  x={15}
                  y={MARGIN + CELL_SIZE}
                  textAnchor="middle"
                  fontSize={12}
                  fontWeight={600}
                  fill="#475569"
                  transform={`rotate(-90, 15, ${MARGIN + CELL_SIZE})`}
                >
                  Actual
                </text>
                <text x={MARGIN - 10} y={MARGIN + CELL_SIZE * 0.5 + 5} textAnchor="end" fontSize={11} fill="#64748b">
                  Positive
                </text>
                <text x={MARGIN - 10} y={MARGIN + CELL_SIZE * 1.5 + 5} textAnchor="end" fontSize={11} fill="#64748b">
                  Negative
                </text>

                {/* TP - green */}
                <rect x={MARGIN} y={MARGIN} width={CELL_SIZE} height={CELL_SIZE} fill="#dcfce7" stroke="#86efac" strokeWidth={1.5} rx={4} />
                <text x={MARGIN + CELL_SIZE / 2} y={MARGIN + CELL_SIZE / 2 - 8} textAnchor="middle" fontSize={22} fontWeight={700} fill="#16a34a">{cm.tp}</text>
                <text x={MARGIN + CELL_SIZE / 2} y={MARGIN + CELL_SIZE / 2 + 12} textAnchor="middle" fontSize={11} fill="#15803d">True Positive</text>

                {/* FP - red */}
                <rect x={MARGIN + CELL_SIZE} y={MARGIN} width={CELL_SIZE} height={CELL_SIZE} fill="#fee2e2" stroke="#fca5a5" strokeWidth={1.5} rx={4} />
                <text x={MARGIN + CELL_SIZE * 1.5} y={MARGIN + CELL_SIZE / 2 - 8} textAnchor="middle" fontSize={22} fontWeight={700} fill="#dc2626">{cm.fp}</text>
                <text x={MARGIN + CELL_SIZE * 1.5} y={MARGIN + CELL_SIZE / 2 + 12} textAnchor="middle" fontSize={11} fill="#b91c1c">False Positive</text>

                {/* FN - orange */}
                <rect x={MARGIN} y={MARGIN + CELL_SIZE} width={CELL_SIZE} height={CELL_SIZE} fill="#ffedd5" stroke="#fdba74" strokeWidth={1.5} rx={4} />
                <text x={MARGIN + CELL_SIZE / 2} y={MARGIN + CELL_SIZE * 1.5 - 8} textAnchor="middle" fontSize={22} fontWeight={700} fill="#ea580c">{cm.fn}</text>
                <text x={MARGIN + CELL_SIZE / 2} y={MARGIN + CELL_SIZE * 1.5 + 12} textAnchor="middle" fontSize={11} fill="#c2410c">False Negative</text>

                {/* TN - blue */}
                <rect x={MARGIN + CELL_SIZE} y={MARGIN + CELL_SIZE} width={CELL_SIZE} height={CELL_SIZE} fill="#dbeafe" stroke="#93c5fd" strokeWidth={1.5} rx={4} />
                <text x={MARGIN + CELL_SIZE * 1.5} y={MARGIN + CELL_SIZE * 1.5 - 8} textAnchor="middle" fontSize={22} fontWeight={700} fill="#2563eb">{cm.tn}</text>
                <text x={MARGIN + CELL_SIZE * 1.5} y={MARGIN + CELL_SIZE * 1.5 + 12} textAnchor="middle" fontSize={11} fill="#1d4ed8">True Negative</text>
              </svg>
            </div>

            {/* Derived metrics */}
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'Sensitivity (Recall)', value: sensitivity, desc: 'TP / (TP + FN)' },
                { label: 'Specificity', value: specificity, desc: 'TN / (TN + FP)' },
                { label: 'PPV (Precision)', value: ppv, desc: 'TP / (TP + FP)' },
                { label: 'NPV', value: npv, desc: 'TN / (TN + FN)' },
              ].map(({ label, value, desc }) => (
                <div key={label} className="bg-slate-50 rounded-lg p-4">
                  <p className="text-xs text-slate-500 font-medium">{label}</p>
                  <p className="text-2xl font-bold text-slate-800 font-mono">
                    {value !== null ? `${(value * 100).toFixed(1)}%` : '—'}
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
